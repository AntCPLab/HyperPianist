// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Main module for the Lookup Check protocol

#![allow(non_snake_case)]
use crate::{
    pcs::PolynomialCommitmentScheme,
    poly_iop::{
        errors::PolyIOPErrors, sum_check::generic_sumcheck::SumcheckInstanceProof, PolyIOP,
    },
    split_bits, Commitment,
};
use arithmetic::{
    bit_decompose, build_eq_x_r, eq_eval, eq_poly::EqPolynomial, math::Math, OptimizedMul,
    VPAuxInfo,
};
use ark_ec::pairing::Pairing;
use ark_ff::{One, PrimeField};
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{end_timer, start_timer, Zero};
use instruction::{concatenate_lookups, evaluate_mle_dechunk_operands, JoltInstruction};
use logup_checking::{LogupChecking, LogupCheckingProof, LogupCheckingSubclaim};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use std::{iter::zip, marker::PhantomData, mem::take, sync::Arc};
use transcript::IOPTranscript;
use util::{polys_from_evals, polys_from_evals_usize, SurgeCommons};

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

pub mod instruction;

#[cfg(feature = "rational_sumcheck_piop")]
mod logup_checking_piop;

#[cfg(feature = "rational_sumcheck_piop")]
use logup_checking_piop as logup_checking;

#[cfg(not(feature = "rational_sumcheck_piop"))]
mod logup_checking;

mod subtable;
mod util;

pub trait LookupCheck<E, PCS, Instruction, const C: usize, const M: usize>:
    SurgeCommons<E::ScalarField, Instruction, C, M>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
    Instruction: JoltInstruction + Default,
{
    type LookupCheckSubClaim;
    type LookupCheckProof: CanonicalSerialize + CanonicalDeserialize;

    type Preprocessing;
    type WitnessPolys;
    type Polys;
    type MultilinearExtension;
    type Transcript;
    type ProveResult;
    type DistProveResult;

    /// Initialize the system with a transcript
    ///
    /// This function is optional -- in the case where a LookupCheck is
    /// an building block for a more complex protocol, the transcript
    /// may be initialized by this complex protocol, and passed to the
    /// LookupCheck prover/verifier.
    fn init_transcript() -> Self::Transcript;

    fn preprocess() -> Self::Preprocessing;

    fn construct_witnesses(ops: &[Instruction]) -> Self::WitnessPolys;

    fn construct_polys(
        preprocessing: &Self::Preprocessing,
        ops: &[Instruction],
        alpha: &E::ScalarField,
    ) -> Self::Polys;

    // Returns (proof, r_f, r_g, r_z, r_primary_sumcheck)
    fn prove(
        preprocessing: &Self::Preprocessing,
        pcs_param: &PCS::ProverParam,
        poly: &mut Self::Polys,
        alpha: &E::ScalarField,
        tau: &E::ScalarField,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::ProveResult, PolyIOPErrors>;

    fn d_prove(
        preprocessing: &Self::Preprocessing,
        pcs_param: &PCS::ProverParam,
        poly: &mut Self::Polys,
        alpha: &E::ScalarField,
        tau: &E::ScalarField,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::DistProveResult, PolyIOPErrors>;

    /// verify the claimed sum using the proof
    fn verify(
        proof: &Self::LookupCheckProof,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::LookupCheckSubClaim, PolyIOPErrors>;

    fn check_openings(
        subclaim: &Self::LookupCheckSubClaim,
        dim_openings: &[E::ScalarField],
        E_openings: &[E::ScalarField],
        m_openings: &[E::ScalarField],
        witness_openings: &[E::ScalarField],
        #[cfg(feature = "rational_sumcheck_piop")] f_inv_openings: &[E::ScalarField],
        #[cfg(feature = "rational_sumcheck_piop")] g_inv_openings: &[E::ScalarField],
        alpha: &E::ScalarField,
        tau: &E::ScalarField,
    ) -> Result<(), PolyIOPErrors>;
}

#[derive(Clone)]
pub struct SurgePreprocessing<F>
where
    F: PrimeField,
{
    pub materialized_subtables: Vec<Vec<F>>,
}

pub struct SurgePolysPrimary<E>
where
    E: Pairing,
{
    // These polynomials are constructed as big-endian
    // Since everything else treats polynomials as little-endian, r_f, r_g, and r_z
    // all have their indices reversed
    pub dim: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>, // Size C
    pub E_polys: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>, // Size NUM_MEMORIES
    pub m: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>,   // Size C

    // Sparse representation of m
    pub m_indices: Vec<Vec<usize>>,
    pub m_values: Vec<Vec<E::ScalarField>>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgeCommitmentPrimary<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub dim_commitment: Vec<PCS::Commitment>, // Size C
    pub E_commitment: Vec<PCS::Commitment>,   // Size NUM_MEMORIES
    pub m_commitment: Vec<PCS::Commitment>,   // Size C
}

impl<E> SurgePolysPrimary<E>
where
    E: Pairing,
{
    // #[tracing::instrument(skip_all, name = "SurgePolysPrimary::commit")]
    fn commit<PCS>(
        &self,
        pcs_params: &PCS::ProverParam,
    ) -> (
        SurgeCommitmentPrimary<E, PCS>,
        Vec<PCS::ProverCommitmentAdvice>,
    )
    where
        PCS: PolynomialCommitmentScheme<
            E,
            Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        >,
    {
        let ((E_commitment, E_advice), ((dim_commitment, dim_advice), (m_commitment, m_advice)))
        :  ((Vec<_>, Vec<_>), ((Vec<_>, Vec<_>), (Vec<_>, Vec<_>))) =
            rayon::join(|| 
                self.E_polys
                    .par_iter()
                    .map(|poly| PCS::commit(pcs_params, poly).unwrap())
                    .unzip(),
                    || rayon::join(
                        || {
                            self.dim
                                .par_iter()
                                .map(|poly| PCS::commit(pcs_params, poly).unwrap())
                                .unzip()
                        },
                        || {
                            self.m
                                .par_iter()
                                .map(|poly| PCS::commit(pcs_params, poly).unwrap())
                                .unzip()
                        },
                    )
            );

        (
            SurgeCommitmentPrimary {
                dim_commitment,
                E_commitment,
                m_commitment,
            },
            vec![dim_advice, E_advice, m_advice].concat(),
        )
    }

    fn d_commit<PCS>(
        &self,
        pcs_params: &PCS::ProverParam,
    ) -> (
        Option<SurgeCommitmentPrimary<E, PCS>>,
        Vec<PCS::ProverCommitmentAdvice>,
    )
    where
        PCS: PolynomialCommitmentScheme<
            E,
            Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        >,
    {
        let (mut dim_commitment, dim_advice): (Vec<_>, Vec<_>) = self
            .dim
            .iter()
            .map(|poly| PCS::d_commit(pcs_params, poly).unwrap())
            .unzip();

        let (mut E_commitment, E_advice): (Vec<_>, Vec<_>) = self
            .E_polys
            .iter()
            .map(|poly| PCS::d_commit(pcs_params, poly).unwrap())
            .unzip();

        let (mut m_commitment, m_advice): (Vec<_>, Vec<_>) = self
            .m
            .iter()
            .map(|poly| PCS::d_commit(pcs_params, poly).unwrap())
            .unzip();

        if Net::am_master() {
            (
                Some(SurgeCommitmentPrimary {
                    dim_commitment: dim_commitment
                        .iter_mut()
                        .map(|comm| take(comm).unwrap())
                        .collect(),
                    E_commitment: E_commitment
                        .iter_mut()
                        .map(|comm| take(comm).unwrap())
                        .collect(),
                    m_commitment: m_commitment
                        .iter_mut()
                        .map(|comm| take(comm).unwrap())
                        .collect(),
                }),
                vec![dim_advice, E_advice, m_advice].concat(),
            )
        } else {
            (None, vec![dim_advice, E_advice, m_advice].concat())
        }
    }

    pub fn collect_m_polys(&mut self) {
        // Combine the m polys to make a new one
        let all_m = Net::send_to_master(&(take(&mut self.m_indices), take(&mut self.m_values)));
        let num_party_vars = Net::n_parties().log_2();
        let num_m_vars = self.m[0].num_vars;

        let (mut m_evals, m_indices, m_values) = if Net::am_master() {
            let all_m = all_m.unwrap();
            self.m.par_iter_mut().enumerate().for_each(|(i, poly)| {
                for (indices, values) in all_m.iter().skip(1) {
                    for (index, value) in zip(&indices[i], &values[i]) {
                        Arc::get_mut(poly).unwrap().evaluations[*index] += *value;
                    }
                }
            });

            let len_per_party = self.m[0].evaluations.len() / Net::n_parties();
            let redistributed_m = (0..Net::n_parties())
                .into_par_iter()
                .map(|party_id| {
                    let m_evals = self
                        .m
                        .iter()
                        .map(|m| {
                            m.evaluations[party_id * len_per_party..(party_id + 1) * len_per_party]
                                .to_vec()
                        })
                        .collect::<Vec<_>>();
                    let (m_indices, m_values): (Vec<_>, Vec<_>) = m_evals
                        .par_iter()
                        .map(|m_evals_single| {
                            let mut indices = vec![];
                            let mut values = vec![];
                            for (i, m) in m_evals_single.iter().enumerate() {
                                if *m != E::ScalarField::zero() {
                                    // Do not add party offset here; each party indexes into its own
                                    // sub-m array
                                    indices.push(i);
                                    values.push(*m);
                                }
                            }
                            (indices, values)
                        })
                        .unzip();
                    (m_evals, m_indices, m_values)
                })
                .collect::<Vec<_>>();

            Net::recv_from_master(Some(redistributed_m))
        } else {
            Net::recv_from_master(None)
        };
        self.m = m_evals
            .iter_mut()
            .map(|m_evals| {
                Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                    num_m_vars - num_party_vars,
                    take(m_evals),
                ))
            })
            .collect();
        self.m_indices = m_indices;
        self.m_values = m_values;
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgePrimarySumcheck<F>
where
    F: PrimeField,
{
    sumcheck_proof: SumcheckInstanceProof<F>,
    num_rounds: usize,
    claimed_evaluation: F,
}

#[cfg(feature = "rational_sumcheck_piop")]
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct LookupCheckProof<E: Pairing, PCS: PolynomialCommitmentScheme<E>> {
    pub primary_sumcheck: SurgePrimarySumcheck<E::ScalarField>,
    pub logup_checking: LogupCheckingProof<E, PCS>,
    pub commitment: SurgeCommitmentPrimary<E, PCS>,
}

#[cfg(not(feature = "rational_sumcheck_piop"))]
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct LookupCheckProof<E: Pairing, PCS: PolynomialCommitmentScheme<E>> {
    pub primary_sumcheck: SurgePrimarySumcheck<E::ScalarField>,
    pub logup_checking: LogupCheckingProof<E::ScalarField>,
    pub commitment: SurgeCommitmentPrimary<E, PCS>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LookupCheckSubClaim<F: PrimeField> {
    pub r_primary_sumcheck: Vec<F>, // Challenge used as eq polynomial parameter
    pub primary_sumcheck_claim: F,

    // Primary sumcheck subclaim. Note this differs from r_primary_sumcheck
    pub r_z: Vec<F>,
    pub primary_sumcheck_expected_evaluation: F,

    pub logup_checking: LogupCheckingSubclaim<F>,
}

impl<E, PCS, Instruction, const C: usize, const M: usize> LookupCheck<E, PCS, Instruction, C, M>
    for PolyIOP<E::ScalarField>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E, Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>>,
    PCS::Commitment: Send,
    Instruction: JoltInstruction + Default,
{
    type LookupCheckSubClaim = LookupCheckSubClaim<E::ScalarField>;
    type LookupCheckProof = LookupCheckProof<E, PCS>;

    type Preprocessing = SurgePreprocessing<E::ScalarField>;
    type WitnessPolys = Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>;
    type Polys = SurgePolysPrimary<E>;
    type MultilinearExtension = Arc<DenseMultilinearExtension<E::ScalarField>>;
    type Transcript = IOPTranscript<E::ScalarField>;

    #[cfg(feature = "rational_sumcheck_piop")]
    type ProveResult = (
        Self::LookupCheckProof,
        Vec<PCS::ProverCommitmentAdvice>,
        Vec<E::ScalarField>,
        Vec<E::ScalarField>,
        Vec<E::ScalarField>,
        Vec<E::ScalarField>,
        Vec<Self::MultilinearExtension>,
        Vec<Self::MultilinearExtension>,
    );

    #[cfg(feature = "rational_sumcheck_piop")]
    type DistProveResult = (
        Option<Self::LookupCheckProof>,
        Vec<PCS::ProverCommitmentAdvice>,
        Vec<E::ScalarField>,
        Vec<E::ScalarField>,
        Vec<E::ScalarField>,
        Vec<E::ScalarField>,
        Vec<Self::MultilinearExtension>,
        Vec<Self::MultilinearExtension>,
    );

    #[cfg(not(feature = "rational_sumcheck_piop"))]
    type ProveResult = (
        Self::LookupCheckProof,
        Vec<PCS::ProverCommitmentAdvice>,
        Vec<E::ScalarField>,
        Vec<E::ScalarField>,
        Vec<E::ScalarField>,
        Vec<E::ScalarField>,
    );

    #[cfg(not(feature = "rational_sumcheck_piop"))]
    type DistProveResult = (
        Option<Self::LookupCheckProof>,
        Vec<PCS::ProverCommitmentAdvice>,
        Vec<E::ScalarField>,
        Vec<E::ScalarField>,
        Vec<E::ScalarField>,
        Vec<E::ScalarField>,
    );

    fn init_transcript() -> Self::Transcript {
        IOPTranscript::<E::ScalarField>::new(b"Initializing LookupCheck transcript")
    }

    fn preprocess() -> Self::Preprocessing {
        let instruction = Instruction::default();

        let materialized_subtables = instruction
            .subtables(C, M)
            .par_iter()
            .map(|(subtable, _)| subtable.materialize(M))
            .collect();

        Self::Preprocessing {
            materialized_subtables,
        }
    }

    fn construct_witnesses(ops: &[Instruction]) -> Self::WitnessPolys {
        let num_lookups = ops.len().next_power_of_two();
        let log_m = ark_std::log2(num_lookups) as usize;

        // aka operands
        let mut witness_evals = vec![vec![0usize; num_lookups]; 3];
        for (op_index, op) in ops.iter().enumerate() {
            let (operand_x, operand_y) = op.operands();
            witness_evals[0][op_index] = operand_x as usize;
            witness_evals[1][op_index] = operand_y as usize;
            witness_evals[2][op_index] = op.lookup_entry() as usize;
        }

        polys_from_evals_usize(log_m, &witness_evals)
    }

    // #[tracing::instrument(skip_all, name = "Surge::construct_polys")]
    fn construct_polys(
        preprocessing: &Self::Preprocessing,
        ops: &[Instruction],
        alpha: &E::ScalarField,
    ) -> Self::Polys {
        let num_memories =
            <Self as SurgeCommons<E::ScalarField, Instruction, C, M>>::num_memories();
        let num_lookups = ops.len().next_power_of_two();
        let log_m = ark_std::log2(num_lookups) as usize;

        // Construct dim, m
        let mut dim_usize: Vec<Vec<usize>> = vec![vec![0; num_lookups]; C];

        let mut m_evals = vec![vec![0usize; M]; C];
        let log_M = ark_std::log2(M) as usize;
        let bits_per_operand = log_M / 2;

        for (op_index, op) in ops.iter().enumerate() {
            let access_sequence = op.to_indices(C, log_M);
            assert_eq!(access_sequence.len(), C);

            for dimension_index in 0..C {
                let memory_address = access_sequence[dimension_index];
                debug_assert!(memory_address < M);

                dim_usize[dimension_index][op_index] = memory_address;
                m_evals[dimension_index][memory_address] += 1;
            }
        }

        // num_ops is padded to the nearest power of 2 for the usage of DensePolynomial.
        // We cannot just fill in zeros for m_evals as this implicitly specifies
        // a read at address 0.
        for _fake_ops_index in ops.len()..num_lookups {
            for dimension_index in 0..C {
                let memory_address = 0;
                m_evals[dimension_index][memory_address] += 1;
            }
        }

        let mut m_indices = vec![];
        let mut m_values = vec![];
        let mut dim_poly = vec![];
        let mut m_poly = vec![];
        let mut E_poly = vec![];
        rayon::scope(|s| {
            s.spawn(|_| {
                (m_indices, m_values) = m_evals
                    .iter()
                    .map(|m_evals_it| {
                        let mut indices = vec![];
                        let mut values = vec![];
                        for (i, m) in m_evals_it.iter().enumerate() {
                            if *m != 0 {
                                indices.push(i);
                                values.push(E::ScalarField::from_u64(*m as u64).unwrap());
                            }
                        }
                        (indices, values)
                    })
                    .unzip();
            });
            s.spawn(|_| {
                dim_poly = polys_from_evals_usize(log_m, &dim_usize);
            });
            s.spawn(|_| {
                m_poly = polys_from_evals_usize(log_M, &m_evals);
            });
            s.spawn(|_| {
                // Construct E
                let mut E_i_evals = Vec::with_capacity(num_memories);
                for E_index in 0..num_memories {
                    let mut E_evals = Vec::with_capacity(num_lookups);
                    for op_index in 0..num_lookups {
                        let dimension_index = <Self as SurgeCommons<
                            E::ScalarField,
                            Instruction,
                            C,
                            M,
                        >>::memory_to_dimension_index(
                            E_index
                        );
                        let subtable_index = <Self as SurgeCommons<
                            E::ScalarField,
                            Instruction,
                            C,
                            M,
                        >>::memory_to_subtable_index(
                            E_index
                        );

                        let eval_index = dim_usize[dimension_index][op_index];
                        let E_eval = if subtable_index >= preprocessing.materialized_subtables.len()
                        {
                            let (x, y) = split_bits(eval_index, bits_per_operand);
                            E::ScalarField::from_u64(x as u64).unwrap()
                                + *alpha * E::ScalarField::from_u64(y as u64).unwrap()
                        } else {
                            preprocessing.materialized_subtables[subtable_index][eval_index]
                        };
                        E_evals.push(E_eval);
                    }
                    E_i_evals.push(E_evals);
                }
                E_poly = polys_from_evals(log_m, &E_i_evals);
            });
        });

        SurgePolysPrimary {
            dim: dim_poly,
            E_polys: E_poly,
            m: m_poly,
            m_indices,
            m_values,
        }
    }

    fn prove(
        preprocessing: &Self::Preprocessing,
        pcs_param: &PCS::ProverParam,
        poly: &mut Self::Polys,
        alpha: &E::ScalarField,
        tau: &E::ScalarField,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::ProveResult, PolyIOPErrors> {
        let start = start_timer!(|| "lookup_check prove");

        let (commitment, mut advices) = poly.commit(pcs_param);
        transcript.append_serializable_element(b"primary_commitment", &commitment)?;

        let num_rounds = poly.dim[0].num_vars;
        let instruction = Instruction::default();

        // TODO(sragss): Commit some of this stuff to transcript?

        // Primary sumcheck
        let mut r_primary_sumcheck =
            transcript.get_and_append_challenge_vectors(b"primary_sumcheck", num_rounds)?;
        let eq = DenseMultilinearExtension::from_evaluations_vec(
            r_primary_sumcheck.len(),
            EqPolynomial::evals(&r_primary_sumcheck),
        );
        r_primary_sumcheck.reverse();

        let log_M = ark_std::log2(M) as usize;

        let result = rayon::join(
            || {
                let mut transcript = transcript.clone();

                let step = start_timer!(|| "compute sumcheck claim");
                let sumcheck_claim = {
                    let hypercube_size = poly.E_polys[0].evaluations.len();
                    poly.E_polys
                        .iter()
                        .for_each(|operand| assert_eq!(operand.evaluations.len(), hypercube_size));

                    let instruction = Instruction::default();

                    (0..hypercube_size)
                        .into_par_iter()
                        .map(|eval_index| {
                            let g_operands: Vec<E::ScalarField> = (0..<Self as SurgeCommons<
                                E::ScalarField,
                                Instruction,
                                C,
                                M,
                            >>::num_memories(
                            ))
                                .map(|memory_index| poly.E_polys[memory_index][eval_index])
                                .collect();

                            let vals: &[E::ScalarField] = &g_operands[0..(g_operands.len() - C)];
                            let fingerprints = &g_operands[g_operands.len() - C..];
                            eq[eval_index]
                                * (instruction.combine_lookups(vals, C, M)
                                    + *alpha * concatenate_lookups(fingerprints, C, log_M / 2)
                                    + *tau)
                        })
                        .sum()
                };
                end_timer!(step);
                transcript
                    .append_field_element(b"sumcheck_claim", &sumcheck_claim)
                    .unwrap();

                let mut combined_sumcheck_polys = poly
                    .E_polys
                    .iter()
                    .map(|it| Arc::new((**it).clone()))
                    .collect::<Vec<_>>();
                combined_sumcheck_polys.push(Arc::new(eq));

                let combine_lookups_eq = |vals: &[E::ScalarField]| -> E::ScalarField {
                    let vals_no_eq: &[E::ScalarField] = &vals[0..(vals.len() - 1 - C)];
                    let fingerprints = &vals[vals.len() - 1 - C..vals.len() - 1];
                    let eq = vals[vals.len() - 1];
                    (instruction.combine_lookups(vals_no_eq, C, M)
                        + *alpha * concatenate_lookups(fingerprints, C, log_M / 2)
                        + *tau)
                        * eq
                };

                let step = start_timer!(|| "primary sumcheck");
                let (primary_sumcheck_proof, r_z, _) =
                    SumcheckInstanceProof::<E::ScalarField>::prove_arbitrary::<_>(
                        &sumcheck_claim,
                        num_rounds,
                        &mut combined_sumcheck_polys,
                        combine_lookups_eq,
                        instruction.g_poly_degree(C) + 1, // combined degree + eq term
                        &mut transcript,
                    );
                end_timer!(step);

                (
                    SurgePrimarySumcheck {
                        sumcheck_proof: primary_sumcheck_proof,
                        num_rounds,
                        claimed_evaluation: sumcheck_claim,
                    },
                    r_z,
                )
            },
            || {
                let mut transcript = transcript.clone();

                let step = start_timer!(|| "logup checking");
                #[cfg(feature = "rational_sumcheck_piop")]
                let (logup_checking, mut logup_advices, f_inv, g_inv) =
                    <Self as LogupChecking<E, PCS, Instruction, C, M>>::prove_logup_checking(
                        pcs_param,
                        preprocessing,
                        poly,
                        alpha,
                        &mut transcript,
                    )
                    .unwrap();

                #[cfg(feature = "rational_sumcheck_piop")]
                advices.append(&mut logup_advices);

                #[cfg(feature = "rational_sumcheck_piop")]
                let (r_f, r_g) = (
                    logup_checking.f_proof.sum_check_proof.point.clone(),
                    logup_checking.g_proof.sum_check_proof.point.clone(),
                );

                #[cfg(not(feature = "rational_sumcheck_piop"))]
                let (logup_checking, r_f, r_g) =
                    <Self as LogupChecking<E, PCS, Instruction, C, M>>::prove_logup_checking(
                        preprocessing,
                        poly,
                        alpha,
                        &mut transcript,
                    )
                    .unwrap();

                end_timer!(step);

                #[cfg(feature = "rational_sumcheck_piop")]
                return (logup_checking, r_f, r_g, f_inv, g_inv);

                #[cfg(not(feature = "rational_sumcheck_piop"))]
                (logup_checking, r_f, r_g)
            },
        );

        #[cfg(feature = "rational_sumcheck_piop")]
        let ((primary_sumcheck, r_z), (logup_checking, r_f, r_g, f_inv, g_inv)) = result;

        #[cfg(not(feature = "rational_sumcheck_piop"))]
        let ((primary_sumcheck, r_z), (logup_checking, r_f, r_g)) = result;

        end_timer!(start);

        Ok((
            LookupCheckProof {
                primary_sumcheck,
                logup_checking,
                commitment,
            },
            advices,
            r_f,
            r_g,
            r_z,
            r_primary_sumcheck,
            #[cfg(feature = "rational_sumcheck_piop")]
            f_inv,
            #[cfg(feature = "rational_sumcheck_piop")]
            g_inv,
        ))
    }

    fn d_prove(
        preprocessing: &Self::Preprocessing,
        pcs_param: &PCS::ProverParam,
        poly: &mut Self::Polys,
        alpha: &E::ScalarField,
        tau: &E::ScalarField,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::DistProveResult, PolyIOPErrors> {
        let start = start_timer!(|| "lookup_check prove");

        let (commitment, mut advices) = poly.d_commit(pcs_param);
        if Net::am_master() {
            transcript
                .append_serializable_element(b"primary_commitment", commitment.as_ref().unwrap())?;
        }

        let num_rounds = poly.dim[0].num_vars;
        let instruction = Instruction::default();

        // TODO(sragss): Commit some of this stuff to transcript?

        // Primary sumcheck
        let num_party_vars = Net::n_parties().log_2();

        let mut r_primary_sumcheck = if Net::am_master() {
            let r = transcript.get_and_append_challenge_vectors(
                b"primary_sumcheck",
                num_rounds + num_party_vars,
            )?;
            Net::recv_from_master_uniform(Some(r))
        } else {
            Net::recv_from_master_uniform(None)
        };

        r_primary_sumcheck.reverse();

        let index_vec: Vec<E::ScalarField> = bit_decompose(Net::party_id() as u64, num_party_vars)
            .into_iter()
            .map(|x| E::ScalarField::from(x))
            .collect();

        let coeff = eq_eval(&r_primary_sumcheck[num_rounds..], &index_vec)?;
        let mut eq = build_eq_x_r(&r_primary_sumcheck[..num_rounds])?;
        Arc::get_mut(&mut eq)
            .unwrap()
            .evaluations
            .par_iter_mut()
            .for_each(|val| *val *= coeff);

        let log_M = ark_std::log2(M) as usize;

        let (sumcheck_ret, sumcheck_claim) = {
            let mut transcript = transcript.clone();

            let mut sumcheck_claim = {
                let hypercube_size = poly.E_polys[0].evaluations.len();
                poly.E_polys
                    .iter()
                    .for_each(|operand| assert_eq!(operand.evaluations.len(), hypercube_size));

                let instruction = Instruction::default();

                (0..hypercube_size)
                    .into_par_iter()
                    .map(|eval_index| {
                        let g_operands: Vec<E::ScalarField> = (0..<Self as SurgeCommons<
                            E::ScalarField,
                            Instruction,
                            C,
                            M,
                        >>::num_memories(
                        ))
                            .map(|memory_index| poly.E_polys[memory_index][eval_index])
                            .collect();

                        let vals: &[E::ScalarField] = &g_operands[0..(g_operands.len() - C)];
                        let fingerprints = &g_operands[g_operands.len() - C..];
                        eq[eval_index]
                            * (instruction.combine_lookups(vals, C, M)
                                + *alpha * concatenate_lookups(fingerprints, C, log_M / 2)
                                + *tau)
                    })
                    .sum()
            };
            let all_claims = Net::send_to_master(&sumcheck_claim);
            if Net::am_master() {
                sumcheck_claim = all_claims.unwrap().iter().sum();
                transcript.append_field_element(b"sumcheck_claim", &sumcheck_claim)?;
            }

            let mut combined_sumcheck_polys = poly
                .E_polys
                .iter()
                .map(|it| Arc::new((**it).clone()))
                .collect::<Vec<_>>();
            combined_sumcheck_polys.push(eq);

            let combine_lookups_eq = |vals: &[E::ScalarField]| -> E::ScalarField {
                let vals_no_eq: &[E::ScalarField] = &vals[0..(vals.len() - 1 - C)];
                let fingerprints = &vals[vals.len() - 1 - C..vals.len() - 1];
                let eq = vals[vals.len() - 1];
                (instruction.combine_lookups(vals_no_eq, C, M)
                    + *alpha * concatenate_lookups(fingerprints, C, log_M / 2)
                    + *tau)
                    * eq
            };

            (
                SumcheckInstanceProof::<E::ScalarField>::d_prove_arbitrary::<_>(
                    &sumcheck_claim,
                    num_rounds,
                    &mut combined_sumcheck_polys,
                    combine_lookups_eq,
                    instruction.g_poly_degree(C) + 1, // combined degree + eq term
                    &mut transcript,
                ),
                sumcheck_claim,
            )
        };

        let logup_ret = {
            let mut transcript = transcript.clone();

            #[cfg(feature = "rational_sumcheck_piop")]
            let mut proof_ret =
                <Self as LogupChecking<E, PCS, Instruction, C, M>>::d_prove_logup_checking(
                    pcs_param,
                    preprocessing,
                    poly,
                    alpha,
                    &mut transcript,
                )
                .unwrap();

            #[cfg(feature = "rational_sumcheck_piop")]
            advices.append(&mut proof_ret.1);

            #[cfg(not(feature = "rational_sumcheck_piop"))]
            let proof_ret =
                <Self as LogupChecking<E, PCS, Instruction, C, M>>::d_prove_logup_checking(
                    preprocessing,
                    poly,
                    alpha,
                    &mut transcript,
                )
                .unwrap();

            proof_ret
        };

        #[cfg(feature = "rational_sumcheck_piop")]
        let (logup_checking, _, f_inv, g_inv) = logup_ret;

        end_timer!(start);

        if Net::am_master() {
            let (primary_sumcheck_proof, r_z, _) = sumcheck_ret.unwrap();

            #[cfg(feature = "rational_sumcheck_piop")]
            {
                let logup_checking = logup_checking.unwrap();
                let (r_f, r_g) = (
                    logup_checking.f_proof.sum_check_proof.point.clone(),
                    logup_checking.g_proof.sum_check_proof.point.clone(),
                );
                Net::recv_from_master_uniform(Some((r_f.clone(), r_g.clone(), r_z.clone())));
                Ok((
                    Some(LookupCheckProof {
                        primary_sumcheck: SurgePrimarySumcheck {
                            sumcheck_proof: primary_sumcheck_proof,
                            num_rounds: num_rounds + num_party_vars,
                            claimed_evaluation: sumcheck_claim,
                        },
                        logup_checking,
                        commitment: commitment.unwrap(),
                    }),
                    advices,
                    r_f,
                    r_g,
                    r_z,
                    r_primary_sumcheck,
                    f_inv,
                    g_inv,
                ))
            }

            #[cfg(not(feature = "rational_sumcheck_piop"))]
            {
                let (logup_checking, r_f, r_g) = logup_ret.unwrap();
                Net::recv_from_master_uniform(Some((r_f.clone(), r_g.clone(), r_z.clone())));

                Ok((
                    Some(LookupCheckProof {
                        primary_sumcheck: SurgePrimarySumcheck {
                            sumcheck_proof: primary_sumcheck_proof,
                            num_rounds: num_rounds + num_party_vars,
                            claimed_evaluation: sumcheck_claim,
                        },
                        logup_checking,
                        commitment: commitment.unwrap(),
                    }),
                    advices,
                    r_f,
                    r_g,
                    r_z,
                    r_primary_sumcheck,
                ))
            }
        } else {
            let (r_f, r_g, r_z) = Net::recv_from_master_uniform(None);

            #[cfg(feature = "rational_sumcheck_piop")]
            return Ok((
                None,
                advices,
                r_f,
                r_g,
                r_z,
                r_primary_sumcheck,
                f_inv,
                g_inv,
            ));

            #[cfg(not(feature = "rational_sumcheck_piop"))]
            return Ok((None, advices, r_f, r_g, r_z, r_primary_sumcheck));
        }
    }

    fn verify(
        proof: &Self::LookupCheckProof,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::LookupCheckSubClaim, PolyIOPErrors> {
        let start = start_timer!(|| "lookup_check verify");

        transcript.append_serializable_element(b"primary_commitment", &proof.commitment)?;

        let instruction = Instruction::default();

        let mut r_primary_sumcheck = transcript.get_and_append_challenge_vectors(
            b"primary_sumcheck",
            proof.primary_sumcheck.num_rounds,
        )?;
        r_primary_sumcheck.reverse();

        let ((claim_last, r_z), logup_checking) = rayon::join(
            || {
                let mut transcript = transcript.clone();
                transcript
                    .append_field_element(
                        b"sumcheck_claim",
                        &proof.primary_sumcheck.claimed_evaluation,
                    )
                    .unwrap();
                let primary_sumcheck_poly_degree = instruction.g_poly_degree(C) + 1;
                let step = start_timer!(|| "primary sumcheck verify");
                let (claim_last, r_z) = proof
                    .primary_sumcheck
                    .sumcheck_proof
                    .verify(
                        proof.primary_sumcheck.claimed_evaluation,
                        proof.primary_sumcheck.num_rounds,
                        primary_sumcheck_poly_degree,
                        &mut transcript,
                    )
                    .unwrap();
                end_timer!(step);
                (claim_last, r_z)
            },
            || {
                let mut transcript = transcript.clone();
                let step = start_timer!(|| "logup checking verify");
                let logup_checking =
                    <Self as LogupChecking<E, PCS, Instruction, C, M>>::verify_logup_checking(
                        &proof.logup_checking,
                        #[cfg(feature = "rational_sumcheck_piop")]
                        &VPAuxInfo {
                            max_degree: 3,
                            num_variables: ark_std::log2(M) as usize,
                            phantom: PhantomData,
                        },
                        #[cfg(feature = "rational_sumcheck_piop")]
                        &VPAuxInfo {
                            max_degree: 3,
                            num_variables: r_primary_sumcheck.len(),
                            phantom: PhantomData,
                        },
                        &mut transcript,
                    )
                    .unwrap();
                end_timer!(step);
                logup_checking
            },
        );

        end_timer!(start);
        Ok(LookupCheckSubClaim {
            r_primary_sumcheck,
            primary_sumcheck_claim: proof.primary_sumcheck.claimed_evaluation,
            r_z,
            primary_sumcheck_expected_evaluation: claim_last,
            logup_checking,
        })
    }

    // m opened to r_f
    // dim opened to r_g
    // E opened to r_g & r_z
    // f_inv: r_f, g_inv: r_g (if applicable)
    // witness opened to r_primary_sumcheck
    fn check_openings(
        subclaim: &Self::LookupCheckSubClaim,
        dim_openings: &[E::ScalarField],
        E_openings: &[E::ScalarField],
        m_openings: &[E::ScalarField],
        witness_openings: &[E::ScalarField],
        #[cfg(feature = "rational_sumcheck_piop")] f_inv_openings: &[E::ScalarField],
        #[cfg(feature = "rational_sumcheck_piop")] g_inv_openings: &[E::ScalarField],
        alpha: &E::ScalarField,
        tau: &E::ScalarField,
    ) -> Result<(), PolyIOPErrors> {
        let (beta, gamma) = subclaim.logup_checking.challenges;

        let num_memories =
            <Self as SurgeCommons<E::ScalarField, Instruction, C, M>>::num_memories();

        let mut f_ok = false;
        let mut g_ok = false;
        let mut primary_ok = false;
        let mut witness_ok = false;
        rayon::scope(|s| {
            #[cfg(feature = "rational_sumcheck_piop")]
            {
                s.spawn(|_| {
                    let mut r_f = subclaim
                        .logup_checking
                        .f_subclaims
                        .sum_check_sub_claim
                        .point
                        .clone();
                    r_f.reverse();

                    let sid: E::ScalarField = (0..r_f.len())
                        .map(|i| {
                            E::ScalarField::from_u64((r_f.len() - i - 1).pow2() as u64).unwrap()
                                * r_f[i]
                        })
                        .sum();
                    let mut t = Instruction::default()
                        .subtables(C, M)
                        .par_iter()
                        .map(|(subtable, _)| subtable.evaluate_mle(&r_f))
                        .collect::<Vec<_>>();
                    t.push(evaluate_mle_dechunk_operands(&r_f, *alpha));

                    let eq_eval = eq_eval(
                        &subclaim
                            .logup_checking
                            .f_subclaims
                            .sum_check_sub_claim
                            .point,
                        &subclaim.logup_checking.f_subclaims.zerocheck_r,
                    )
                    .unwrap();

                    f_ok = (0..num_memories)
                        .into_par_iter()
                        .map(|i| {
                            let dim_idx = <Self as SurgeCommons<
                                E::ScalarField,
                                Instruction,
                                C,
                                M,
                            >>::memory_to_dimension_index(
                                i
                            );
                            let subtable_idx = <Self as SurgeCommons<
                                E::ScalarField,
                                Instruction,
                                C,
                                M,
                            >>::memory_to_subtable_index(
                                i
                            );

                            subclaim.logup_checking.f_subclaims.coeffs[i]
                                * ((beta + sid + t[subtable_idx].mul_01_optimized(gamma))
                                    * f_inv_openings[subtable_idx]
                                    - E::ScalarField::one())
                                * eq_eval
                                + subclaim.logup_checking.f_subclaims.coeffs[num_memories + i]
                                    * m_openings[dim_idx]
                                    * f_inv_openings[subtable_idx]
                        })
                        .sum::<E::ScalarField>()
                        == subclaim
                            .logup_checking
                            .f_subclaims
                            .sum_check_sub_claim
                            .expected_evaluation;
                });
                s.spawn(|_| {
                    let eq_eval = eq_eval(
                        &subclaim
                            .logup_checking
                            .g_subclaims
                            .sum_check_sub_claim
                            .point,
                        &subclaim.logup_checking.g_subclaims.zerocheck_r,
                    )
                    .unwrap();

                    g_ok = (0..num_memories)
                        .into_par_iter()
                        .map(|i| {
                            let dim_idx = <Self as SurgeCommons<
                                E::ScalarField,
                                Instruction,
                                C,
                                M,
                            >>::memory_to_dimension_index(
                                i
                            );
                            subclaim.logup_checking.g_subclaims.coeffs[i]
                                * ((beta + dim_openings[dim_idx] + E_openings[i] * gamma)
                                    * g_inv_openings[i]
                                    - E::ScalarField::one())
                                * eq_eval
                                + subclaim.logup_checking.g_subclaims.coeffs[num_memories + i]
                                    * g_inv_openings[i]
                        })
                        .sum::<E::ScalarField>()
                        == subclaim
                            .logup_checking
                            .g_subclaims
                            .sum_check_sub_claim
                            .expected_evaluation;
                });
            }

            #[cfg(not(feature = "rational_sumcheck_piop"))]
            {
                s.spawn(|_| {
                    let mut r_f = subclaim.logup_checking.point_f.clone();
                    r_f.reverse();

                    let sid: E::ScalarField = (0..r_f.len())
                        .map(|i| {
                            E::ScalarField::from_u64((r_f.len() - i - 1).pow2() as u64).unwrap()
                                * r_f[i]
                        })
                        .sum();
                    let mut t = Instruction::default()
                        .subtables(C, M)
                        .par_iter()
                        .map(|(subtable, _)| subtable.evaluate_mle(&r_f))
                        .collect::<Vec<_>>();
                    t.push(evaluate_mle_dechunk_operands(&r_f, *alpha));

                    f_ok = subclaim
                        .logup_checking
                        .expected_evaluations_f
                        .par_iter()
                        .enumerate()
                        .all(|(i, claim)| {
                            let dim_idx = <Self as SurgeCommons<
                                E::ScalarField,
                                Instruction,
                                C,
                                M,
                            >>::memory_to_dimension_index(
                                i
                            );
                            let subtable_idx = <Self as SurgeCommons<
                                E::ScalarField,
                                Instruction,
                                C,
                                M,
                            >>::memory_to_subtable_index(
                                i
                            );

                            if claim.p != m_openings[dim_idx] {
                                return false;
                            }
                            claim.q == beta + sid + t[subtable_idx].mul_01_optimized(gamma)
                        });
                });
                s.spawn(|_| {
                    g_ok = subclaim
                        .logup_checking
                        .expected_evaluations_g
                        .par_iter()
                        .enumerate()
                        .all(|(i, claim)| {
                            let dim_idx = <Self as SurgeCommons<
                                E::ScalarField,
                                Instruction,
                                C,
                                M,
                            >>::memory_to_dimension_index(
                                i
                            );

                            claim.p == E::ScalarField::one()
                                && claim.q == beta + dim_openings[dim_idx] + E_openings[i] * gamma
                        });
                });
            }
            s.spawn(|_| {
                let instruction = Instruction::default();

                // Remaining part is for primary opening
                let vals = &E_openings[num_memories..];
                let log_M = ark_std::log2(M) as usize;

                let vals_no_eq: &[E::ScalarField] = &vals[0..(vals.len() - C)];
                let fingerprints = &vals[vals.len() - C..];

                let eq = eq_eval(&subclaim.r_primary_sumcheck, &subclaim.r_z);
                primary_ok = if let Ok(eq_val) = eq {
                    subclaim.primary_sumcheck_expected_evaluation
                        == (instruction.combine_lookups(vals_no_eq, C, M)
                            + *alpha * concatenate_lookups(fingerprints, C, log_M / 2)
                            + *tau)
                            * eq_val
                } else {
                    false
                }
            });
            s.spawn(|_| {
                witness_ok = subclaim.primary_sumcheck_claim
                    == witness_openings[2]
                        + *alpha * (witness_openings[0] + *alpha * witness_openings[1])
                        + *tau;
            })
        });
        if f_ok && g_ok && primary_ok && witness_ok {
            Ok(())
        } else {
            Err(PolyIOPErrors::InvalidProof(format!(
                "wrong subclaim w/ check openings"
            )))
        }
    }
}

#[cfg(test)]
mod test {
    use arithmetic::evaluate_opt;
    use ark_bls12_381::Bls12_381;
    use ark_ff::UniformRand;
    use transcript::IOPTranscript;

    use super::{
        instruction::{xor::XORInstruction, JoltInstruction},
        LookupCheck,
    };
    use crate::{pcs::PolynomialCommitmentScheme, MultilinearKzgPCS, PolyIOP};
    use ark_ec::pairing::Pairing;
    use ark_std::test_rng;

    fn test_helper<
        E: Pairing,
        Instruction: JoltInstruction + Default,
        const C: usize,
        const M: usize,
    >(
        ops: &[Instruction],
    ) {
        let mut transcript = IOPTranscript::new(b"test_transcript");
        let preprocessing = <PolyIOP<E::ScalarField> as LookupCheck<
            E,
            MultilinearKzgPCS<E>,
            Instruction,
            C,
            M,
        >>::preprocess();

        let mut rng = test_rng();
        let srs = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 10).unwrap();
        let (pcs_param, _) = MultilinearKzgPCS::<E>::trim(&srs, None, Some(10)).unwrap();

        let alpha = E::ScalarField::rand(&mut rng);
        let tau = E::ScalarField::rand(&mut rng);

        let witnesses = <PolyIOP<E::ScalarField> as LookupCheck<
            E,
            MultilinearKzgPCS<E>,
            Instruction,
            C,
            M,
        >>::construct_witnesses(ops);
        let mut poly = <PolyIOP<E::ScalarField> as LookupCheck<
            E,
            MultilinearKzgPCS<E>,
            Instruction,
            C,
            M,
        >>::construct_polys(&preprocessing, ops, &alpha);

        #[cfg(feature = "rational_sumcheck_piop")]
        let (proof, _advices, r_f, r_g, r_z, r_primary_sumcheck, f_inv, g_inv) = <PolyIOP<
            E::ScalarField,
        > as LookupCheck<
            E,
            MultilinearKzgPCS<E>,
            Instruction,
            C,
            M,
        >>::prove(
            &preprocessing,
            &pcs_param,
            &mut poly,
            &alpha,
            &tau,
            &mut transcript,
        )
        .unwrap();

        #[cfg(not(feature = "rational_sumcheck_piop"))]
        let (proof, _advices, r_f, r_g, r_z, r_primary_sumcheck) = <PolyIOP<E::ScalarField> as LookupCheck<
            E,
            MultilinearKzgPCS<E>,
            Instruction,
            C,
            M,
        >>::prove(
            &preprocessing,
            &pcs_param,
            &mut poly,
            &alpha,
            &tau,
            &mut transcript,
        )
        .unwrap();

        let mut transcript = IOPTranscript::new(b"test_transcript");
        let subclaim = <PolyIOP<E::ScalarField> as LookupCheck<
            E,
            MultilinearKzgPCS<E>,
            Instruction,
            C,
            M,
        >>::verify(&proof, &mut transcript)
        .unwrap();

        assert_eq!(subclaim.r_primary_sumcheck, r_primary_sumcheck);
        assert_eq!(subclaim.r_z, r_z);
        #[cfg(feature = "rational_sumcheck_piop")]
        {
            assert_eq!(
                subclaim
                    .logup_checking
                    .f_subclaims
                    .sum_check_sub_claim
                    .point,
                r_f
            );
            assert_eq!(
                subclaim
                    .logup_checking
                    .g_subclaims
                    .sum_check_sub_claim
                    .point,
                r_g
            );
        }
        #[cfg(not(feature = "rational_sumcheck_piop"))]
        {
            assert_eq!(subclaim.logup_checking.point_f, r_f);
            assert_eq!(subclaim.logup_checking.point_g, r_g);
        }

        let m_openings = poly
            .m
            .iter()
            .map(|poly| evaluate_opt(poly, &r_f))
            .collect::<Vec<_>>();
        let dim_openings = poly
            .dim
            .iter()
            .map(|poly| evaluate_opt(poly, &r_g))
            .collect::<Vec<_>>();
        let E_openings = poly
            .E_polys
            .iter()
            .map(|poly| evaluate_opt(poly, &r_g))
            .chain(poly.E_polys.iter().map(|poly| evaluate_opt(poly, &r_z)))
            .collect::<Vec<_>>();
        let witness_openings = witnesses
            .iter()
            .map(|poly| evaluate_opt(poly, &r_primary_sumcheck))
            .collect::<Vec<_>>();
        #[cfg(feature = "rational_sumcheck_piop")]
        {
            let f_inv_openings = f_inv
                .iter()
                .map(|poly| evaluate_opt(poly, &r_f))
                .collect::<Vec<_>>();
            let g_inv_openings = g_inv
                .iter()
                .map(|poly| evaluate_opt(poly, &r_g))
                .collect::<Vec<_>>();

            <PolyIOP<E::ScalarField> as LookupCheck<E, MultilinearKzgPCS<E>, Instruction, C, M>>::check_openings(&subclaim, &dim_openings, &E_openings, &m_openings, &witness_openings, &f_inv_openings, &g_inv_openings, &alpha, &tau).unwrap();
        }

        #[cfg(not(feature = "rational_sumcheck_piop"))]
        <PolyIOP<E::ScalarField> as LookupCheck<E, MultilinearKzgPCS<E>, Instruction, C, M>>::check_openings(&subclaim, &dim_openings, &E_openings, &m_openings, &witness_openings, &alpha, &tau).unwrap();
    }

    #[test]
    fn e2e() {
        let ops = vec![
            XORInstruction(12, 12),
            XORInstruction(12, 82),
            XORInstruction(12, 12),
            XORInstruction(25, 12),
        ];
        const C: usize = 8;
        const M: usize = 1 << 8;
        test_helper::<Bls12_381, XORInstruction, C, M>(&ops);
    }

    #[test]
    fn e2e_non_pow_2() {
        let ops = vec![
            XORInstruction(0, 1),
            XORInstruction(101, 101),
            XORInstruction(202, 1),
            XORInstruction(220, 1),
            XORInstruction(220, 1),
        ];
        const C: usize = 2;
        const M: usize = 1 << 8;
        test_helper::<Bls12_381, XORInstruction, C, M>(&ops);
    }
}

use crate::{
    pcs::PolynomialCommitmentScheme,
    poly_iop::{
        errors::PolyIOPErrors,
        rational_sumcheck::layered_circuit::{
            BatchedDenseRationalSum, BatchedRationalSum, BatchedRationalSumProof,
            BatchedSparseRationalSum,
        },
        utils::drop_in_background_thread,
        PolyIOP,
    },
};
use arithmetic::{math::Math, Fraction, OptimizedMul};
use ark_ec::pairing::Pairing;
use ark_ff::{One, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{end_timer, start_timer};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use transcript::IOPTranscript;

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

use super::{
    instruction::JoltInstruction, util::SurgeCommons, SurgePolysPrimary, SurgePreprocessing,
};

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct LogupCheckingProof<F>
where
    F: PrimeField,
{
    pub f_final_claims: Vec<Fraction<F>>,
    pub f_proof: BatchedRationalSumProof<F>,
    pub g_final_claims: Vec<Fraction<F>>,
    pub g_proof: BatchedRationalSumProof<F>,
}

/// A permutation subclaim consists of
/// - the SubClaim from the ProductCheck
/// - Challenges beta and gamma
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LogupCheckingSubclaim<F>
where
    F: PrimeField,
{
    pub point_f: Vec<F>,
    pub expected_evaluations_f: Vec<Fraction<F>>,
    pub point_g: Vec<F>,
    pub expected_evaluations_g: Vec<Fraction<F>>,

    /// Challenges beta and gamma
    pub challenges: (F, F),
}

pub(super) trait LogupChecking<E, PCS, Instruction, const C: usize, const M: usize>:
    SurgeCommons<E::ScalarField, Instruction, C, M>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
    Instruction: JoltInstruction + Default,
{
    fn compute_f_leaves(
        preprocessing: &SurgePreprocessing<E::ScalarField>,
        polynomials: &SurgePolysPrimary<E>,
        beta: &E::ScalarField,
        gamma: &E::ScalarField,
        alpha: &E::ScalarField,
    ) -> (
        Vec<Vec<usize>>,
        Vec<Vec<E::ScalarField>>,
        Vec<Vec<E::ScalarField>>,
    ) {
        let mut f_leaves_q = preprocessing
            .materialized_subtables
            .par_iter()
            .map(|subtable| {
                subtable
                    .iter()
                    .enumerate()
                    .map(|(i, t_eval)| {
                        t_eval.mul_0_optimized(*gamma)
                            + E::ScalarField::from_u64(i as u64).unwrap()
                            + *beta
                    })
                    .collect()
            })
            .collect::<Vec<_>>();

        let bits_per_operand = (ark_std::log2(M) / 2) as usize;
        let sqrtM = bits_per_operand.pow2() as u64;
        let mut dechunk_evals = Vec::with_capacity(M);
        for x in 0..sqrtM {
            for y in 0..sqrtM {
                dechunk_evals.push(
                    (E::ScalarField::from_u64(x).unwrap()
                        + E::ScalarField::from_u64(y).unwrap() * *alpha)
                        * *gamma
                        + E::ScalarField::from_u64(((x << bits_per_operand) | y) as u64).unwrap()
                        + *beta,
                )
            }
        }
        f_leaves_q.push(dechunk_evals);

        (
            polynomials.m_indices.clone(),
            polynomials.m_values.clone(),
            f_leaves_q,
        )
    }

    fn d_compute_f_leaves(
        preprocessing: &SurgePreprocessing<E::ScalarField>,
        polynomials: &SurgePolysPrimary<E>,
        beta: &E::ScalarField,
        gamma: &E::ScalarField,
        alpha: &E::ScalarField,
    ) -> (
        Vec<Vec<usize>>,
        Vec<Vec<E::ScalarField>>,
        Vec<Vec<E::ScalarField>>,
    ) {
        let mut f_leaves_q = preprocessing
            .materialized_subtables
            .par_iter()
            .map(|subtable| {
                let len_per_party = subtable.len() / Net::n_parties();
                subtable[Net::party_id() * len_per_party..(Net::party_id() + 1) * len_per_party]
                    .iter()
                    .enumerate()
                    .map(|(i, t_eval)| {
                        t_eval.mul_0_optimized(*gamma)
                            + E::ScalarField::from_u64((i + Net::party_id() * len_per_party) as u64)
                                .unwrap()
                            + *beta
                    })
                    .collect()
            })
            .collect::<Vec<_>>();

        let bits_per_operand = (ark_std::log2(M) / 2) as usize;
        let sqrtM = bits_per_operand.pow2() as u64;
        let mut dechunk_evals = Vec::with_capacity(M);
        let len_per_party = sqrtM / (Net::n_parties() as u64);
        let party_id = Net::party_id() as u64;
        for x in party_id * len_per_party..(party_id + 1) * len_per_party {
            for y in 0..sqrtM {
                dechunk_evals.push(
                    (E::ScalarField::from_u64(x).unwrap()
                        + E::ScalarField::from_u64(y).unwrap() * *alpha)
                        * *gamma
                        + E::ScalarField::from_u64(((x << bits_per_operand) | y) as u64).unwrap()
                        + *beta,
                )
            }
        }
        f_leaves_q.push(dechunk_evals);

        (
            polynomials.m_indices.clone(),
            polynomials.m_values.clone(),
            f_leaves_q,
        )
    }

    // #[tracing::instrument(skip_all, name = "LogupCheckingProof::compute_leaves")]
    fn compute_g_leaves(
        polynomials: &SurgePolysPrimary<E>,
        beta: &E::ScalarField,
        gamma: &E::ScalarField,
    ) -> (Vec<Vec<E::ScalarField>>, Vec<Vec<E::ScalarField>>) {
        let num_lookups = polynomials.dim[0].evaluations.len();

        (0..Self::num_memories())
            .into_par_iter()
            .map(
                |memory_index| -> (Vec<E::ScalarField>, Vec<E::ScalarField>) {
                    let dim_index = Self::memory_to_dimension_index(memory_index);
                    (0..num_lookups)
                        .map(|i| {
                            (
                                E::ScalarField::one(),
                                polynomials.E_polys[memory_index][i].mul_0_optimized(*gamma)
                                    + polynomials.dim[dim_index][i]
                                    + *beta,
                            )
                        })
                        .unzip()
                },
            )
            .unzip()
    }

    // #[tracing::instrument(skip_all, name =
    // "LogupCheckingProof::prove_logup_checking")]
    fn prove_logup_checking(
        preprocessing: &SurgePreprocessing<E::ScalarField>,
        polynomials_primary: &SurgePolysPrimary<E>,
        alpha: &E::ScalarField,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            LogupCheckingProof<E::ScalarField>,
            Vec<E::ScalarField>,
            Vec<E::ScalarField>,
        ),
        PolyIOPErrors,
    > {
        // We assume that primary commitments are already appended to the transcript
        let beta = transcript.get_and_append_challenge(b"logup_beta")?;
        let gamma = transcript.get_and_append_challenge(b"logup_gamma")?;

        let ((f_claims, f_proof, r_f), (g_claims, g_proof, r_g)) = rayon::join(
            || {
                let mut transcript = transcript.clone();
                let f_leaves = Self::compute_f_leaves(
                    preprocessing,
                    &polynomials_primary,
                    &beta,
                    &gamma,
                    alpha,
                );
                let mut f_batched_circuit =
                    <BatchedSparseRationalSum<E::ScalarField, C> as BatchedRationalSum<
                        E::ScalarField,
                    >>::construct(f_leaves);
                let f_claims =
                    <BatchedSparseRationalSum<E::ScalarField, C> as BatchedRationalSum<
                        E::ScalarField,
                    >>::claims(&f_batched_circuit);

                let (f_proof, r_f) =
                    <BatchedSparseRationalSum<E::ScalarField, C> as BatchedRationalSum<
                        E::ScalarField,
                    >>::prove_rational_sum(
                        &mut f_batched_circuit, &mut transcript
                    );

                drop_in_background_thread(f_batched_circuit);
                (f_claims, f_proof, r_f)
            },
            || {
                let mut transcript = transcript.clone();
                let g_leaves = Self::compute_g_leaves(&polynomials_primary, &beta, &gamma);
                let mut g_batched_circuit =
                    <BatchedDenseRationalSum<E::ScalarField, 1> as BatchedRationalSum<
                        E::ScalarField,
                    >>::construct(g_leaves);
                let g_claims = <BatchedDenseRationalSum<E::ScalarField, 1> as BatchedRationalSum<
                    E::ScalarField,
                >>::claims(&g_batched_circuit);

                let (g_proof, r_g) =
                    <BatchedDenseRationalSum<E::ScalarField, 1> as BatchedRationalSum<
                        E::ScalarField,
                    >>::prove_rational_sum(
                        &mut g_batched_circuit, &mut transcript
                    );

                drop_in_background_thread(g_batched_circuit);

                (g_claims, g_proof, r_g)
            },
        );

        Ok((
            LogupCheckingProof {
                f_final_claims: f_claims,
                f_proof,
                g_final_claims: g_claims,
                g_proof,
            },
            r_f,
            r_g,
        ))
    }

    fn d_prove_logup_checking(
        preprocessing: &SurgePreprocessing<E::ScalarField>,
        polynomials_primary: &mut SurgePolysPrimary<E>,
        alpha: &E::ScalarField,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        Option<(
            LogupCheckingProof<E::ScalarField>,
            Vec<E::ScalarField>,
            Vec<E::ScalarField>,
        )>,
        PolyIOPErrors,
    > {
        let (beta, gamma) = if Net::am_master() {
            // We assume that primary commitments are already appended to the transcript
            let beta = transcript.get_and_append_challenge(b"logup_beta")?;
            let gamma = transcript.get_and_append_challenge(b"logup_gamma")?;
            Net::recv_from_master_uniform(Some((beta, gamma)))
        } else {
            Net::recv_from_master_uniform(None)
        };

        let (f_leaves, g_leaves) = rayon::join(
            || Self::d_compute_f_leaves(preprocessing, &polynomials_primary, &beta, &gamma, alpha),
            || Self::compute_g_leaves(&polynomials_primary, &beta, &gamma),
        );

        let (mut f_batched_circuit, mut f_companion_circuit) = <BatchedSparseRationalSum<
            E::ScalarField,
            C,
        > as BatchedRationalSum<
            E::ScalarField,
        >>::d_construct(f_leaves);

        let (mut g_batched_circuit, mut g_companion_circuit) = <BatchedDenseRationalSum<
            E::ScalarField,
            1,
        > as BatchedRationalSum<
            E::ScalarField,
        >>::d_construct(g_leaves);

        let mut transcript_f = transcript.clone();
        let f_proof_ret = <BatchedSparseRationalSum<E::ScalarField, C> as BatchedRationalSum<
            E::ScalarField,
        >>::d_prove_rational_sum(
            &mut f_batched_circuit,
            f_companion_circuit.as_mut(),
            &mut transcript_f,
        );

        let mut transcript_g = transcript.clone();
        let g_proof_ret = <BatchedDenseRationalSum<E::ScalarField, 1> as BatchedRationalSum<
            E::ScalarField,
        >>::d_prove_rational_sum(
            &mut g_batched_circuit,
            g_companion_circuit.as_mut(),
            &mut transcript_g,
        );

        drop_in_background_thread(f_batched_circuit);
        drop_in_background_thread(g_batched_circuit);

        if Net::am_master() {
            let f_claims = f_companion_circuit.as_ref().unwrap().claims();
            let g_claims = g_companion_circuit.as_ref().unwrap().claims();

            drop_in_background_thread(f_companion_circuit);
            drop_in_background_thread(g_companion_circuit);

            let (f_proof, r_f) = f_proof_ret.unwrap();
            let (g_proof, r_g) = g_proof_ret.unwrap();

            Ok(Some((
                LogupCheckingProof {
                    f_final_claims: f_claims,
                    f_proof,
                    g_final_claims: g_claims,
                    g_proof,
                },
                r_f,
                r_g,
            )))
        } else {
            drop_in_background_thread(f_companion_circuit);
            drop_in_background_thread(g_companion_circuit);

            Ok(None)
        }
    }

    // #[tracing::instrument(skip_all, name =
    // "LogupCheckingProof::verify_logup_checking")]
    fn verify_logup_checking(
        proof: &LogupCheckingProof<E::ScalarField>,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<LogupCheckingSubclaim<E::ScalarField>, PolyIOPErrors> {
        let num_memories = Self::num_memories();

        // Check that the final claims are equal
        rayon::join(
            || {
                (0..num_memories).into_par_iter().for_each(|i| {
                    assert_eq!(
                        proof.f_final_claims[i].p * proof.g_final_claims[i].q,
                        proof.f_final_claims[i].q * proof.g_final_claims[i].p,
                        "Final claims are inconsistent"
                    );
                });
            },
            || {
                // Assumes that primary commitments have been added to transcript
                let beta = transcript.get_and_append_challenge(b"logup_beta")?;
                let gamma = transcript.get_and_append_challenge(b"logup_gamma")?;

                let ((f_claims, r_f), (g_claims, r_g)) =
                    rayon::join(
                        || {
                            let mut transcript = transcript.clone();

                            let step = start_timer!(|| "verify rational sum f");
                            let (f_claims, r_f) =
                    <BatchedSparseRationalSum<E::ScalarField, C> as BatchedRationalSum<
                        E::ScalarField,
                    >>::verify_rational_sum(
                        &proof.f_proof, &proof.f_final_claims, &mut transcript
                    );
                            end_timer!(step);

                            (f_claims, r_f)
                        },
                        || {
                            let mut transcript = transcript.clone();

                            let step = start_timer!(|| "verify rational sum g");
                            let (g_claims, r_g) =
                    <BatchedDenseRationalSum<E::ScalarField, 1> as BatchedRationalSum<
                        E::ScalarField,
                    >>::verify_rational_sum(
                        &proof.g_proof, &proof.g_final_claims, &mut transcript
                    );
                            end_timer!(step);

                            (g_claims, r_g)
                        },
                    );

                Ok(LogupCheckingSubclaim {
                    point_f: r_f,
                    expected_evaluations_f: f_claims,
                    point_g: r_g,
                    expected_evaluations_g: g_claims,
                    challenges: (beta, gamma),
                })
            },
        )
        .1
    }
}

impl<E, PCS, Instruction, const C: usize, const M: usize> LogupChecking<E, PCS, Instruction, C, M>
    for PolyIOP<E::ScalarField>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
    Instruction: JoltInstruction + Default,
{
}

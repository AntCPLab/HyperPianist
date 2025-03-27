use crate::{
    pcs::PolynomialCommitmentScheme,
    poly_iop::{
        errors::PolyIOPErrors,
        rational_sumcheck::{
            RationalSumcheckProof, RationalSumcheckSlow, RationalSumcheckSubClaim,
        },
        PolyIOP,
    },
};
use arithmetic::{math::Math, OptimizedMul, VirtualPolynomial};
use ark_ec::pairing::Pairing;
use ark_ff::{batch_inversion, One, PrimeField};
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::{mem::take, sync::Arc};
use transcript::IOPTranscript;

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

use super::{
    instruction::JoltInstruction, util::SurgeCommons, SurgePolysPrimary, SurgePreprocessing,
};

#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct LogupCheckingProof<E: Pairing, PCS: PolynomialCommitmentScheme<E>> {
    pub f_proof: RationalSumcheckProof<E::ScalarField>,
    pub g_proof: RationalSumcheckProof<E::ScalarField>,
    pub f_inv_comm: Vec<PCS::Commitment>,
    pub g_inv_comm: Vec<PCS::Commitment>,
}

/// A permutation subclaim consists of
/// - the SubClaim from the ProductCheck
/// - Challenges beta and gamma
#[derive(Clone, Debug, PartialEq)]
pub struct LogupCheckingSubclaim<F: PrimeField>
where
    F: PrimeField,
{
    pub f_subclaims: RationalSumcheckSubClaim<F>,
    pub g_subclaims: RationalSumcheckSubClaim<F>,

    /// Challenges beta and gamma
    pub challenges: (F, F),
}

pub(super) trait LogupChecking<E, PCS, Instruction, const C: usize, const M: usize>:
    RationalSumcheckSlow<E::ScalarField>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
    Instruction: JoltInstruction + Default,
{
    type LogupCheckingProof: CanonicalSerialize + CanonicalDeserialize;
    type LogupCheckingSubclaim;
    type Preprocessing;
    type Polys;

    fn compute_f_leaves(
        preprocessing: &Self::Preprocessing,
        polynomials: &Self::Polys,
        beta: &E::ScalarField,
        gamma: &E::ScalarField,
        alpha: &E::ScalarField,
    ) -> (
        Vec<Self::VirtualPolynomial>,
        Vec<Self::MultilinearExtension>,
        Vec<Self::MultilinearExtension>,
    );

    fn d_compute_f_leaves(
        preprocessing: &Self::Preprocessing,
        polynomials: &mut Self::Polys,
        beta: &E::ScalarField,
        gamma: &E::ScalarField,
        alpha: &E::ScalarField,
    ) -> (
        Vec<Self::VirtualPolynomial>,
        Vec<Self::MultilinearExtension>,
        Vec<Self::MultilinearExtension>,
    );

    fn compute_g_leaves(
        polynomials: &Self::Polys,
        beta: &E::ScalarField,
        gamma: &E::ScalarField,
    ) -> (
        Vec<Self::VirtualPolynomial>,
        Vec<Self::MultilinearExtension>,
        Vec<Self::MultilinearExtension>,
    );

    fn prove_logup_checking(
        pcs_param: &PCS::ProverParam,
        preprocessing: &Self::Preprocessing,
        polynomials_primary: &Self::Polys,
        alpha: &E::ScalarField,
        transcript: &mut Self::Transcript,
    ) -> Result<
        (
            Self::LogupCheckingProof,
            Vec<PCS::ProverCommitmentAdvice>,
            Vec<Self::MultilinearExtension>,
            Vec<Self::MultilinearExtension>,
        ),
        PolyIOPErrors,
    >;

    fn d_prove_logup_checking(
        pcs_param: &PCS::ProverParam,
        preprocessing: &Self::Preprocessing,
        polynomials_primary: &mut Self::Polys,
        alpha: &E::ScalarField,
        transcript: &mut Self::Transcript,
    ) -> Result<
        (
            Option<Self::LogupCheckingProof>,
            Vec<PCS::ProverCommitmentAdvice>,
            Vec<Self::MultilinearExtension>,
            Vec<Self::MultilinearExtension>,
        ),
        PolyIOPErrors,
    >;

    fn verify_logup_checking(
        proof: &Self::LogupCheckingProof,
        aux_info_f: &Self::VPAuxInfo,
        aux_info_g: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::LogupCheckingSubclaim, PolyIOPErrors>;
}

impl<E, PCS, Instruction, const C: usize, const M: usize> LogupChecking<E, PCS, Instruction, C, M>
    for PolyIOP<E::ScalarField>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E, Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>>,
    Instruction: JoltInstruction + Default,
{
    type LogupCheckingProof = LogupCheckingProof<E, PCS>;
    type LogupCheckingSubclaim = LogupCheckingSubclaim<E::ScalarField>;
    type Preprocessing = SurgePreprocessing<E::ScalarField>;
    type Polys = SurgePolysPrimary<E>;

    fn compute_f_leaves(
        preprocessing: &SurgePreprocessing<E::ScalarField>,
        polynomials: &SurgePolysPrimary<E>,
        beta: &E::ScalarField,
        gamma: &E::ScalarField,
        alpha: &E::ScalarField,
    ) -> (
        Vec<Self::VirtualPolynomial>,
        Vec<Self::MultilinearExtension>,
        Vec<Self::MultilinearExtension>,
    ) {
        let mut f_leaves_q = preprocessing
            .materialized_subtables
            .par_iter()
            .map(|subtable| {
                let q = DenseMultilinearExtension::from_evaluations_vec(
                    subtable.len().log_2(),
                    subtable
                        .iter()
                        .enumerate()
                        .map(|(i, t_eval)| {
                            t_eval.mul_0_optimized(*gamma)
                                + E::ScalarField::from_u64(i as u64).unwrap()
                                + *beta
                        })
                        .collect(),
                );

                let mut q_inv = DenseMultilinearExtension::clone(&q);
                batch_inversion(&mut q_inv.evaluations);

                (Arc::new(q), Arc::new(q_inv))
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();

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
        let mut dechunk_evals_inv = dechunk_evals.clone();
        batch_inversion(&mut dechunk_evals_inv);

        f_leaves_q
            .0
            .push(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                ark_std::log2(M) as usize,
                dechunk_evals,
            )));
        f_leaves_q
            .1
            .push(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                ark_std::log2(M) as usize,
                dechunk_evals_inv,
            )));

        let f_leaves_p = polynomials
            .m
            .iter()
            .map(|m_poly| {
                VirtualPolynomial::new_from_mle(
                    &Arc::new(DenseMultilinearExtension::clone(m_poly)),
                    E::ScalarField::one(),
                )
            })
            .collect();
        (f_leaves_p, f_leaves_q.0, f_leaves_q.1)
    }

    fn d_compute_f_leaves(
        preprocessing: &SurgePreprocessing<E::ScalarField>,
        polynomials: &mut SurgePolysPrimary<E>,
        beta: &E::ScalarField,
        gamma: &E::ScalarField,
        alpha: &E::ScalarField,
    ) -> (
        Vec<Self::VirtualPolynomial>,
        Vec<Self::MultilinearExtension>,
        Vec<Self::MultilinearExtension>,
    ) {
        let num_party_vars = Net::n_parties().log_2();
        let mut f_leaves_q = preprocessing
            .materialized_subtables
            .par_iter()
            .map(|subtable| {
                let len_per_party = subtable.len() / Net::n_parties();
                let q = DenseMultilinearExtension::from_evaluations_vec(
                    subtable.len().log_2() - num_party_vars,
                    subtable
                        [Net::party_id() * len_per_party..(Net::party_id() + 1) * len_per_party]
                        .iter()
                        .enumerate()
                        .map(|(i, t_eval)| {
                            t_eval.mul_0_optimized(*gamma)
                                + E::ScalarField::from_u64(
                                    (i + Net::party_id() * len_per_party) as u64,
                                )
                                .unwrap()
                                + *beta
                        })
                        .collect(),
                );

                let mut q_inv = DenseMultilinearExtension::clone(&q);
                batch_inversion(&mut q_inv.evaluations);

                (Arc::new(q), Arc::new(q_inv))
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();

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

        let mut dechunk_evals_inv = dechunk_evals.clone();
        batch_inversion(&mut dechunk_evals_inv);

        f_leaves_q
            .0
            .push(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                ark_std::log2(M) as usize - num_party_vars,
                dechunk_evals,
            )));
        f_leaves_q
            .1
            .push(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                ark_std::log2(M) as usize - num_party_vars,
                dechunk_evals_inv,
            )));

        let f_leaves_p = polynomials
            .m
            .iter()
            .map(|m_poly| {
                VirtualPolynomial::new_from_mle(
                    &Arc::new(DenseMultilinearExtension::clone(m_poly)),
                    E::ScalarField::one(),
                )
            })
            .collect();
        (f_leaves_p, f_leaves_q.0, f_leaves_q.1)
    }

    fn compute_g_leaves(
        polynomials: &SurgePolysPrimary<E>,
        beta: &E::ScalarField,
        gamma: &E::ScalarField,
    ) -> (
        Vec<Self::VirtualPolynomial>,
        Vec<Self::MultilinearExtension>,
        Vec<Self::MultilinearExtension>,
    ) {
        let num_vars_g = polynomials.dim[0].num_vars;
        let num_lookups = polynomials.dim[0].evaluations.len();
        let g_leaves_q = (0
            ..<Self as SurgeCommons<E::ScalarField, Instruction, C, M>>::num_memories())
            .into_par_iter()
            .map(|memory_index| {
                let dim_index = <Self as SurgeCommons<E::ScalarField,
    Instruction, C, M>>::memory_to_dimension_index(memory_index);

                let q = DenseMultilinearExtension::from_evaluations_vec(
                    num_vars_g,
                    (0..num_lookups)
                        .map(|i| {
                            polynomials.E_polys[memory_index][i].mul_0_optimized(*gamma)
                                + polynomials.dim[dim_index][i]
                                + *beta
                        })
                        .collect(),
                );

                let mut q_inv = DenseMultilinearExtension::clone(&q);
                batch_inversion(&mut q_inv.evaluations);

                (Arc::new(q), Arc::new(q_inv))
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();
        let mut g_leaves_p = VirtualPolynomial::new(num_vars_g);
        g_leaves_p.add_mle_list([], E::ScalarField::one()).unwrap();

        (
            vec![g_leaves_p.clone(); g_leaves_q.0.len()],
            g_leaves_q.0,
            g_leaves_q.1,
        )
    }

    // #[tracing::instrument(skip_all, name =
    // "LogupCheckingProof::prove_logup_checking")]
    fn prove_logup_checking(
        pcs_param: &PCS::ProverParam,
        preprocessing: &SurgePreprocessing<E::ScalarField>,
        polynomials_primary: &SurgePolysPrimary<E>,
        alpha: &E::ScalarField,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Self::LogupCheckingProof,
            Vec<PCS::ProverCommitmentAdvice>,
            Vec<Self::MultilinearExtension>,
            Vec<Self::MultilinearExtension>,
        ),
        PolyIOPErrors,
    > {
        // We assume that primary commitments are already appended to the transcript
        let beta = transcript.get_and_append_challenge(b"logup_beta")?;
        let gamma = transcript.get_and_append_challenge(b"logup_gamma")?;

        let (f_leaves, g_leaves) = (
            <Self as LogupChecking<E, PCS, Instruction, C, M>>::compute_f_leaves(
                preprocessing,
                polynomials_primary,
                &beta,
                &gamma,
                alpha,
            ),
            <Self as LogupChecking<E, PCS, Instruction, C, M>>::compute_g_leaves(
                polynomials_primary,
                &beta,
                &gamma,
            ),
        );

        let ((f_inv_comm, mut f_advice), (g_inv_comm, mut g_advice)): (
            (Vec<_>, Vec<_>),
            (Vec<_>, Vec<_>),
        ) = rayon::join(
            || {
                f_leaves
                    .2
                    .iter()
                    .map(|inv| PCS::commit(pcs_param, inv).unwrap())
                    .unzip()
            },
            || {
                g_leaves
                    .2
                    .iter()
                    .map(|inv| PCS::commit(pcs_param, inv).unwrap())
                    .unzip()
            },
        );
        f_advice.append(&mut g_advice);

        let claimed_sums = (0
            ..<Self as SurgeCommons<E::ScalarField, Instruction, C, M>>::num_memories())
            .map(|memory_index| g_leaves.2[memory_index].evaluations.iter().sum())
            .collect::<Vec<_>>();

        transcript.append_serializable_element(b"f_inv_comm", &f_inv_comm)?;
        transcript.append_serializable_element(b"g_inv_comm", &g_inv_comm)?;

        let f_inv_copy = f_leaves
            .2
            .iter()
            .map(|x| Arc::new(DenseMultilinearExtension::clone(x)))
            .collect::<Vec<_>>();
        let f_p = (0..<Self as SurgeCommons<E::ScalarField, Instruction, C, M>>::num_memories())
            .map(|memory_index| {
                let dim_index = <Self as SurgeCommons<E::ScalarField, Instruction, C, M>>::memory_to_dimension_index(memory_index);
                f_leaves.0[dim_index].clone()
            }).collect::<Vec<_>>();
        let (f_q, f_q_inv) = (0
            ..<Self as SurgeCommons<E::ScalarField, Instruction, C, M>>::num_memories())
            .map(|memory_index| {
                let subtable_index = <Self as SurgeCommons<
                        E::ScalarField,
                        Instruction,
                        C,
                        M,
                    >>::memory_to_subtable_index(
                        memory_index
                    );
                (
                    f_leaves.1[subtable_index].clone(),
                    f_leaves.2[subtable_index].clone(),
                )
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();
        drop(f_leaves);

        let f_proof = <Self as RationalSumcheckSlow<E::ScalarField>>::prove(
            f_p,
            f_q,
            f_q_inv,
            claimed_sums.clone(),
            transcript,
        )?;

        let (fx, gx, g_inv) = g_leaves;
        let g_proof = <Self as RationalSumcheckSlow<E::ScalarField>>::prove(
            fx,
            gx,
            g_inv
                .iter()
                .map(|x| Arc::new(DenseMultilinearExtension::clone(x)))
                .collect(),
            claimed_sums,
            transcript,
        )?;

        Ok((
            LogupCheckingProof {
                f_proof,
                f_inv_comm,
                g_proof,
                g_inv_comm,
            },
            f_advice,
            f_inv_copy,
            g_inv,
        ))
    }

    fn d_prove_logup_checking(
        pcs_param: &PCS::ProverParam,
        preprocessing: &SurgePreprocessing<E::ScalarField>,
        polynomials_primary: &mut SurgePolysPrimary<E>,
        alpha: &E::ScalarField,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Option<Self::LogupCheckingProof>,
            Vec<PCS::ProverCommitmentAdvice>,
            Vec<Self::MultilinearExtension>,
            Vec<Self::MultilinearExtension>,
        ),
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

        let f_leaves = <Self as LogupChecking<E, PCS, Instruction, C, M>>::d_compute_f_leaves(
            preprocessing,
            polynomials_primary,
            &beta,
            &gamma,
            alpha,
        );

        let (mut f_inv_comm, mut f_advice): (Vec<_>, Vec<_>) = f_leaves
            .2
            .iter()
            .map(|inv| PCS::d_commit(pcs_param, inv).unwrap())
            .unzip();

        let f_inv_copy = f_leaves
            .2
            .iter()
            .map(|x| Arc::new(DenseMultilinearExtension::clone(x)))
            .collect::<Vec<_>>();
        let f_p = (0..<Self as SurgeCommons<E::ScalarField, Instruction, C, M>>::num_memories())
                    .map(|memory_index| {
                        let dim_index = <Self as SurgeCommons<E::ScalarField, Instruction, C, M>>::memory_to_dimension_index(memory_index);
                        f_leaves.0[dim_index].clone()
                    }).collect::<Vec<_>>();
        let (f_q, f_q_inv) = (0
            ..<Self as SurgeCommons<E::ScalarField, Instruction, C, M>>::num_memories())
            .map(|memory_index| {
                let subtable_index = <Self as SurgeCommons<
                                E::ScalarField,
                                Instruction,
                                C,
                                M,
                            >>::memory_to_subtable_index(
                                memory_index
                            );
                (
                    f_leaves.1[subtable_index].clone(),
                    f_leaves.2[subtable_index].clone(),
                )
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();
        drop(f_leaves);

        let g_leaves = <Self as LogupChecking<E, PCS, Instruction, C, M>>::compute_g_leaves(
            polynomials_primary,
            &beta,
            &gamma,
        );

        let (mut g_inv_comm, mut g_advice): (Vec<_>, Vec<_>) = g_leaves
            .2
            .iter()
            .map(|inv| PCS::d_commit(pcs_param, inv).unwrap())
            .unzip();

        f_advice.append(&mut g_advice);

        let claimed_sums = (0
            ..<Self as SurgeCommons<E::ScalarField, Instruction, C, M>>::num_memories())
            .map(|memory_index| g_leaves.2[memory_index].evaluations.iter().sum())
            .collect::<Vec<_>>();
        let all_claimed_sums = Net::send_to_master(&claimed_sums);

        if Net::am_master() {
            let all_claimed_sums = all_claimed_sums.unwrap();
            let claimed_sums = (0..all_claimed_sums[0].len())
                .map(|memory_index| {
                    all_claimed_sums
                        .iter()
                        .map(|claimed_sums| claimed_sums[memory_index])
                        .sum::<E::ScalarField>()
                })
                .collect::<Vec<_>>();

            let f_inv_comm = f_inv_comm
                .iter_mut()
                .map(|comm| take(comm).unwrap())
                .collect::<Vec<_>>();
            let g_inv_comm = g_inv_comm
                .iter_mut()
                .map(|comm| take(comm).unwrap())
                .collect::<Vec<_>>();

            transcript.append_serializable_element(b"f_inv_comm", &f_inv_comm)?;
            transcript.append_serializable_element(b"g_inv_comm", &g_inv_comm)?;

            let f_proof_ret = <Self as RationalSumcheckSlow<E::ScalarField>>::d_prove(
                f_p,
                f_q,
                f_q_inv,
                claimed_sums.clone(),
                transcript,
            )?;
            let f_proof = f_proof_ret.unwrap();

            let (fx, gx, g_inv) = g_leaves;
            let g_proof_ret = <Self as RationalSumcheckSlow<E::ScalarField>>::d_prove(
                fx,
                gx,
                g_inv
                    .iter()
                    .map(|x| Arc::new(DenseMultilinearExtension::clone(x)))
                    .collect(),
                claimed_sums,
                transcript,
            )?;
            let g_proof = g_proof_ret.unwrap();

            Ok((
                Some(LogupCheckingProof {
                    f_proof,
                    f_inv_comm,
                    g_proof,
                    g_inv_comm,
                }),
                f_advice,
                f_inv_copy,
                g_inv,
            ))
        } else {
            // Claimed sums are not correct here but it doesn't matter for non-master
            <Self as RationalSumcheckSlow<E::ScalarField>>::d_prove(
                f_p,
                f_q,
                f_q_inv,
                claimed_sums.clone(),
                transcript,
            )?;
            let (fx, gx, g_inv) = g_leaves;
            <Self as RationalSumcheckSlow<E::ScalarField>>::d_prove(
                fx,
                gx,
                g_inv
                    .iter()
                    .map(|x| Arc::new(DenseMultilinearExtension::clone(x)))
                    .collect(),
                claimed_sums,
                transcript,
            )?;
            Ok((None, f_advice, f_inv_copy, g_inv))
        }
    }

    // #[tracing::instrument(skip_all, name =
    // "LogupCheckingProof::verify_logup_checking")]
    fn verify_logup_checking(
        proof: &LogupCheckingProof<E, PCS>,
        aux_info_f: &Self::VPAuxInfo,
        aux_info_g: &Self::VPAuxInfo,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<LogupCheckingSubclaim<E::ScalarField>, PolyIOPErrors> {
        // Check that the final claims are equal
        rayon::join(
            || {
                (0..<Self as SurgeCommons<E::ScalarField, Instruction, C, M>>::num_memories())
                    .into_par_iter()
                    .for_each(|i| {
                        assert_eq!(
                            proof.f_proof.claimed_sums[i], proof.g_proof.claimed_sums[i],
                            "Final claims are inconsistent"
                        );
                    });
            },
            || {
                // Assumes that primary commitments have been added to transcript
                let beta = transcript.get_and_append_challenge(b"logup_beta")?;
                let gamma = transcript.get_and_append_challenge(b"logup_gamma")?;

                transcript.append_serializable_element(b"f_inv_comm", &proof.f_inv_comm)?;
                transcript.append_serializable_element(b"g_inv_comm", &proof.g_inv_comm)?;

                let f_subclaims = <Self as RationalSumcheckSlow<E::ScalarField>>::verify(
                    &proof.f_proof,
                    aux_info_f,
                    transcript,
                )?;

                let g_subclaims = <Self as RationalSumcheckSlow<E::ScalarField>>::verify(
                    &proof.g_proof,
                    aux_info_g,
                    transcript,
                )?;

                Ok(LogupCheckingSubclaim {
                    f_subclaims,
                    g_subclaims,
                    challenges: (beta, gamma),
                })
            },
        )
        .1
    }
}

// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Main module for the Permutation Check protocol

use crate::{
    poly_iop::{errors::PolyIOPErrors, PolyIOP},
    MultiRationalSumcheck, MultiRationalSumcheckProof, PolynomialCommitmentScheme,
};
use arithmetic::{eq_eval, math::Math, products_except_self};
use ark_ec::pairing::Pairing;
use ark_ff::{batch_inversion, PrimeField};
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{end_timer, start_timer, One, Zero};
use itertools::{izip, Itertools};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::Arc;
use transcript::IOPTranscript;
use util::compute_leaves;

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

use super::multi_rational_sumcheck::MultiRationalSumcheckSubClaim;

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct PermutationCheckProof<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub proofs: Vec<MultiRationalSumcheckProof<E::ScalarField>>,
    pub h_comms: Vec<PCS::Commitment>,
}

/// A permutation subclaim consists of
/// - the SubClaim from the ProductCheck
/// - Challenges beta and gamma
#[derive(Clone, Debug, Default, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct PermutationCheckSubClaim<F>
where
    F: PrimeField,
{
    pub subclaims: Vec<(MultiRationalSumcheckSubClaim<F>, usize)>,
    /// Challenges beta and gamma
    pub challenges: (F, F),
}

pub mod util;

/// A PermutationCheck w.r.t. `(fs, gs, perms)`
/// proves that (g1, ..., gk) is a permutation of (f1, ..., fk) under
/// permutation `(p1, ..., pk)`
/// It is derived from ProductCheck.
///
/// A Permutation Check IOP takes the following steps:
///
/// Inputs:
/// - fs = (f1, ..., fk)
/// - gs = (g1, ..., gk)
/// - permutation oracles = (p1, ..., pk)
pub trait PermutationCheck<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    type PermutationCheckSubClaim;
    type PermutationProof: CanonicalSerialize + CanonicalDeserialize;

    type MultilinearExtension;
    type Transcript;

    /// Initialize the system with a transcript
    ///
    /// This function is optional -- in the case where a PermutationCheck is
    /// an building block for a more complex protocol, the transcript
    /// may be initialized by this complex protocol, and passed to the
    /// PermutationCheck prover/verifier.
    fn init_transcript() -> Self::Transcript;

    /// Inputs:
    /// - fs = (f1, ..., fk)
    /// - gs = (g1, ..., gk)
    /// - permutation oracles = (p1, ..., pk)
    /// Outputs:
    /// - a permutation check proof proving that gs is a permutation of fs under
    ///   permutation
    ///
    /// Cost: O(N)
    #[allow(clippy::type_complexity)]
    fn prove(
        prover_param: &PCS::ProverParam,
        fxs: &[Self::MultilinearExtension],
        gxs: &[Self::MultilinearExtension],
        perms: &[Self::MultilinearExtension],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Self::PermutationProof,
            Vec<PCS::ProverCommitmentAdvice>,
            Vec<Vec<E::ScalarField>>,
            Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>,
        ),
        PolyIOPErrors,
    >;

    fn d_prove_prepare(
        prover_param: &PCS::ProverParam,
        fxs: &[Self::MultilinearExtension],
        gxs: &[Self::MultilinearExtension],
        perms: &[Self::MultilinearExtension],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Vec<(
                Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>,
                Arc<DenseMultilinearExtension<E::ScalarField>>,
                E::ScalarField,
                Option<PCS::Commitment>,
                PCS::ProverCommitmentAdvice,
            )>,
            Vec<E::ScalarField>,
        ),
        PolyIOPErrors,
    >;

    fn d_prove(
        prover_param: &PCS::ProverParam,
        to_prove: Vec<(
            Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>,
            Arc<DenseMultilinearExtension<E::ScalarField>>,
            E::ScalarField,
            Option<PCS::Commitment>,
            PCS::ProverCommitmentAdvice,
        )>,
        claims: Vec<E::ScalarField>,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Option<(Self::PermutationProof, Vec<Vec<E::ScalarField>>)>,
            Vec<PCS::ProverCommitmentAdvice>,
            Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>,
        ),
        PolyIOPErrors,
    >;

    /// Verify that (g1, ..., gk) is a permutation of
    /// (f1, ..., fk) over the permutation oracles (perm1, ..., permk)
    fn verify(
        proof: &Self::PermutationProof,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::PermutationCheckSubClaim, PolyIOPErrors>;

    fn check_openings(
        subclaim: &Self::PermutationCheckSubClaim,
        f_openings: &[E::ScalarField],
        g_openings: &[E::ScalarField],
        h_openings: &[E::ScalarField],
        perm_openings: &[E::ScalarField],
    ) -> Result<(), PolyIOPErrors>;
}

impl<E, PCS> PermutationCheck<E, PCS> for PolyIOP<E::ScalarField>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E, Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>>,
{
    type PermutationCheckSubClaim = PermutationCheckSubClaim<E::ScalarField>;
    type PermutationProof = PermutationCheckProof<E, PCS>;
    type MultilinearExtension = Arc<DenseMultilinearExtension<E::ScalarField>>;
    type Transcript = IOPTranscript<E::ScalarField>;

    fn init_transcript() -> Self::Transcript {
        IOPTranscript::<E::ScalarField>::new(b"Initializing PermutationCheck transcript")
    }

    // Strictly speaking the list of points is redundant as it is present in the
    // proofs, but we try to keep the interface uniform
    fn prove(
        prover_param: &PCS::ProverParam,
        fxs: &[Self::MultilinearExtension],
        gxs: &[Self::MultilinearExtension],
        perms: &[Self::MultilinearExtension],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Self::PermutationProof,
            Vec<PCS::ProverCommitmentAdvice>,
            Vec<Vec<E::ScalarField>>,
            Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>,
        ),
        PolyIOPErrors,
    > {
        let start = start_timer!(|| "Permutation check prove");
        if fxs.is_empty() {
            return Err(PolyIOPErrors::InvalidParameters("fxs is empty".to_string()));
        }
        if (fxs.len() != gxs.len()) || (fxs.len() != perms.len()) {
            return Err(PolyIOPErrors::InvalidProof(format!(
                "fxs.len() = {}, gxs.len() = {}, perms.len() = {}",
                fxs.len(),
                gxs.len(),
                perms.len(),
            )));
        }

        // generate challenge `beta` and `gamma` from current transcript
        let beta = transcript.get_and_append_challenge(b"beta")?;
        let gamma = transcript.get_and_append_challenge(b"gamma")?;
        let leaves = compute_leaves::<E::ScalarField, false>(&beta, &gamma, fxs, gxs, perms)?;

        let leaves_len = leaves.len();

        let to_prove = leaves
            .into_par_iter()
            .map(|leave| {
                let half_len = leave.len() / 2;
                let nv = leave[0].len().log_2();
                let (g_polys, inv_evals): (Vec<_>, Vec<_>) = leave
                    .into_par_iter()
                    .map(|evals| {
                        let mut inv_evals = evals.clone();
                        batch_inversion(&mut inv_evals);

                        (
                            Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, evals)),
                            inv_evals,
                        )
                    })
                    .unzip();
                let h_evals = (0..inv_evals[0].len())
                    .into_par_iter()
                    .map(|i| {
                        inv_evals[..half_len]
                            .iter()
                            .map(|eval| eval[i])
                            .sum::<E::ScalarField>()
                            - inv_evals[half_len..]
                                .iter()
                                .map(|eval| eval[i])
                                .sum::<E::ScalarField>()
                    })
                    .collect::<Vec<_>>();
                let claim = if leaves_len == 1 {
                    E::ScalarField::zero()
                } else {
                    h_evals.iter().sum::<E::ScalarField>()
                };

                let h_poly = Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, h_evals));
                let (h_comm, h_advice) = PCS::commit(prover_param, &h_poly).unwrap();

                (g_polys, h_poly, claim, h_comm, h_advice)
            })
            .collect::<Vec<_>>();

        let (proofs, points, comms, advices, polys): (Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>) =
            to_prove
                .into_iter()
                .map(|(g_polys, h_poly, claim, h_comm, h_advice)| {
                    let mut f_values = vec![E::ScalarField::one(); g_polys.len()];
                    f_values[g_polys.len() / 2..].fill(-E::ScalarField::one());
                    let (proof, point) = <Self as MultiRationalSumcheck<E::ScalarField>>::prove(
                        &f_values,
                        g_polys,
                        Arc::new(DenseMultilinearExtension::clone(&h_poly)),
                        claim,
                        transcript,
                    )
                    .unwrap();
                    (proof, point, h_comm, h_advice, h_poly)
                })
                .multiunzip();

        end_timer!(start);

        Ok((
            Self::PermutationProof {
                proofs,
                h_comms: comms,
            },
            advices,
            points,
            polys,
        ))
    }

    fn d_prove_prepare(
        prover_param: &PCS::ProverParam,
        fxs: &[Self::MultilinearExtension],
        gxs: &[Self::MultilinearExtension],
        perms: &[Self::MultilinearExtension],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Vec<(
                Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>,
                Arc<DenseMultilinearExtension<E::ScalarField>>,
                E::ScalarField,
                Option<PCS::Commitment>,
                PCS::ProverCommitmentAdvice,
            )>,
            Vec<E::ScalarField>,
        ),
        PolyIOPErrors,
    > {
        let start = start_timer!(|| "Permutation check prove");
        if fxs.is_empty() {
            return Err(PolyIOPErrors::InvalidParameters("fxs is empty".to_string()));
        }
        if (fxs.len() != gxs.len()) || (fxs.len() != perms.len()) {
            return Err(PolyIOPErrors::InvalidProof(format!(
                "fxs.len() = {}, gxs.len() = {}, perms.len() = {}",
                fxs.len(),
                gxs.len(),
                perms.len(),
            )));
        }

        let (beta, gamma) = if Net::am_master() {
            let beta = transcript.get_and_append_challenge(b"beta")?;
            let gamma = transcript.get_and_append_challenge(b"gamma")?;
            Net::recv_from_master_uniform(Some((beta, gamma)))
        } else {
            Net::recv_from_master_uniform(None)
        };

        let leaves = compute_leaves::<E::ScalarField, true>(&beta, &gamma, fxs, gxs, perms)?;

        let leaves_len = leaves.len();
        let to_prove = leaves
            .into_iter()
            .map(|leave| {
                let half_len = leave.len() / 2;
                let nv = leave[0].len().log_2();
                let (g_polys, inv_evals): (Vec<_>, Vec<_>) = leave
                    .into_par_iter()
                    .map(|evals| {
                        let mut inv_evals = evals.clone();
                        batch_inversion(&mut inv_evals);

                        (
                            Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, evals)),
                            inv_evals,
                        )
                    })
                    .unzip();
                let h_evals = (0..inv_evals[0].len())
                    .into_par_iter()
                    .map(|i| {
                        inv_evals[..half_len]
                            .iter()
                            .map(|eval| eval[i])
                            .sum::<E::ScalarField>()
                            - inv_evals[half_len..]
                                .iter()
                                .map(|eval| eval[i])
                                .sum::<E::ScalarField>()
                    })
                    .collect::<Vec<_>>();
                let claim = if leaves_len == 1 {
                    E::ScalarField::zero()
                } else {
                    h_evals.iter().sum::<E::ScalarField>()
                };
                let h_poly = Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, h_evals));
                let (h_comm, h_advice) = PCS::d_commit(prover_param, &h_poly).unwrap();

                (g_polys, h_poly, claim, h_comm, h_advice)
            })
            .collect::<Vec<_>>();

        let mut claims = to_prove
            .iter()
            .map(|(_, _, claim, ..)| *claim)
            .collect::<Vec<_>>();
        let all_claims = Net::send_to_master(&claims);
        if Net::am_master() {
            let all_claims = all_claims.unwrap();
            claims = (0..all_claims[0].len())
                .map(|i| {
                    all_claims
                        .iter()
                        .map(|claims| claims[i])
                        .sum::<E::ScalarField>()
                })
                .collect::<Vec<_>>();
        }

        end_timer!(start);

        Ok((to_prove, claims))
    }

    fn d_prove(
        _prover_param: &PCS::ProverParam,
        to_prove: Vec<(
            Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>,
            Arc<DenseMultilinearExtension<E::ScalarField>>,
            E::ScalarField,
            Option<PCS::Commitment>,
            PCS::ProverCommitmentAdvice,
        )>,
        claims: Vec<E::ScalarField>,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Option<(Self::PermutationProof, Vec<Vec<E::ScalarField>>)>,
            Vec<PCS::ProverCommitmentAdvice>,
            Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>,
        ),
        PolyIOPErrors,
    > {
        let start = start_timer!(|| "Permutation check prove");

        if !Net::am_master() {
            let (advices, polys): (Vec<_>, Vec<_>) = to_prove
                .into_iter()
                .map(|(g_polys, h_poly, _, _, h_advice)| {
                    let mut f_values = vec![E::ScalarField::one(); g_polys.len()];
                    f_values[g_polys.len() / 2..].fill(-E::ScalarField::one());

                    <Self as MultiRationalSumcheck<E::ScalarField>>::d_prove(
                        &f_values,
                        g_polys,
                        Arc::new(DenseMultilinearExtension::clone(&h_poly)),
                        E::ScalarField::zero(),
                        transcript,
                    )
                    .unwrap();
                    (h_advice, h_poly)
                })
                .unzip();

            end_timer!(start);
            return Ok((None, advices, polys));
        }

        let (proofs, points, comms, advices, polys): (Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>) =
            to_prove
                .into_iter()
                .zip(claims)
                .map(|((g_polys, h_poly, _, h_comm, h_advice), claim)| {
                    let mut f_values = vec![E::ScalarField::one(); g_polys.len()];
                    f_values[g_polys.len() / 2..].fill(-E::ScalarField::one());

                    let (proof, point) = <Self as MultiRationalSumcheck<E::ScalarField>>::d_prove(
                        &f_values,
                        g_polys,
                        Arc::new(DenseMultilinearExtension::clone(&h_poly)),
                        claim,
                        transcript,
                    )
                    .unwrap()
                    .unwrap();
                    (proof, point, h_comm, h_advice, h_poly)
                })
                .multiunzip();

        end_timer!(start);

        let comms = comms
            .into_iter()
            .map(|comm| comm.unwrap())
            .collect::<Vec<_>>();
        Ok((
            Some((
                Self::PermutationProof {
                    proofs,
                    h_comms: comms,
                },
                points,
            )),
            advices,
            polys,
        ))
    }

    fn verify(
        proof: &Self::PermutationProof,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::PermutationCheckSubClaim, PolyIOPErrors> {
        let start = start_timer!(|| "Permutation check verify");

        let beta = transcript.get_and_append_challenge(b"beta")?;
        let gamma = transcript.get_and_append_challenge(b"gamma")?;

        let mut subclaims = Vec::with_capacity(proof.proofs.len());
        let mut claimed_sum = E::ScalarField::zero();
        for proof in proof.proofs.iter() {
            claimed_sum += proof.claimed_sum;
            let subclaim =
                <Self as MultiRationalSumcheck<E::ScalarField>>::verify(proof, transcript)?;
            subclaims.push((subclaim, proof.num_polys / 2));
        }

        if claimed_sum != E::ScalarField::zero() {
            return Err(PolyIOPErrors::InvalidProof(format!(
                "Claimed sums do not add to zero",
            )));
        }

        end_timer!(start);
        Ok(PermutationCheckSubClaim {
            subclaims,
            challenges: (beta, gamma),
        })
    }

    fn check_openings(
        subclaim: &Self::PermutationCheckSubClaim,
        f_openings: &[E::ScalarField],
        g_openings: &[E::ScalarField],
        h_openings: &[E::ScalarField],
        perm_openings: &[E::ScalarField],
    ) -> Result<(), PolyIOPErrors> {
        let (beta, gamma) = subclaim.challenges;

        let mut shift = 0;
        let mut offset = 0;
        for (subclaim_idx, (subclaim, len)) in subclaim.subclaims.iter().enumerate() {
            let num_vars = subclaim.sumcheck_point.len();

            let sid: E::ScalarField = (0..num_vars)
                .map(|i| {
                    E::ScalarField::from_u64(i.pow2() as u64).unwrap() * subclaim.sumcheck_point[i]
                })
                .sum::<E::ScalarField>()
                + E::ScalarField::from_u64(shift as u64).unwrap();

            let eq_eval = eq_eval(&subclaim.sumcheck_point, &subclaim.zerocheck_r)?;
            let g_evals = f_openings[offset..offset + len]
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    *f + beta * (sid + E::ScalarField::from((i * (1 << num_vars)) as u64)) + gamma
                })
                .chain(
                    g_openings[offset..offset + len]
                        .iter()
                        .zip(perm_openings[offset..offset + len].iter())
                        .map(|(g, perm)| *g + beta * perm + gamma),
                )
                .collect::<Vec<_>>();
            let g_products = products_except_self(&g_evals);
            let sum = h_openings[subclaim_idx]
                + subclaim.coeff
                    * eq_eval
                    * (g_products[0] * g_evals[0] * h_openings[subclaim_idx]
                        - g_products[..*len].iter().sum::<E::ScalarField>()
                        + g_products[*len..].iter().sum::<E::ScalarField>());

            if sum != subclaim.sumcheck_expected_evaluation {
                return Err(PolyIOPErrors::InvalidVerifier("wrong subclaim".to_string()));
            }

            shift += len * num_vars.pow2();
            offset += len;
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::PermutationCheck;
    use crate::{
        poly_iop::{errors::PolyIOPErrors, PolyIOP},
        MultilinearKzgPCS, PolynomialCommitmentScheme,
    };
    use arithmetic::{
        evaluate_opt, identity_permutation_mle, identity_permutation_mles, math::Math,
        random_permutation_u64,
    };
    use ark_bn254::{Bn254, Fr};
    use ark_ec::pairing::Pairing;
    use ark_ff::PrimeField;
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::test_rng;
    use rand_core::RngCore;
    use std::sync::Arc;

    fn test_permutation_check_helper<
        E: Pairing,
        PCS: PolynomialCommitmentScheme<E, Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>>,
    >(
        fxs: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
        gxs: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
        perms: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
        pcs_param: &PCS::ProverParam,
    ) -> Result<(), PolyIOPErrors> {
        // prover
        let mut transcript =
            <PolyIOP<E::ScalarField> as PermutationCheck<E, PCS>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;
        let (proof, _, _, h_polys) =
            <PolyIOP<E::ScalarField> as PermutationCheck<E, PCS>>::prove(
                pcs_param,
                fxs,
                gxs,
                perms,
                &mut transcript,
            )?;

        // verifier
        let mut transcript =
            <PolyIOP<E::ScalarField> as PermutationCheck<E, PCS>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;
        let perm_check_sub_claim =
            <PolyIOP<E::ScalarField> as PermutationCheck<E, PCS>>::verify(&proof, &mut transcript)?;

        let mut f_openings = vec![];
        let mut g_openings = vec![];
        let mut h_openings = vec![];
        let mut perm_openings = vec![];
        let mut offset = 0;
        for (idx, (subclaim, len)) in perm_check_sub_claim.subclaims.iter().enumerate() {
            let mut f_evals = fxs[offset..offset + len]
                .iter()
                .map(|f| evaluate_opt(f, &subclaim.sumcheck_point))
                .collect::<Vec<_>>();
            let mut g_evals = gxs[offset..offset + len]
                .iter()
                .map(|g| evaluate_opt(g, &subclaim.sumcheck_point))
                .collect::<Vec<_>>();
            let h_eval = evaluate_opt(&h_polys[idx], &subclaim.sumcheck_point);
            let mut perm_evals = perms[offset..offset + len]
                .iter()
                .map(|perm| evaluate_opt(perm, &subclaim.sumcheck_point))
                .collect::<Vec<_>>();

            f_openings.append(&mut f_evals);
            g_openings.append(&mut g_evals);
            h_openings.push(h_eval);
            perm_openings.append(&mut perm_evals);
            offset += len;
        }

        <PolyIOP<E::ScalarField> as PermutationCheck<E, PCS>>::check_openings(
            &perm_check_sub_claim,
            &f_openings,
            &g_openings,
            &h_openings,
            &perm_openings,
        )
    }

    fn generate_polys<R: RngCore>(
        nv: &[usize],
        rng: &mut R,
    ) -> Vec<Arc<DenseMultilinearExtension<Fr>>> {
        nv.iter()
            .map(|x| Arc::new(DenseMultilinearExtension::rand(*x, rng)))
            .collect()
    }

    fn test_permutation_check(
        nv: Vec<usize>,
        id_perms: Vec<Arc<DenseMultilinearExtension<Fr>>>,
    ) -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();

        let max_nv = nv.iter().max().unwrap();
        let srs = MultilinearKzgPCS::<Bn254>::gen_srs_for_testing(&mut rng, *max_nv)?;
        let (pcs_param, _) = MultilinearKzgPCS::<Bn254>::trim(&srs, None, Some(*max_nv))?;

        {
            // good path: (w1, w2) is a permutation of (w1, w2) itself under the identify
            // map
            let ws = generate_polys(&nv, &mut rng);
            // perms is the identity map
            test_permutation_check_helper::<Bn254, MultilinearKzgPCS<Bn254>>(
                &ws, &ws, &id_perms, &pcs_param,
            )?;
        }

        {
            let fs = generate_polys(&nv, &mut rng);

            let size0 = nv[0].pow2();

            let perm = random_permutation_u64(nv[0].pow2() + nv[1].pow2(), &mut rng);
            let perms = vec![
                Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                    nv[0],
                    perm[..size0]
                        .iter()
                        .map(|x| Fr::from_u64(*x).unwrap())
                        .collect(),
                )),
                Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                    nv[1],
                    perm[size0..]
                        .iter()
                        .map(|x| Fr::from_u64(*x).unwrap())
                        .collect(),
                )),
            ];

            let get_f = |index: usize| {
                if index < size0 {
                    fs[0].evaluations[index]
                } else {
                    fs[1].evaluations[index - size0]
                }
            };

            let g_evals = (
                (0..size0)
                    .map(|x| get_f(perm[x] as usize))
                    .collect::<Vec<_>>(),
                (size0..size0 + nv[1].pow2())
                    .map(|x| get_f(perm[x] as usize))
                    .collect::<Vec<_>>(),
            );
            let gs = vec![
                Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                    nv[0], g_evals.0,
                )),
                Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                    nv[1], g_evals.1,
                )),
            ];
            test_permutation_check_helper::<Bn254, MultilinearKzgPCS<Bn254>>(
                &fs, &gs, &perms, &pcs_param,
            )?;
        }

        {
            // bad path 1: w is a not permutation of w itself under a random map
            let ws = generate_polys(&nv, &mut rng);
            // perms is a random map
            let perms = id_perms
                .iter()
                .map(|perm| {
                    let mut evals = perm.evaluations.clone();
                    evals.reverse();
                    Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                        perm.num_vars(),
                        evals,
                    ))
                })
                .collect::<Vec<_>>();

            assert!(
                test_permutation_check_helper::<Bn254, MultilinearKzgPCS::<Bn254>>(
                    &ws, &ws, &perms, &pcs_param
                )
                .is_err()
            );
        }

        {
            // bad path 2: f is a not permutation of g under a identity map
            let fs = generate_polys(&nv, &mut rng);
            let gs = generate_polys(&nv, &mut rng);
            // s_perm is the identity map

            assert!(
                test_permutation_check_helper::<Bn254, MultilinearKzgPCS::<Bn254>>(
                    &fs, &gs, &id_perms, &pcs_param
                )
                .is_err()
            );
        }

        Ok(())
    }

    #[test]
    fn test_trivial_polynomial() -> Result<(), PolyIOPErrors> {
        let id_perms = identity_permutation_mles(1, 2);
        test_permutation_check(vec![1, 1], id_perms)
    }
    #[test]
    fn test_normal_polynomial() -> Result<(), PolyIOPErrors> {
        let id_perms = identity_permutation_mles(5, 2);
        test_permutation_check(vec![5, 5], id_perms)
    }

    #[test]
    fn test_different_lengths() -> Result<(), PolyIOPErrors> {
        let id_perms = vec![
            identity_permutation_mle(0, 5),
            identity_permutation_mle(32, 4),
        ];
        test_permutation_check(vec![5, 4], id_perms)
    }
}

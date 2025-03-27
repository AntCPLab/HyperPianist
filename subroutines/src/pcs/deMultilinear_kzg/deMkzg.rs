//! Main module for distributed multilinear KZG commitment scheme

use super::batching::{batch_verify_internal, d_multi_open_internal};
use crate::{
    pcs::{
        multilinear_kzg::{
            srs::{MultilinearProverParam, MultilinearUniversalParams, MultilinearVerifierParam},
            MultilinearKzgProof,
        },
        prelude::Commitment,
        PCSError, PolynomialCommitmentScheme, StructuredReferenceString,
    },
    BatchProof,
};
use arithmetic::{evaluate_opt, math::Math, unsafe_allocate_zero_vec};
use ark_ec::{
    pairing::Pairing,
    scalar_mul::{fixed_base::FixedBase, variable_base::VariableBaseMSM},
    AffineRepr, CurveGroup,
};
use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    borrow::Borrow, end_timer, format, marker::PhantomData, rand::Rng, start_timer,
    string::ToString, sync::Arc, vec, vec::Vec, One, Zero,
};
use std::ops::Mul;
use transcript::IOPTranscript;

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};

/// KZG Polynomial Commitment Scheme on multilinear polynomials.
pub struct DeMkzg<E: Pairing> {
    #[doc(hidden)]
    phantom: PhantomData<E>,
}

#[derive(Debug, Clone)]
pub enum DeMkzgSRS<E: Pairing> {
    Unprocessed(MultilinearUniversalParams<E>),
    Processed((MultilinearProverParam<E>, MultilinearVerifierParam<E>)),
}

#[derive(Default, Clone, CanonicalSerialize, CanonicalDeserialize, PartialEq, Eq, Debug)]
pub struct SentToMasterData<E: Pairing> {
    pub F_vec: Vec<E::ScalarField>,
    pub G1_vec: Vec<E::G1Affine>,
}

impl<E: Pairing> SentToMasterData<E> {
    pub(super) fn new() -> Self {
        Self {
            F_vec: Vec::new(),
            G1_vec: Vec::new(),
        }
    }
}

impl<E: Pairing> PolynomialCommitmentScheme<E> for DeMkzg<E> {
    // Parameters
    type ProverParam = MultilinearProverParam<E>;
    type VerifierParam = MultilinearVerifierParam<E>;
    type SRS = DeMkzgSRS<E>;
    // Polynomial and its associated types
    type Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>;
    type ProverCommitmentAdvice = ();
    type Point = Vec<E::ScalarField>;
    type Evaluation = E::ScalarField;
    // Commitments and proofs
    type Commitment = Commitment<E>;
    type Proof = Option<MultilinearKzgProof<E>>;
    type BatchProof = BatchProof<E, Self>;

    /// Build SRS for testing.
    ///
    /// - For univariate polynomials, `log_size` is the log of maximum degree.
    /// - For multilinear polynomials, `log_size` is the number of variables.
    ///
    /// WARNING: THIS FUNCTION IS FOR TESTING PURPOSE ONLY.
    /// THE OUTPUT SRS SHOULD NOT BE USED IN PRODUCTION.
    fn gen_srs_for_testing<R: Rng>(rng: &mut R, log_size: usize) -> Result<Self::SRS, PCSError> {
        Ok(DeMkzgSRS::Unprocessed(
            MultilinearUniversalParams::<E>::gen_srs_for_testing(rng, log_size).unwrap(),
        ))
    }

    /// Trim the universal parameters to specialize the public parameters.
    /// Input both `supported_log_degree` for univariate and
    /// `supported_num_vars` for multilinear.
    fn trim(
        srs: impl Borrow<Self::SRS>,
        supported_degree: Option<usize>,
        supported_num_vars: Option<usize>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        assert!(supported_degree.is_none());

        let supported_num_vars = match supported_num_vars {
            Some(p) => p,
            None => {
                return Err(PCSError::InvalidParameters(
                    "multilinear should receive a num_var param".to_string(),
                ))
            },
        };

        Ok(match srs.borrow() {
            DeMkzgSRS::Unprocessed(pp) => pp.trim(supported_num_vars)?,
            DeMkzgSRS::Processed((prover, verifier)) => (prover.clone(), verifier.clone()),
        })
    }

    /// Generate a commitment for a polynomial.
    ///
    /// This function takes `2^num_vars` number of scalar multiplications over
    /// G1.
    fn d_commit(
        sub_prover_param: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
    ) -> Result<(Option<Self::Commitment>, Self::ProverCommitmentAdvice), PCSError> {
        let sub_prover_param = sub_prover_param.borrow();
        let commit_timer = start_timer!(|| "commit");

        let sub_prover_id = Net::party_id();
        let M = Net::n_parties();
        let m = M.log_2();
        let total_nv = m + poly.num_vars;
        if sub_prover_param.num_vars < total_nv {
            return Err(PCSError::InvalidParameters(format!(
                "MlE length ({}) exceeds param limit ({})",
                total_nv, sub_prover_param.num_vars
            )));
        }
        let ignored = sub_prover_param.num_vars - total_nv;

        let sub_g_powers_size = 1usize << poly.num_vars;
        let msm_timer = start_timer!(|| format!("msm of size {}", sub_g_powers_size,));
        let start = sub_prover_id * sub_g_powers_size;
        let end = start + sub_g_powers_size;
        let sub_comm = E::G1MSM::msm_unchecked_par_auto(
            &sub_prover_param.powers_of_g[ignored].evals[start..end],
            &poly.evaluations,
        )
        .into();
        end_timer!(msm_timer);

        let sub_comms = Net::send_to_master(&sub_comm);
        if Net::am_master() {
            let agg_timer = start_timer!(|| "aggregation");
            let comm = sub_comms
                .unwrap()
                .iter()
                .fold(E::G1MSM::zero(), |acc, x| acc + x)
                .into();
            end_timer!(agg_timer);
            end_timer!(commit_timer);
            Ok((Some(Commitment(comm)), ()))
        } else {
            end_timer!(commit_timer);
            Ok((None, ()))
        }
    }

    fn batch_d_commit(
        sub_prover_param: impl Borrow<Self::ProverParam>,
        polys: &[Self::Polynomial],
    ) -> Result<
        (
            Vec<Option<Self::Commitment>>,
            Vec<Self::ProverCommitmentAdvice>,
        ),
        PCSError,
    > {
        let commit_timer = start_timer!(|| "batch_d_commit");

        let sub_prover_param = sub_prover_param.borrow();
        let sub_prover_id = Net::party_id();
        let M = Net::n_parties();
        let m = M.log_2();
        let total_nv = m + polys[0].num_vars;
        if sub_prover_param.num_vars < total_nv {
            return Err(PCSError::InvalidParameters(format!(
                "MlE length ({}) exceeds param limit ({})",
                total_nv, sub_prover_param.num_vars
            )));
        }
        let ignored = sub_prover_param.num_vars - total_nv;

        let sub_g_powers_size = 1usize << polys[0].num_vars;
        let start = sub_prover_id * sub_g_powers_size;
        let end = start + sub_g_powers_size;

        let sub_comms = polys
            .iter()
            .map(|poly| {
                E::G1MSM::msm_unchecked_par_auto(
                    &sub_prover_param.powers_of_g[ignored].evals[start..end],
                    &poly.evaluations,
                )
                .into()
            })
            .collect::<Vec<_>>();

        let all_sub_comms = Net::send_to_master(&sub_comms);
        if Net::am_master() {
            let agg_timer = start_timer!(|| "aggregation");

            let all_sub_comms = all_sub_comms.unwrap();
            let comms = (0..all_sub_comms[0].len())
                .into_par_iter()
                .map(|i| {
                    Some(Commitment(
                        all_sub_comms
                            .iter()
                            .map(|sub_comms| sub_comms[i])
                            .fold(E::G1MSM::zero(), |acc, x| acc + x)
                            .into(),
                    ))
                })
                .collect::<Vec<_>>();

            end_timer!(agg_timer);
            end_timer!(commit_timer);
            Ok((comms, vec![(); polys.len()]))
        } else {
            end_timer!(commit_timer);
            Ok((vec![None; polys.len()], vec![(); polys.len()]))
        }
    }

    /// On input a polynomial `p` and a point `point`, outputs a proof for the
    /// same. This function does not need to take the evaluation value as an
    /// input.
    ///
    /// This function takes 2^{num_var +1} number of scalar multiplications over
    /// G1:
    /// - it prodceeds with `num_var` number of rounds,
    /// - at round i, we compute an MSM for `2^{num_var - i + 1}` number of G2
    ///   elements.
    fn open(
        sub_prover_param: impl Borrow<Self::ProverParam>,
        polynomial: &Self::Polynomial,
        _advice: &Self::ProverCommitmentAdvice,
        point: &Self::Point,
    ) -> Result<Self::Proof, PCSError> {
        d_open_internal(sub_prover_param.borrow(), polynomial, point)
    }

    /// Input a list of multilinear extensions, and a same number of points, and
    /// a transcript, compute a multi-opening for all the polynomials.
    fn d_multi_open(
        sub_prover_param: impl Borrow<Self::ProverParam>,
        polynomials: Vec<Self::Polynomial>,
        _advices: &[Self::ProverCommitmentAdvice],
        points: &[Self::Point],
        evals: &[Self::Evaluation],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<Option<BatchProof<E, Self>>, PCSError> {
        d_multi_open_internal(
            sub_prover_param.borrow(),
            polynomials,
            points,
            evals,
            transcript,
        )
    }

    /// Verifies that `value` is the evaluation at `x` of the polynomial
    /// committed inside `comm`.
    ///
    /// This function takes
    /// - num_var number of pairing product.
    /// - num_var number of MSM
    fn verify(
        verifier_param: &Self::VerifierParam,
        commitment: &Self::Commitment,
        point: &Self::Point,
        value: &E::ScalarField,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        verify_internal(
            verifier_param,
            commitment,
            point,
            value,
            &proof.as_ref().unwrap().clone(),
        )
    }

    /// Verifies that `value_i` is the evaluation at `x_i` of the polynomial
    /// `poly_i` committed inside `comm`.
    fn batch_verify(
        verifier_param: &Self::VerifierParam,
        commitments: &[Self::Commitment],
        points: &[Self::Point],
        batch_proof: &Self::BatchProof,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError> {
        batch_verify_internal(verifier_param, commitments, points, batch_proof, transcript)
    }
}

/// On input a polynomial `p` and a point `point`, outputs a proof for the
/// same. This function does not need to take the evaluation value as an
/// input.
///
/// This function takes 2^{num_var} number of scalar multiplications over
/// G1:
/// - it proceeds with `num_var` number of rounds,
/// - at round i, we compute an MSM for `2^{num_var - i}` number of G1 elements.

fn d_open_internal<E: Pairing>(
    sub_prover_param: &MultilinearProverParam<E>,
    polynomial: &DenseMultilinearExtension<E::ScalarField>,
    point: &[E::ScalarField],
) -> Result<Option<MultilinearKzgProof<E>>, PCSError> {
    let open_timer =
        start_timer!(|| format!("subprovre open mle with {} variable", polynomial.num_vars));

    let sub_prover_id = Net::party_id();
    let M = Net::n_parties();
    let m = M.log_2();
    let sub_nv = polynomial.num_vars;
    let total_nv = m + sub_nv;
    if sub_prover_param.num_vars < total_nv {
        return Err(PCSError::InvalidParameters(format!(
            "MlE length ({}) exceeds param limit ({})",
            total_nv, sub_prover_param.num_vars
        )));
    }
    // From g^eq(t,x) with total_nv - 1 variables
    let ignored = sub_prover_param.num_vars - total_nv + 1;
    let mut sub_qs_comms = Vec::new();
    let mut f = polynomial.to_evaluations();
    let mut q = unsafe_allocate_zero_vec::<E::ScalarField>(1 << (sub_nv - 1));
    let mut r = unsafe_allocate_zero_vec::<E::ScalarField>(1 << (sub_nv - 1));

    for (i, (&point_at_k, gi)) in point[..sub_nv]
        .iter()
        .zip(sub_prover_param.powers_of_g[ignored..ignored + sub_nv].iter())
        .enumerate()
    {
        let k = sub_nv - 1 - i;
        let cur_dim = 1 << k;
        let ith_round = start_timer!(|| format!("{}-th round", i));

        let ith_round_eval = start_timer!(|| format!("{}-th round eval", i));
        q[..cur_dim]
            .par_iter_mut()
            .zip(r[..cur_dim].par_iter_mut())
            .enumerate()
            .for_each(|(b, (q, r))| {
                *q = f[(b << 1) + 1] - f[b << 1];
                *r = f[b << 1] + (*q * point_at_k)
            });
        (r, f) = (f, r);

        end_timer!(ith_round_eval);

        // this is a MSM over G1 and is likely to be the bottleneck
        let msm_timer = start_timer!(|| format!("msm of size {} at round {}", cur_dim, i));

        let start = sub_prover_id * cur_dim;
        let end = start + cur_dim;
        sub_qs_comms
            .push(E::G1MSM::msm_unchecked_par_auto(&gi.evals[start..end], &q[..cur_dim]).into());
        end_timer!(msm_timer);

        end_timer!(ith_round);
    }

    let sub_data: SentToMasterData<E> = SentToMasterData {
        F_vec: vec![f[0]],
        G1_vec: sub_qs_comms,
    };

    drop(q);
    drop(r);
    drop(f);

    let sub_data_vec = Net::send_to_master(&sub_data);

    end_timer!(open_timer);

    if Net::am_master() {
        let agg_timer = start_timer!(|| "aggregation");

        // let mut proofs = vec![E::G1Affine::zero(); nv];
        let sub_data_vec = sub_data_vec.unwrap();

        let mut proofs = (0..sub_nv)
            .into_par_iter()
            .map(|i| {
                sub_data_vec
                    .iter()
                    .map(|pi_data| pi_data.G1_vec[i])
                    .fold(E::G1MSM::zero(), |acc, x| acc + x)
                    .into()
            })
            .collect::<Vec<E::G1Affine>>();

        let mut f = sub_data_vec
            .iter()
            .map(|pi_data| pi_data.F_vec[0])
            .collect::<Vec<E::ScalarField>>();
        let mut q = unsafe_allocate_zero_vec::<E::ScalarField>(1 << (m - 1));
        let mut r = unsafe_allocate_zero_vec::<E::ScalarField>(1 << (m - 1));

        for (i, (&point_at_k, gi)) in point[sub_nv..]
            .iter()
            .zip(sub_prover_param.powers_of_g[ignored + sub_nv..].iter())
            .enumerate()
        {
            let ith_round = start_timer!(|| format!("{}-th round", i));

            let k = m - 1 - i;
            let cur_dim = 1 << k;

            let ith_round_eval = start_timer!(|| format!("{}-th round eval for master prover", i));
            q[..cur_dim]
                .par_iter_mut()
                .zip(r[..cur_dim].par_iter_mut())
                .enumerate()
                .for_each(|(b, (q, r))| {
                    *q = f[(b << 1) + 1] - f[b << 1];
                    *r = f[b << 1] + (*q * point_at_k)
                });
            (r, f) = (f, r);
            end_timer!(ith_round_eval);

            // this is a MSM over G1 and is likely to be the bottleneck
            let msm_timer = start_timer!(|| format!("msm of size {} at round {}", cur_dim, i));

            proofs.push(E::G1MSM::msm_unchecked_par_auto(&gi.evals, &q[..cur_dim]).into());
            end_timer!(msm_timer);

            end_timer!(ith_round);
        }

        end_timer!(agg_timer);
        Ok(Some(MultilinearKzgProof { proofs }))
    } else {
        Ok(None)
    }
}

/// Verifies that `value` is the evaluation at `x` of the polynomial
/// committed inside `comm`.
///
/// This function takes
/// - num_var number of pairing product.
/// - num_var number of MSM
fn verify_internal<E: Pairing>(
    verifier_param: &MultilinearVerifierParam<E>,
    commitment: &Commitment<E>,
    point: &[E::ScalarField],
    value: &E::ScalarField,
    proof: &MultilinearKzgProof<E>,
) -> Result<bool, PCSError> {
    let verify_timer = start_timer!(|| "verify");
    let num_var = point.len();

    if num_var > verifier_param.num_vars {
        return Err(PCSError::InvalidParameters(format!(
            "point length ({}) exceeds param limit ({})",
            num_var, verifier_param.num_vars
        )));
    }

    let prepare_inputs_timer = start_timer!(|| "prepare pairing inputs");

    let scalar_size = E::ScalarField::MODULUS_BIT_SIZE as usize;
    let window_size = FixedBase::get_mul_window_size(num_var);

    let h_table =
        FixedBase::get_window_table(scalar_size, window_size, verifier_param.h.into_group());
    let h_mul: Vec<E::G2> = FixedBase::msm(scalar_size, window_size, &h_table, point);

    let ignored = verifier_param.num_vars - num_var;
    let h_vec: Vec<_> = (0..num_var)
        .map(|i| -h_mul[i] + verifier_param.h_mask[ignored + i])
        .collect();
    let h_vec: Vec<E::G2Affine> = E::G2::normalize_batch(&h_vec);
    end_timer!(prepare_inputs_timer);

    let pairing_product_timer = start_timer!(|| "pairing product");

    let mut ps = proof
        .proofs
        .iter()
        .map(|&x| E::G1Prepared::from(x))
        .collect::<Vec<_>>();

    let mut hs = h_vec
        .into_par_iter()
        .take(num_var)
        .map(E::G2Prepared::from)
        .collect::<Vec<_>>();

    ps.push(E::G1Prepared::from(
        (verifier_param.g.mul(*value) - commitment.0.into_group()).into_affine(),
    ));
    hs.push(E::G2Prepared::from(verifier_param.h));

    let res = E::multi_pairing(ps, hs) == ark_ec::pairing::PairingOutput(E::TargetField::one());

    end_timer!(pairing_product_timer);
    end_timer!(verify_timer);
    Ok(res)
}

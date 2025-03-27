use crate::{
    barycentric_weights, extrapolate,
    pcs::PolynomialCommitmentScheme,
    poly_iop::{errors::PolyIOPErrors, perm_check::util::compute_leaves, PolyIOP},
    ZerocheckInstanceProof,
};
use arithmetic::{
    bind_poly_var_bot_par, bit_decompose, build_eq_table, eq_eval, interpolate_uni_poly,
    math::Math, products_except_self, unipoly::UniPoly,
};
use ark_ec::pairing::Pairing;
use ark_ff::{batch_inversion, Field, PrimeField, Zero};
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{end_timer, start_timer};
use itertools::{izip, zip_eq};
use rayon::prelude::*;
use std::{iter::zip, mem::take, sync::Arc};
use transcript::IOPTranscript;

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

pub trait CombinedCheck<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    type MultilinearExtension;
    type CombinedCheckSubClaim;
    type Transcript;
    type CombinedCheckProof: CanonicalSerialize + CanonicalDeserialize;

    /// Initialize the system with a transcript
    ///
    /// This function is optional -- in the case where a CombinedCheck
    /// is an building block for a more complex protocol, the transcript
    /// may be initialized by this complex protocol, and passed to the
    /// CombinedCheck prover/verifier.
    fn init_transcript() -> Self::Transcript;

    fn prove_prepare(
        prover_param: &PCS::ProverParam,
        witness: &[Self::MultilinearExtension],
        perms: &[Self::MultilinearExtension],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Arc<DenseMultilinearExtension<E::ScalarField>>,
            PCS::Commitment,
            PCS::ProverCommitmentAdvice,
            E::ScalarField,
            E::ScalarField,
        ),
        PolyIOPErrors,
    >;

    fn d_prove_prepare(
        prover_param: &PCS::ProverParam,
        witness: &[Self::MultilinearExtension],
        perms: &[Self::MultilinearExtension],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Arc<DenseMultilinearExtension<E::ScalarField>>,
            Option<PCS::Commitment>,
            PCS::ProverCommitmentAdvice,
            E::ScalarField,
            E::ScalarField,
        ),
        PolyIOPErrors,
    >;

    fn prove(
        to_prove: (
            Arc<DenseMultilinearExtension<E::ScalarField>>,
            PCS::Commitment,
            PCS::ProverCommitmentAdvice,
            E::ScalarField,
            E::ScalarField,
        ),
        witness: &[Self::MultilinearExtension],
        perms: &[Self::MultilinearExtension],
        selectors: &[Self::MultilinearExtension],
        gate: &[(Option<usize>, Vec<usize>)],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Self::CombinedCheckProof,
            PCS::ProverCommitmentAdvice,
            Arc<DenseMultilinearExtension<E::ScalarField>>,
            Vec<E::ScalarField>,
        ),
        PolyIOPErrors,
    >;

    fn d_prove(
        to_prove: (
            Arc<DenseMultilinearExtension<E::ScalarField>>,
            Option<PCS::Commitment>,
            PCS::ProverCommitmentAdvice,
            E::ScalarField,
            E::ScalarField,
        ),
        witness: &[Self::MultilinearExtension],
        perms: &[Self::MultilinearExtension],
        selectors: &[Self::MultilinearExtension],
        gate: &[(Option<usize>, Vec<usize>)],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Option<(Self::CombinedCheckProof, Vec<E::ScalarField>)>,
            PCS::ProverCommitmentAdvice,
            Arc<DenseMultilinearExtension<E::ScalarField>>,
        ),
        PolyIOPErrors,
    >;

    fn verify(
        proof: &Self::CombinedCheckProof,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::CombinedCheckSubClaim, PolyIOPErrors>;

    fn check_openings(
        subclaim: &Self::CombinedCheckSubClaim,
        witness_openings: &[E::ScalarField],
        perm_openings: &[E::ScalarField],
        selector_openings: &[E::ScalarField],
        h_opening: &E::ScalarField,
        gate: &[(Option<usize>, Vec<usize>)],
    ) -> Result<(), PolyIOPErrors>;
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct CombinedCheckProof<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub proof: (ZerocheckInstanceProof<E::ScalarField>, Vec<E::ScalarField>),
    pub h_comm: PCS::Commitment,
    pub num_rounds: usize,
    pub degree_bound: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct CombinedCheckSubClaim<F: PrimeField> {
    pub point: Vec<F>,
    pub zerocheck_expected_evaluation: F,
    pub h_expected_evaluation: F,
    pub zerocheck_r: Vec<F>,
    pub coeff: F,
    pub beta: F,
    pub gamma: F,
}

struct CombinedCheckInfo<'a, F: PrimeField> {
    num_witnesses: usize,
    num_selectors: usize,
    gate: &'a [(Option<usize>, Vec<usize>)],
    coeff: F,
    sid_offset: F,
}

// The first element in values is the eq_eval, second element is h
fn combined_check_combine_permcheck<F: PrimeField>(
    mut sid: F,
    values: &[F],
    info: &CombinedCheckInfo<F>,
) -> F {
    let witnesses = &values[..info.num_witnesses];
    let perms = &values[info.num_witnesses..2 * info.num_witnesses];
    let h_eval = values.last().unwrap();

    let mut g_evals = vec![F::zero(); witnesses.len() * 2];
    for i in 0..witnesses.len() {
        // sid contains beta & gamma
        g_evals[i] = witnesses[i] + sid;
        sid += info.sid_offset;
    }
    for i in 0..witnesses.len() {
        // perm contains beta & gamma
        g_evals[witnesses.len() + i] = witnesses[i] + perms[i];
    }
    let g_products = products_except_self(&g_evals);

    // g_products is the product of all the g except self
    let mut sum = F::zero();
    for g_product in &g_products[..witnesses.len()] {
        sum += g_product;
    }
    for g_product in &g_products[witnesses.len()..] {
        sum -= g_product;
    }

    *h_eval * g_products[0] * g_evals[0] - sum
}

fn combined_check_combine_zerocheck<F: PrimeField>(
    _sid: F,
    values: &[F],
    info: &CombinedCheckInfo<F>,
) -> F {
    let witnesses = &values[..info.num_witnesses];
    let selectors = &values[2 * info.num_witnesses..2 * info.num_witnesses + info.num_selectors];

    info.gate
        .iter()
        .map(|(selector, witness_indices)| {
            let mut product = F::one();
            if let Some(selector_idx) = selector {
                product *= selectors[*selector_idx];
            }
            for witness_idx in witness_indices {
                product *= witnesses[*witness_idx];
            }
            product
        })
        .sum::<F>()
}

fn combined_sumcheck_prove_step<'a, F: PrimeField, Func1, Func2>(
    polys: &mut Vec<Arc<DenseMultilinearExtension<F>>>,
    eq_table: &[F],
    sid: F,
    multiplier: F,
    comb_func_1: &Func1,
    combined_degree_1: usize,
    comb_func_2: &Func2,
    combined_degree_2: usize,
    combine_coeff: F,
    mut previous_claim_1: &'a mut F,
    mut previous_claim_2: &'a mut F,
    r: F,
    r_inv: F,
    extrapolation_aux: &(Vec<F>, Vec<F>),
) -> (Vec<F>, Vec<F>, Vec<F>, F)
where
    Func1: Fn(F, &[F]) -> F + std::marker::Sync,
    Func2: Fn(F, &[F]) -> F + std::marker::Sync,
{
    let start = start_timer!(|| "zero check step");
    // Vector storing evaluations of combined polynomials g(x) = P_0(x) * ...
    // P_{num_polys} (x) for points {0, ..., |g(x)|}
    let mle_half = polys[0].evaluations.len() / 2;

    let (mut accum1, mut accum2, h_eval) = (0..mle_half)
        .into_par_iter()
        .fold(
            || {
                (
                    vec![F::zero(); polys.len()],
                    vec![F::zero(); polys.len()],
                    vec![F::zero(); combined_degree_1 + 1],
                    vec![F::zero(); combined_degree_2 + 1],
                    F::zero(),
                )
            },
            |(mut eval, mut step, mut acc1, mut acc2, mut h_eval), b| {
                let mut sid = multiplier * F::from_u64(2 * (b as u64)).unwrap() + sid;

                let eq_eval = eq_table[b];
                izip!(eval.iter_mut(), step.iter_mut(), polys.iter()).for_each(
                    |(eval, step, poly)| {
                        *eval = poly[b << 1];
                        *step = poly[(b << 1) + 1] - poly[b << 1];
                    },
                );
                acc1[0] += comb_func_1(sid, &eval) * eq_eval;
                acc2[0] += comb_func_2(sid, &eval) * eq_eval;
                h_eval += eval.last().unwrap();

                eval.iter_mut()
                    .zip(step.iter())
                    .for_each(|(eval, step)| *eval += step as &_);
                sid += multiplier;
                for eval_i in 2..(std::cmp::max(combined_degree_1, combined_degree_2) + 1) {
                    eval.iter_mut()
                        .zip(step.iter())
                        .for_each(|(eval, step)| *eval += step as &_);
                    sid += multiplier;

                    if eval_i < acc1.len() {
                        acc1[eval_i] += comb_func_1(sid, &eval) * eq_eval;
                    }
                    if eval_i < acc2.len() {
                        acc2[eval_i] += comb_func_2(sid, &eval) * eq_eval;
                    }
                }
                (eval, step, acc1, acc2, h_eval)
            },
        )
        .map(|(_, _, partial1, partial2, partial_heval)| (partial1, partial2, partial_heval))
        .reduce(
            || {
                (
                    vec![F::zero(); combined_degree_1 + 1],
                    vec![F::zero(); combined_degree_2 + 1],
                    F::zero(),
                )
            },
            |(mut sum1, mut sum2, h_eval), (partial1, partial2, partial_h_eval)| {
                sum1.iter_mut()
                    .zip(partial1.iter())
                    .for_each(|(sum, partial)| *sum += partial);
                sum2.iter_mut()
                    .zip(partial2.iter())
                    .for_each(|(sum, partial)| *sum += partial);
                (sum1, sum2, h_eval + partial_h_eval)
            },
        );

    let mut should_swap = accum1.len() < accum2.len();
    if should_swap {
        (accum1, accum2) = (accum2, accum1);
        (previous_claim_1, previous_claim_2) = (previous_claim_2, previous_claim_1);
    }
    accum1[1] = r_inv * (*previous_claim_1 - (F::one() - r) * accum1[0]);
    accum2[1] = r_inv * (*previous_claim_2 - (F::one() - r) * accum2[0]);

    let (points, weights) = extrapolation_aux;
    let evals = accum1
        .iter()
        .zip(
            accum2
                .iter()
                .map(|x| *x)
                .chain((accum2.len()..accum1.len()).map(|i| {
                    let at = F::from(i as u64);
                    extrapolate(points, weights, &accum2, &at)
                })),
        )
        .map(|(sum1, sum2)| {
            if should_swap {
                combine_coeff * *sum1 + sum2
            } else {
                *sum1 + combine_coeff * sum2
            }
        })
        .collect::<Vec<_>>();

    end_timer!(start);
    if should_swap {
        (evals, accum2, accum1, h_eval)
    } else {
        (evals, accum1, accum2, h_eval)
    }
}

fn combined_sumcheck_prove<F: PrimeField, Func1, Func2>(
    claim_1: &F,
    claim_2: &F,
    num_rounds: usize,
    mut multiplier: F,
    mut sid: F,
    polys: &mut Vec<Arc<DenseMultilinearExtension<F>>>,
    zerocheck_r: &[F],
    comb_func_1: Func1,
    combined_degree_1: usize,
    comb_func_2: Func2,
    combined_degree_2: usize,
    extrapolation_aux: &(Vec<F>, Vec<F>),
    combine_coeff: F,
    transcript: &mut IOPTranscript<F>,
) -> ((ZerocheckInstanceProof<F>, Vec<F>), Vec<F>, Vec<F>)
where
    Func1: Fn(F, &[F]) -> F + std::marker::Sync,
    Func2: Fn(F, &[F]) -> F + std::marker::Sync,
{
    let mut zerocheck_r_inv = zerocheck_r.to_vec();
    batch_inversion(&mut zerocheck_r_inv);

    let eq_table = build_eq_table(&zerocheck_r, F::one());

    let mut r: Vec<F> = Vec::new();
    let mut proof_polys: Vec<UniPoly<F>> = Vec::new();
    let mut proof_h_evals = Vec::new();

    let mut previous_claim_1 = claim_1.clone();
    let mut previous_claim_2 = claim_2.clone();

    for round in 0..num_rounds {
        let (eval_points, evals_1, evals_2, eval_h) = combined_sumcheck_prove_step(
            polys,
            &eq_table[round],
            sid,
            multiplier,
            &comb_func_1,
            combined_degree_1,
            &comb_func_2,
            combined_degree_2,
            combine_coeff,
            &mut previous_claim_1,
            &mut previous_claim_2,
            zerocheck_r[round],
            zerocheck_r_inv[round],
            extrapolation_aux,
        );

        let step = start_timer!(|| "from evals");
        let round_uni_poly = UniPoly::from_evals(&eval_points);
        end_timer!(step);

        // append the prover's message to the transcript
        transcript
            .append_serializable_element(b"poly", &round_uni_poly)
            .unwrap();
        transcript.append_field_element(b"eval_h", &eval_h).unwrap();
        let r_j = transcript
            .get_and_append_challenge(b"challenge_nextround")
            .unwrap();
        r.push(r_j);

        sid += r_j * multiplier;
        multiplier.double_in_place();

        // bound all tables to the verifier's challenege
        let step = start_timer!(|| "bind polys");
        let concurrency = (rayon::current_num_threads() * 2 + polys.len() - 1) / polys.len();
        polys
            .par_iter_mut()
            .for_each(|poly| bind_poly_var_bot_par(Arc::get_mut(poly).unwrap(), &r_j, concurrency));
        rayon::join(
            || previous_claim_1 = interpolate_uni_poly(&evals_1, r_j),
            || previous_claim_2 = interpolate_uni_poly(&evals_2, r_j),
        );
        proof_polys.push(round_uni_poly);
        proof_h_evals.push(eval_h);
        end_timer!(step);
    }

    let final_evals = polys.iter().map(|poly| poly[0]).collect::<Vec<_>>();

    (
        (ZerocheckInstanceProof::new(proof_polys), proof_h_evals),
        r,
        final_evals,
    )
}

fn d_combined_sumcheck_prove<F: PrimeField, Func1, Func2>(
    claim_1: &F,
    claim_2: &F,
    num_rounds: usize,
    polys: &mut Vec<Arc<DenseMultilinearExtension<F>>>,
    zerocheck_r: &[F],
    mut multiplier: F,
    sid_init: F,
    comb_func_1: Func1,
    combined_degree_1: usize,
    comb_func_2: Func2,
    combined_degree_2: usize,
    extrapolation_aux: &(Vec<F>, Vec<F>),
    combine_coeff: F,
    transcript: &mut IOPTranscript<F>,
) -> Option<((ZerocheckInstanceProof<F>, Vec<F>), Vec<F>, Vec<F>)>
where
    Func1: Fn(F, &[F]) -> F + std::marker::Sync,
    Func2: Fn(F, &[F]) -> F + std::marker::Sync,
{
    let num_party_vars = Net::n_parties().log_2();

    let index_vec: Vec<F> = bit_decompose(Net::party_id() as u64, num_party_vars)
        .into_iter()
        .map(|x| F::from(x))
        .collect();
    let coeff = eq_eval(&zerocheck_r[num_rounds..], &index_vec).unwrap();

    let eq_table = build_eq_table(&zerocheck_r[..num_rounds], coeff);
    let mut zerocheck_r_inv = zerocheck_r[..num_rounds].to_vec();
    batch_inversion(&mut zerocheck_r_inv);

    let mut r: Vec<F> = Vec::new();
    let mut proof_polys: Vec<UniPoly<F>> = Vec::new();
    let mut proof_h_evals = Vec::new();

    let mut previous_claim_1 = claim_1.clone();
    let mut previous_claim_2 = claim_2.clone();

    let mut sid =
        sid_init + multiplier * F::from_u64((Net::party_id() * num_rounds.pow2()) as u64).unwrap();

    for round in 0..num_rounds {
        let eval_points = combined_sumcheck_prove_step(
            polys,
            &eq_table[round],
            sid,
            multiplier,
            &comb_func_1,
            combined_degree_1,
            &comb_func_2,
            combined_degree_2,
            combine_coeff,
            &mut previous_claim_1,
            &mut previous_claim_2,
            zerocheck_r[round],
            zerocheck_r_inv[round],
            extrapolation_aux,
        );
        let all_eval_points = Net::send_to_master(&eval_points);

        // append the prover's message to the transcript
        let r_j = if Net::am_master() {
            let all_eval_points = all_eval_points.unwrap();
            let (eval_points, evals_1, evals_2, eval_h) = all_eval_points.iter().fold(
                (
                    vec![F::zero(); all_eval_points[0].0.len()],
                    vec![F::zero(); all_eval_points[0].1.len()],
                    vec![F::zero(); all_eval_points[0].2.len()],
                    F::zero(),
                ),
                |(mut evals, mut evals_1, mut evals_2, eval_h),
                 (partial, partial_1, partial_2, partial_h)| {
                    zip_eq(&mut evals, partial).for_each(|(acc, x)| *acc += *x);
                    zip_eq(&mut evals_1, partial_1).for_each(|(acc, x)| *acc += *x);
                    zip_eq(&mut evals_2, partial_2).for_each(|(acc, x)| *acc += *x);
                    (evals, evals_1, evals_2, eval_h + partial_h)
                },
            );

            let step = start_timer!(|| "from evals");
            let round_uni_poly = UniPoly::from_evals(&eval_points);
            end_timer!(step);

            transcript
                .append_serializable_element(b"poly", &round_uni_poly)
                .unwrap();
            transcript.append_field_element(b"eval_h", &eval_h).unwrap();

            let r_j = transcript
                .get_and_append_challenge(b"challenge_nextround")
                .unwrap();
            rayon::join(
                || previous_claim_1 = interpolate_uni_poly(&evals_1, r_j),
                || previous_claim_2 = interpolate_uni_poly(&evals_2, r_j),
            );
            proof_polys.push(round_uni_poly);
            proof_h_evals.push(eval_h);
            Net::recv_from_master_uniform(Some(r_j))
        } else {
            Net::recv_from_master_uniform(None)
        };
        r.push(r_j);

        sid += r_j * multiplier;
        multiplier.double_in_place();

        // bound all tables to the verifier's challenege
        let step = start_timer!(|| "bind polys");
        let concurrency = (rayon::current_num_threads() * 2 + polys.len() - 1) / polys.len();
        polys
            .par_iter_mut()
            .for_each(|poly| bind_poly_var_bot_par(Arc::get_mut(poly).unwrap(), &r_j, concurrency));
        end_timer!(step);
    }

    let final_evals = polys.iter().map(|poly| poly[0]).collect::<Vec<_>>();
    let all_final_evals = Net::send_to_master(&final_evals);

    if !Net::am_master() {
        return None;
    }

    let all_final_evals = all_final_evals.unwrap();
    let mut polys = (0..all_final_evals[0].len())
        .into_par_iter()
        .map(|poly_id| {
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_party_vars,
                all_final_evals
                    .iter()
                    .map(|party_evals| party_evals[poly_id])
                    .collect(),
            ))
        })
        .collect::<Vec<_>>();
    let ((mut proof, mut h_evals), mut r_final, final_evals) = combined_sumcheck_prove(
        &previous_claim_1,
        &previous_claim_2,
        num_party_vars,
        multiplier,
        sid,
        &mut polys,
        &zerocheck_r[num_rounds..],
        comb_func_1,
        combined_degree_1,
        comb_func_2,
        combined_degree_2,
        extrapolation_aux,
        combine_coeff,
        transcript,
    );
    proof_polys.append(&mut proof.polys);
    proof_h_evals.append(&mut h_evals);
    r.append(&mut r_final);

    Some((
        (ZerocheckInstanceProof::new(proof_polys), proof_h_evals),
        r,
        final_evals,
    ))
}

fn combined_sumcheck_verify<F: PrimeField>(
    proof: &(ZerocheckInstanceProof<F>, Vec<F>),
    num_rounds: usize,
    degree_bound: usize,
    zerocheck_r: &[F],
    transcript: &mut IOPTranscript<F>,
) -> Result<(F, F, Vec<F>), PolyIOPErrors> {
    let mut e = F::zero();
    let mut e2 = F::zero();
    let mut r: Vec<F> = Vec::new();

    let (proof, h_evals) = proof;
    // verify that there is a univariate polynomial for each round
    assert_eq!(proof.polys.len(), num_rounds);
    for i in 0..proof.polys.len() {
        let poly = &proof.polys[i];
        // append the prover's message to the transcript
        transcript.append_serializable_element(b"poly", poly)?;
        transcript.append_field_element(b"eval_h", &h_evals[i])?;

        // verify degree bound
        if poly.degree() != degree_bound {
            return Err(PolyIOPErrors::InvalidProof(format!(
                "degree_bound = {}, poly.degree() = {}",
                degree_bound,
                poly.degree(),
            )));
        }

        if poly.coeffs[0] + zerocheck_r[i] * (poly.coeffs.iter().skip(1).sum::<F>()) != e {
            return Err(PolyIOPErrors::InvalidProof(
                "Inconsistent message".to_string(),
            ));
        }

        // derive the verifier's challenge for the next round
        let r_i = transcript.get_and_append_challenge(b"challenge_nextround")?;

        r.push(r_i);

        // evaluate the claimed degree-ell polynomial at r_i
        e = poly.evaluate(&r_i);

        // (eval_h) + r_i (eval_1 - eval_h), eval_1 = e2 - eval_h
        e2 = h_evals[i] + r_i * (e2 - h_evals[i] - h_evals[i]);
    }

    Ok((e, e2, r))
}

impl<E, PCS> CombinedCheck<E, PCS> for PolyIOP<E::ScalarField>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E, Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>>,
{
    type MultilinearExtension = Arc<DenseMultilinearExtension<E::ScalarField>>;
    type Transcript = IOPTranscript<E::ScalarField>;
    type CombinedCheckSubClaim = CombinedCheckSubClaim<E::ScalarField>;
    type CombinedCheckProof = CombinedCheckProof<E, PCS>;

    fn init_transcript() -> Self::Transcript {
        IOPTranscript::<E::ScalarField>::new(b"Initializing CombinedCheck transcript")
    }

    fn prove_prepare(
        prover_param: &PCS::ProverParam,
        witness: &[Self::MultilinearExtension],
        perms: &[Self::MultilinearExtension],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Arc<DenseMultilinearExtension<E::ScalarField>>,
            PCS::Commitment,
            PCS::ProverCommitmentAdvice,
            E::ScalarField,
            E::ScalarField,
        ),
        PolyIOPErrors,
    > {
        let start = start_timer!(|| "prove prepare");
        let beta = transcript.get_and_append_challenge(b"beta")?;
        let gamma = transcript.get_and_append_challenge(b"gamma")?;

        let mut leaves =
            compute_leaves::<E::ScalarField, false>(&beta, &gamma, witness, witness, perms)?;
        let leaves_len = leaves.len();
        let mut leave = take(&mut leaves[0]);
        assert_eq!(leaves_len, 1);

        let half_len = leave.len() / 2;
        let nv = leave[0].len().log_2();
        leave.par_iter_mut().for_each(|evals| {
            batch_inversion(evals);
        });
        let h_evals = (0..leave[0].len())
            .into_par_iter()
            .map(|i| {
                leave[..half_len]
                    .iter()
                    .map(|eval| eval[i])
                    .sum::<E::ScalarField>()
                    - leave[half_len..]
                        .iter()
                        .map(|eval| eval[i])
                        .sum::<E::ScalarField>()
            })
            .collect::<Vec<_>>();
        let h_poly = Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, h_evals));
        let (h_comm, h_advice) = PCS::commit(prover_param, &h_poly).unwrap();

        end_timer!(start);

        Ok((h_poly, h_comm, h_advice, beta, gamma))
    }

    fn d_prove_prepare(
        prover_param: &PCS::ProverParam,
        witness: &[Self::MultilinearExtension],
        perms: &[Self::MultilinearExtension],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Arc<DenseMultilinearExtension<E::ScalarField>>,
            Option<PCS::Commitment>,
            PCS::ProverCommitmentAdvice,
            E::ScalarField,
            E::ScalarField,
        ),
        PolyIOPErrors,
    > {
        let start = start_timer!(|| "prove prepare");
        let (beta, gamma) = if Net::am_master() {
            let beta = transcript.get_and_append_challenge(b"beta")?;
            let gamma = transcript.get_and_append_challenge(b"gamma")?;
            Net::recv_from_master_uniform(Some((beta, gamma)))
        } else {
            Net::recv_from_master_uniform(None)
        };

        let mut leaves =
            compute_leaves::<E::ScalarField, true>(&beta, &gamma, witness, witness, perms)?;
        let leaves_len = leaves.len();
        let mut leave = take(&mut leaves[0]);
        assert_eq!(leaves_len, 1);

        let half_len = leave.len() / 2;
        let nv = leave[0].len().log_2();
        leave.par_iter_mut().for_each(|evals| {
            batch_inversion(evals);
        });
        let h_evals = (0..leave[0].len())
            .into_par_iter()
            .map(|i| {
                leave[..half_len]
                    .iter()
                    .map(|eval| eval[i])
                    .sum::<E::ScalarField>()
                    - leave[half_len..]
                        .iter()
                        .map(|eval| eval[i])
                        .sum::<E::ScalarField>()
            })
            .collect::<Vec<_>>();
        let h_poly = Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, h_evals));
        let (h_comm, h_advice) = PCS::d_commit(prover_param, &h_poly).unwrap();

        end_timer!(start);

        Ok((h_poly, h_comm, h_advice, beta, gamma))
    }

    // Proves the Rational Sumcheck relation as a batched statement of multiple
    // independent instances f, g, g_inv
    fn prove(
        to_prove: (
            Arc<DenseMultilinearExtension<E::ScalarField>>,
            PCS::Commitment,
            PCS::ProverCommitmentAdvice,
            E::ScalarField,
            E::ScalarField,
        ),
        witness: &[Self::MultilinearExtension],
        perms: &[Self::MultilinearExtension],
        selectors: &[Self::MultilinearExtension],
        gate: &[(Option<usize>, Vec<usize>)],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Self::CombinedCheckProof,
            PCS::ProverCommitmentAdvice,
            Arc<DenseMultilinearExtension<E::ScalarField>>,
            Vec<E::ScalarField>,
        ),
        PolyIOPErrors,
    > {
        let start = start_timer!(|| "combined_check prove");

        let num_vars = to_prove.0.num_vars;
        let evals_len = witness[0].evaluations.len();

        let r = transcript.get_and_append_challenge_vectors(b"0check r", num_vars)?;
        let coeff = transcript.get_and_append_challenge(b"coeff")?;

        let num_witnesses = witness.len();
        let (h_poly, h_comm, h_advice, beta, gamma) = to_prove;
        let (mut polys, h_poly_clone) = rayon::join(
            || {
                witness
                    .par_iter()
                    .map(|poly| Arc::new(DenseMultilinearExtension::clone(poly)))
                    .chain(perms.par_iter().map(|poly| {
                        Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                            num_vars,
                            poly.iter().map(|x| *x * beta + gamma).collect(),
                        ))
                    }))
                    .chain(
                        selectors
                            .par_iter()
                            .map(|poly| Arc::new(DenseMultilinearExtension::clone(poly))),
                    )
                    .collect::<Vec<_>>()
            },
            || Arc::new(DenseMultilinearExtension::clone(&h_poly)),
        );
        polys.push(h_poly_clone);
        let max_gate_degree = gate
            .iter()
            .map(|(selector, witnesses)| {
                if *selector == None {
                    witnesses.len()
                } else {
                    witnesses.len() + 1
                }
            })
            .max()
            .unwrap();

        let info = CombinedCheckInfo {
            coeff,
            num_witnesses,
            num_selectors: selectors.len(),
            gate,
            sid_offset: beta * E::ScalarField::from_u64(evals_len as u64).unwrap(),
        };

        let degree_zerocheck = max_gate_degree;
        let degree_permcheck = 2 * num_witnesses + 1;
        let extrapolation_aux = {
            let degree = std::cmp::min(degree_zerocheck, degree_permcheck);
            let points = (0..1 + degree as u64)
                .map(E::ScalarField::from)
                .collect::<Vec<_>>();
            let weights = barycentric_weights(&points);
            (points, weights)
        };
        let (proof, point, _) = combined_sumcheck_prove(
            &E::ScalarField::zero(),
            &E::ScalarField::zero(),
            num_vars,
            beta,
            gamma,
            &mut polys,
            &r,
            |sid, evals| combined_check_combine_zerocheck(sid, evals, &info),
            degree_zerocheck,
            |sid, evals| combined_check_combine_permcheck(sid, evals, &info),
            degree_permcheck,
            &extrapolation_aux,
            coeff,
            transcript,
        );

        end_timer!(start);

        Ok((
            CombinedCheckProof {
                proof,
                h_comm,
                num_rounds: num_vars,
                degree_bound: std::cmp::max(degree_zerocheck, degree_permcheck),
            },
            h_advice,
            h_poly,
            point,
        ))
    }

    fn d_prove(
        to_prove: (
            Arc<DenseMultilinearExtension<E::ScalarField>>,
            Option<PCS::Commitment>,
            PCS::ProverCommitmentAdvice,
            E::ScalarField,
            E::ScalarField,
        ),
        witness: &[Self::MultilinearExtension],
        perms: &[Self::MultilinearExtension],
        selectors: &[Self::MultilinearExtension],
        gate: &[(Option<usize>, Vec<usize>)],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Option<(Self::CombinedCheckProof, Vec<E::ScalarField>)>,
            PCS::ProverCommitmentAdvice,
            Arc<DenseMultilinearExtension<E::ScalarField>>,
        ),
        PolyIOPErrors,
    > {
        let start = start_timer!(|| "combined_check prove");

        let length = to_prove.0.num_vars;
        let num_party_vars = Net::n_parties().log_2();
        let evals_len = witness[0].evaluations.len();

        let (r, coeff) = if Net::am_master() {
            let r = transcript
                .get_and_append_challenge_vectors(b"0check r", length + num_party_vars)?;
            let coeff = transcript.get_and_append_challenge(b"coeff")?;
            Net::recv_from_master_uniform(Some((r, coeff)))
        } else {
            Net::recv_from_master_uniform(None)
        };

        let num_witnesses = witness.len();
        let (h_poly, h_comm, h_advice, beta, gamma) = to_prove;
        let (mut polys, h_poly_clone) = rayon::join(
            || {
                witness
                    .par_iter()
                    .map(|poly| Arc::new(DenseMultilinearExtension::clone(poly)))
                    .chain(perms.par_iter().map(|poly| {
                        Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                            length,
                            poly.iter().map(|x| *x * beta + gamma).collect(),
                        ))
                    }))
                    .chain(
                        selectors
                            .par_iter()
                            .map(|poly| Arc::new(DenseMultilinearExtension::clone(poly))),
                    )
                    .collect::<Vec<_>>()
            },
            || Arc::new(DenseMultilinearExtension::clone(&h_poly)),
        );
        polys.push(h_poly_clone);
        let max_gate_degree = gate
            .iter()
            .map(|(selector, witnesses)| {
                if *selector == None {
                    witnesses.len()
                } else {
                    witnesses.len() + 1
                }
            })
            .max()
            .unwrap();

        let info = CombinedCheckInfo {
            coeff,
            num_witnesses,
            num_selectors: selectors.len(),
            gate,
            sid_offset: beta
                * E::ScalarField::from_u64((evals_len * Net::n_parties()) as u64).unwrap(),
        };

        let degree_zerocheck = max_gate_degree;
        let degree_permcheck = 2 * num_witnesses + 1;
        let extrapolation_aux = {
            let degree = std::cmp::min(degree_zerocheck, degree_permcheck);
            let points = (0..1 + degree as u64)
                .map(E::ScalarField::from)
                .collect::<Vec<_>>();
            let weights = barycentric_weights(&points);
            (points, weights)
        };
        let result = d_combined_sumcheck_prove(
            &E::ScalarField::zero(),
            &E::ScalarField::zero(),
            length,
            &mut polys,
            &r,
            beta,
            gamma,
            |sid, evals| combined_check_combine_zerocheck(sid, evals, &info),
            degree_zerocheck,
            |sid, evals| combined_check_combine_permcheck(sid, evals, &info),
            degree_permcheck,
            &extrapolation_aux,
            coeff,
            transcript,
        );

        end_timer!(start);

        if Net::am_master() {
            let (proof, point, _) = result.unwrap();
            Ok((
                Some((
                    CombinedCheckProof {
                        proof,
                        h_comm: h_comm.unwrap(),
                        num_rounds: length + num_party_vars,
                        degree_bound: std::cmp::max(degree_zerocheck, degree_permcheck),
                    },
                    point,
                )),
                h_advice,
                h_poly,
            ))
        } else {
            Ok((None, h_advice, h_poly))
        }
    }

    fn verify(
        proof: &Self::CombinedCheckProof,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::CombinedCheckSubClaim, PolyIOPErrors> {
        let start = start_timer!(|| "combined_check verify");

        let beta = transcript.get_and_append_challenge(b"beta")?;
        let gamma = transcript.get_and_append_challenge(b"gamma")?;
        let zerocheck_r =
            transcript.get_and_append_challenge_vectors(b"0check r", proof.num_rounds)?;
        let coeff = transcript.get_and_append_challenge(b"coeff")?;

        let (zerocheck_expected_evaluation, h_expected_evaluation, point) =
            combined_sumcheck_verify(
                &proof.proof,
                proof.num_rounds,
                proof.degree_bound,
                &zerocheck_r,
                transcript,
            )?;

        end_timer!(start);

        Ok(CombinedCheckSubClaim {
            zerocheck_expected_evaluation,
            h_expected_evaluation,
            point,
            zerocheck_r,
            coeff,
            beta,
            gamma,
        })
    }

    fn check_openings(
        subclaim: &Self::CombinedCheckSubClaim,
        witness_openings: &[E::ScalarField],
        perm_openings: &[E::ScalarField],
        selector_openings: &[E::ScalarField],
        h_opening: &E::ScalarField,
        gate: &[(Option<usize>, Vec<usize>)],
    ) -> Result<(), PolyIOPErrors> {
        if *h_opening != subclaim.h_expected_evaluation {
            return Err(PolyIOPErrors::InvalidVerifier(
                "wrong subclaim on h".to_string(),
            ));
        }

        let nv = subclaim.point.len();
        let num_constraints = nv.pow2();
        let info = CombinedCheckInfo {
            gate,
            num_witnesses: witness_openings.len(),
            num_selectors: selector_openings.len(),
            coeff: subclaim.coeff,
            sid_offset: subclaim.beta * E::ScalarField::from_u64(num_constraints as u64).unwrap(),
        };
        let mut evals = witness_openings
            .iter()
            .map(|x| *x)
            .chain(
                perm_openings
                    .iter()
                    .map(|perm| *perm * subclaim.beta + subclaim.gamma),
            )
            .chain(selector_openings.iter().map(|x| *x))
            .collect::<Vec<_>>();
        evals.push(*h_opening);
        let mut sid_eval = subclaim.gamma;
        let mut multiplier = subclaim.beta;
        for r in &subclaim.point {
            sid_eval += *r * multiplier;
            multiplier.double_in_place();
        }

        if combined_check_combine_zerocheck(sid_eval, &evals, &info)
            + info.coeff * combined_check_combine_permcheck(sid_eval, &evals, &info)
            != subclaim.zerocheck_expected_evaluation
        {
            return Err(PolyIOPErrors::InvalidVerifier(
                "wrong subclaim on zerocheck".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::CombinedCheck;
    use crate::{
        poly_iop::{errors::PolyIOPErrors, PolyIOP},
        MultilinearKzgPCS, PolynomialCommitmentScheme,
    };
    use arithmetic::{evaluate_opt, math::Math};
    use ark_bn254::{Bn254, Fr};
    use ark_ec::pairing::Pairing;
    use ark_ff::{One, PrimeField, UniformRand, Zero};
    use ark_poly::DenseMultilinearExtension;
    use ark_std::test_rng;
    use rand_core::RngCore;
    use std::sync::Arc;

    fn generate_polys<R: RngCore>(
        num_witnesses: usize,
        num_selectors: usize,
        nv: usize,
        gate: &[(Option<usize>, Vec<usize>)],
        rng: &mut R,
    ) -> (
        Vec<Arc<DenseMultilinearExtension<Fr>>>,
        Vec<Arc<DenseMultilinearExtension<Fr>>>,
        Vec<Arc<DenseMultilinearExtension<Fr>>>,
    ) {
        let num_constraints = nv.pow2();
        let mut selectors: Vec<Vec<Fr>> = vec![vec![]; num_selectors];
        let mut witnesses: Vec<Vec<Fr>> = vec![vec![]; num_witnesses];

        for cs in 0..num_constraints {
            let mut cur_selectors: Vec<Fr> =
                (0..(num_selectors - 1)).map(|_| Fr::rand(rng)).collect();
            let cur_witness: Vec<Fr> = if cs < num_constraints / 4 {
                (0..num_witnesses).map(|_| Fr::rand(rng)).collect()
            } else {
                let row = cs % (num_constraints / 4);
                (0..num_witnesses).map(|i| witnesses[i][row]).collect()
            };
            let mut last_selector = Fr::zero();
            for (index, (q, wit)) in gate.iter().enumerate() {
                if index != num_selectors - 1 {
                    let mut cur_monomial = Fr::one();
                    cur_monomial = match q {
                        Some(p) => cur_monomial * cur_selectors[*p],
                        None => cur_monomial,
                    };
                    for wit_index in wit.iter() {
                        cur_monomial *= cur_witness[*wit_index];
                    }
                    last_selector += cur_monomial;
                } else {
                    let mut cur_monomial = Fr::one();
                    for wit_index in wit.iter() {
                        cur_monomial *= cur_witness[*wit_index];
                    }
                    last_selector /= -cur_monomial;
                }
            }
            cur_selectors.push(last_selector);
            for i in 0..num_selectors {
                selectors[i].push(cur_selectors[i]);
            }
            for i in 0..num_witnesses {
                witnesses[i].push(cur_witness[i]);
            }
        }

        let permutation = (0..num_witnesses)
            .map(|witness_idx| {
                let portion_len = num_constraints / 4;
                (0..portion_len)
                    .map(|i| {
                        Fr::from_u64((witness_idx * num_constraints + i + portion_len) as u64)
                            .unwrap()
                    })
                    .chain((0..portion_len).map(|i| {
                        Fr::from_u64((witness_idx * num_constraints + i + 3 * portion_len) as u64)
                            .unwrap()
                    }))
                    .chain(
                        (0..portion_len).map(|i| {
                            Fr::from_u64((witness_idx * num_constraints + i) as u64).unwrap()
                        }),
                    )
                    .chain((0..portion_len).map(|i| {
                        Fr::from_u64((witness_idx * num_constraints + i + 2 * portion_len) as u64)
                            .unwrap()
                    }))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        (
            witnesses
                .into_iter()
                .map(|vec| Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, vec)))
                .collect(),
            permutation
                .into_iter()
                .map(|vec| Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, vec)))
                .collect(),
            selectors
                .into_iter()
                .map(|vec| Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, vec)))
                .collect(),
        )
    }

    fn test_combined_check_helper<
        E: Pairing,
        PCS: PolynomialCommitmentScheme<E, Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>>,
    >(
        witnesses: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
        perms: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
        selectors: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
        gate: &[(Option<usize>, Vec<usize>)],
        pcs_param: &PCS::ProverParam,
    ) -> Result<(), PolyIOPErrors> {
        // prover
        let mut transcript = <PolyIOP<E::ScalarField> as CombinedCheck<E, PCS>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;

        let to_prove = <PolyIOP<E::ScalarField> as CombinedCheck<E, PCS>>::prove_prepare(
            pcs_param,
            &witnesses,
            perms,
            &mut transcript,
        )?;

        let (proof, _, h_poly, _) = <PolyIOP<E::ScalarField> as CombinedCheck<E, PCS>>::prove(
            to_prove,
            witnesses,
            perms,
            selectors,
            gate,
            &mut transcript,
        )?;

        // verifier
        let mut transcript = <PolyIOP<E::ScalarField> as CombinedCheck<E, PCS>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;
        let subclaim =
            <PolyIOP<E::ScalarField> as CombinedCheck<E, PCS>>::verify(&proof, &mut transcript)?;

        let witness_openings = witnesses
            .iter()
            .map(|f| evaluate_opt(f, &subclaim.point))
            .collect::<Vec<_>>();
        let perm_openings = perms
            .iter()
            .map(|f| evaluate_opt(f, &subclaim.point))
            .collect::<Vec<_>>();
        let selector_openings = selectors
            .iter()
            .map(|f| evaluate_opt(f, &subclaim.point))
            .collect::<Vec<_>>();
        let h_opening = evaluate_opt(&h_poly, &subclaim.point);

        <PolyIOP<E::ScalarField> as CombinedCheck<E, PCS>>::check_openings(
            &subclaim,
            &witness_openings,
            &perm_openings,
            &selector_openings,
            &h_opening,
            gate,
        )
    }

    fn test_combined_check(nv: usize) -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();

        let srs = MultilinearKzgPCS::<Bn254>::gen_srs_for_testing(&mut rng, nv)?;
        let (pcs_param, _) = MultilinearKzgPCS::<Bn254>::trim(&srs, None, Some(nv))?;

        let gate = vec![
            (Some(0), vec![0]),
            (Some(1), vec![1]),
            (Some(2), vec![2]),
            (Some(3), vec![3]),
            (Some(4), vec![0, 1]),
            (Some(5), vec![2, 3]),
            (Some(6), vec![0, 0, 0, 0, 0]),
            (Some(7), vec![1, 1, 1, 1, 1]),
            (Some(8), vec![2, 2, 2, 2, 2]),
            (Some(9), vec![3, 3, 3, 3, 3]),
            (Some(10), vec![0, 1, 2, 3]),
            (Some(11), vec![4]),
            (Some(12), vec![]),
        ];
        let (witnesses, perms, selectors) = generate_polys(5, 13, nv, &gate, &mut rng);
        test_combined_check_helper::<Bn254, MultilinearKzgPCS<Bn254>>(
            &witnesses, &perms, &selectors, &gate, &pcs_param,
        )
    }

    #[test]
    fn test_normal_polynomial() -> Result<(), PolyIOPErrors> {
        test_combined_check(5)
    }
}

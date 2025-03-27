//! Sumcheck based batch opening and verify commitment.
// TODO: this is basically a complete copy

use crate::{
    pcs::{
        multilinear_kzg::util::eq_eval,
        prelude::{Commitment, PCSError},
        PolynomialCommitmentScheme,
    },
    poly_iop::{prelude::SumCheck, PolyIOP},
    BatchProof, IOPProof,
};
use arithmetic::{
    bit_decompose, build_eq_x_r_vec, build_eq_x_r_with_coeff, math::Math, DenseMultilinearExtension, VPAuxInfo,
    VirtualPolynomial,
};
use ark_ec::{pairing::Pairing, scalar_mul::variable_base::VariableBaseMSM, CurveGroup};
use ark_std::{end_timer, log2, start_timer, One, Zero};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use std::{collections::BTreeMap, iter, marker::PhantomData, ops::Deref, sync::Arc};
use transcript::IOPTranscript;

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

/// Steps:
/// 1. get challenge point t from transcript
/// 2. build eq(t,i) for i in [0..k]
/// 3. build \tilde g_i(b) = eq(t, i) * f_i(b)
/// 4. compute \tilde eq_i(b) = eq(b, point_i)
/// 5. run sumcheck on \sum_i=1..k \tilde eq_i * \tilde g_i
/// 6. build g'(X) = \sum_i=1..k \tilde eq_i(a2) * \tilde g_i(X) where (a2) is
///    the sumcheck's point
/// 7. open g'(X) at point (a2)
pub(crate) fn d_multi_open_internal<E, PCS>(
    prover_param: &PCS::ProverParam,
    polynomials: Vec<PCS::Polynomial>,
    points: &[PCS::Point],
    evals: &[PCS::Evaluation],
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<Option<BatchProof<E, PCS>>, PCSError>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        Point = Vec<E::ScalarField>,
        Evaluation = E::ScalarField,
        ProverCommitmentAdvice = (),
    >,
{
    let open_timer = start_timer!(|| format!("multi open {} points", points.len()));

    // TODO: sanity checks
    let num_var = polynomials[0].num_vars;
    let k = polynomials.len();
    let ell = log2(k) as usize;

    // challenge point t
    let t = if Net::am_master() {
        let t = transcript.get_and_append_challenge_vectors("t".as_ref(), ell)?;
        Net::recv_from_master_uniform(Some(t))
    } else {
        Net::recv_from_master_uniform(None)
    };

    // eq(t, i) for i in [0..k]
    let eq_t_i_list = build_eq_x_r_vec(t.as_ref())?;

    // \tilde g_i(b) = eq(t, i) * f_i(b)
    let timer = start_timer!(|| format!("compute tilde g for {} points", points.len()));
    // combine the polynomials that have same opening point first to reduce the
    // cost of sum check later.
    let point_indices = points
        .iter()
        .fold(BTreeMap::<_, _>::new(), |mut indices, point| {
            let idx = indices.len();
            indices.entry(point).or_insert(idx);
            indices
        });
    let deduped_points =
        BTreeMap::from_iter(point_indices.iter().map(|(point, idx)| (*idx, *point)))
            .into_values()
            .collect::<Vec<_>>();

    let mut point_ids = vec![vec![]; point_indices.len()];
    for (i, point) in points.iter().enumerate() {
        point_ids[point_indices[point]].push(i);
    }

    let mut tilde_gs = polynomials;
    tilde_gs
        .par_iter_mut()
        .zip(&eq_t_i_list)
        .for_each(|(poly, coeff)| {
            Arc::make_mut(poly).evaluations.iter_mut().for_each(|eval| *eval *= coeff);
        });

    let (mut merged_tilde_gs, merged_tilde_gs_copy) = point_ids
        .par_iter()
        .map(|point_ids| {
            let mut merged_tilde_g = Arc::new(DenseMultilinearExtension::zero());
            for &idx in point_ids {
                let poly = &tilde_gs[idx];
                *Arc::get_mut(&mut merged_tilde_g).unwrap() += poly.deref();
            }
            (Arc::new(DenseMultilinearExtension::clone(&merged_tilde_g)), merged_tilde_g)
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();
    end_timer!(timer);

    let timer = start_timer!(|| format!("compute tilde eq for {} points", points.len()));
    let num_party_vars = Net::n_parties().log_2();
    let index_vec: Vec<E::ScalarField> = bit_decompose(Net::party_id() as u64, num_party_vars)
        .into_iter()
        .map(|x| E::ScalarField::from(x))
        .collect();

    let tilde_eqs: Vec<_> = deduped_points
        .par_iter()
        .map(|point| {
            let coeff = eq_eval(&point[num_var..], &index_vec).unwrap();
            build_eq_x_r_with_coeff(&point[..num_var], &coeff).unwrap()
        })
        .collect();
    end_timer!(timer);

    // built the virtual polynomial for SumCheck
    let timer = start_timer!(|| format!("sum check prove of {} variables", num_var));

    let proof = {
        let step = start_timer!(|| "add mle");
        let mut sum_check_vp = VirtualPolynomial::new(num_var);
        for (merged_tilde_g, tilde_eq) in merged_tilde_gs_copy.into_iter().zip(tilde_eqs.into_iter()) {
            sum_check_vp.add_mle_list([merged_tilde_g, tilde_eq], E::ScalarField::one())?;
        }
        end_timer!(step);

        match <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::d_prove(
            sum_check_vp,
            transcript,
        ) {
            Ok(p) => p,
            Err(_e) => {
                // cannot wrap IOPError with PCSError due to cyclic dependency
                return Err(PCSError::InvalidProver(
                    "Sumcheck in batch proving Failed".to_string(),
                ));
            },
        }
    };

    end_timer!(timer);

    // a2 := sumcheck's point
    let a2 = if Net::am_master() {
        let a2 = proof.as_ref().unwrap().point.clone();
        Net::recv_from_master_uniform(Some(a2))
    } else {
        Net::recv_from_master_uniform(None)
    };

    // build g'(X) = \sum_i=1..k \tilde eq_i(a2) * \tilde g_i(X) where (a2) is the
    // sumcheck's point \tilde eq_i(a2) = eq(a2, point_i)
    let step = start_timer!(|| "evaluate at a2");
    (&mut merged_tilde_gs, &deduped_points)
        .into_par_iter()
        .for_each(|(merged_tilde_g, point)| {
            let eq_i_a2 = eq_eval(&a2, point).unwrap();
            Arc::get_mut(merged_tilde_g)
                .unwrap()
                .evaluations
                .par_iter_mut()
                .for_each(|x| *x *= eq_i_a2)
        });
    let g_prime = merged_tilde_gs.into_par_iter().reduce(
        || Arc::new(DenseMultilinearExtension::zero()),
        |mut a, b| {
            if a.is_zero() {
                return b;
            }
            if b.is_zero() {
                return a;
            }
            *Arc::get_mut(&mut a).unwrap() += &*b;
            a
        },
    );
    end_timer!(step);

    let step = start_timer!(|| "pcs open");
    let g_prime_proof = PCS::open(prover_param, &g_prime, &(), a2.to_vec().as_ref())?;
    // assert_eq!(g_prime_eval, tilde_g_eval);
    end_timer!(step);

    end_timer!(open_timer);

    if Net::am_master() {
        Ok(Some(BatchProof {
            sum_check_proof: proof.unwrap(),
            f_i_eval_at_point_i: evals.to_vec(),
            g_prime_proof,
        }))
    } else {
        Ok(None)
    }
}

/// Steps:
/// 1. get challenge point t from transcript
/// 2. build g' commitment
/// 3. ensure \sum_i eq(a2, point_i) * eq(t, <i>) * f_i_evals matches the sum
///    via SumCheck verification
/// 4. verify commitment
pub(crate) fn batch_verify_internal<E, PCS>(
    verifier_param: &PCS::VerifierParam,
    f_i_commitments: &[Commitment<E>],
    points: &[PCS::Point],
    proof: &BatchProof<E, PCS>,
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<bool, PCSError>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        Point = Vec<E::ScalarField>,
        Evaluation = E::ScalarField,
        Commitment = Commitment<E>,
    >,
{
    let open_timer = start_timer!(|| "batch verification");

    // TODO: sanity checks

    let k = f_i_commitments.len();
    let ell = log2(k) as usize;
    let num_var = proof.sum_check_proof.point.len();

    // challenge point t
    let t = transcript.get_and_append_challenge_vectors("t".as_ref(), ell)?;

    // sum check point (a2)
    let a2 = &proof.sum_check_proof.point[..num_var];

    // build g' commitment
    let step = start_timer!(|| "build homomorphic commitment");
    let eq_t_list = build_eq_x_r_vec(t.as_ref())?;

    let mut scalars = vec![];
    let mut bases = vec![];

    for (i, point) in points.iter().enumerate() {
        let eq_i_a2 = eq_eval(a2, point)?;
        scalars.push(eq_i_a2 * eq_t_list[i]);
        bases.push(f_i_commitments[i].0);
    }
    let g_prime_commit = E::G1MSM::msm_unchecked_par_auto(&bases, &scalars);
    end_timer!(step);

    // ensure \sum_i eq(t, <i>) * f_i_evals matches the sum via SumCheck
    let mut sum = E::ScalarField::zero();
    for (i, &e) in eq_t_list.iter().enumerate().take(k) {
        sum += e * proof.f_i_eval_at_point_i[i];
    }
    let aux_info = VPAuxInfo {
        max_degree: 2,
        num_variables: num_var,
        phantom: PhantomData,
    };
    let subclaim = match <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::verify(
        sum,
        &proof.sum_check_proof,
        &aux_info,
        transcript,
    ) {
        Ok(p) => p,
        Err(_e) => {
            // cannot wrap IOPError with PCSError due to cyclic dependency
            return Err(PCSError::InvalidProver(
                "Sumcheck in batch verification failed".to_string(),
            ));
        },
    };
    let tilde_g_eval = subclaim.expected_evaluation;

    // verify commitment
    let res = PCS::verify(
        verifier_param,
        &Commitment(g_prime_commit.into()),
        a2.to_vec().as_ref(),
        &tilde_g_eval,
        &proof.g_prime_proof,
    )?;

    end_timer!(open_timer);
    Ok(res)
}

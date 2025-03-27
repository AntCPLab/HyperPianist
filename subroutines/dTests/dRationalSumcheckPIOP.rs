mod common;

use arithmetic::{eq_eval, math::Math, VPAuxInfo, VirtualPolynomial};
use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::{batch_inversion, One};
use ark_poly::DenseMultilinearExtension;
use ark_std::{UniformRand, Zero};
use common::{d_evaluate_mle, test_rng};
use itertools::{izip, MultiUnzip};
use rand::RngCore;
use std::{iter::zip, marker::PhantomData, sync::Arc};
use subroutines::{PolyIOP, PolyIOPErrors, RationalSumcheckSlow};

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

fn create_polys<R: RngCore>(
    num_vars: usize,
    rng: &mut R,
) -> (
    Arc<DenseMultilinearExtension<Fr>>,
    Arc<DenseMultilinearExtension<Fr>>,
    Arc<DenseMultilinearExtension<Fr>>,
) {
    let evals_p = std::iter::repeat_with(|| Fr::rand(rng))
        .take(1 << num_vars)
        .collect();
    let p = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
        num_vars, evals_p,
    ));

    let evals_q = std::iter::repeat_with(|| {
        let mut val = Fr::zero();
        while val == Fr::zero() {
            val = Fr::rand(rng);
        }
        val
    })
    .take(1 << num_vars)
    .collect();

    let q = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
        num_vars, evals_q,
    ));

    let mut g_inv = Arc::new(DenseMultilinearExtension::clone(&q));
    batch_inversion(&mut Arc::get_mut(&mut g_inv).unwrap().evaluations);

    (p, q, g_inv)
}

fn test_rational_sumcheck() -> Result<(), PolyIOPErrors> {
    let num_vars = 5;
    let num_party_vars = Net::n_parties().log_2();

    let mut rng = test_rng();

    let (p_polys, q_polys, q_inv_polys, mut expected_sums) =
        MultiUnzip::<(Vec<_>, Vec<_>, Vec<_>, Vec<_>)>::multiunzip(
            std::iter::repeat_with(|| {
                let (p, q, q_inv) = create_polys(num_vars, &mut rng);
                let expected_sum = zip(p.evaluations.iter(), q.evaluations.iter())
                    .map(|(p, q)| p / q)
                    .sum::<Fr>();
                (p, q, q_inv, expected_sum)
            })
            .take(10),
        );

    let all_expected_sums = Net::send_to_master(&expected_sums);
    if Net::am_master() {
        let all_expected_sums = all_expected_sums.unwrap();
        expected_sums = all_expected_sums
            .iter()
            .fold(vec![Fr::zero(); 10], |acc, x| {
                zip(&acc, x).map(|(a, b)| a + b).collect()
            });
    }

    let p_virt_polys = p_polys
        .iter()
        .map(|p| VirtualPolynomial::new_from_mle(&p, Fr::one()))
        .collect::<Vec<_>>();

    let mut transcript = <PolyIOP<Fr> as RationalSumcheckSlow<Fr>>::init_transcript();
    let proof = <PolyIOP<Fr> as RationalSumcheckSlow<Fr>>::d_prove(
        p_virt_polys.iter().map(|x| x.deep_copy()).collect(),
        q_polys
            .iter()
            .map(|x| Arc::new(DenseMultilinearExtension::clone(x)))
            .collect(),
        q_inv_polys
            .iter()
            .map(|x| Arc::new(DenseMultilinearExtension::clone(x)))
            .collect(),
        expected_sums,
        &mut transcript,
    )?;

    if !Net::am_master() {
        for (p, q, q_inv) in izip!(p_polys.iter(), q_polys.iter(), q_inv_polys.iter(),) {
            d_evaluate_mle(p, None);
            d_evaluate_mle(q, None);
            d_evaluate_mle(q_inv, None);
        }
        return Ok(());
    }

    // Apparently no one knows what's this for?
    let proof = proof.unwrap();
    let aux_info = VPAuxInfo {
        max_degree: 3,
        num_variables: num_vars + num_party_vars,
        phantom: PhantomData::default(),
    };
    let mut transcript = <PolyIOP<Fr> as RationalSumcheckSlow<Fr>>::init_transcript();
    let subclaim =
        <PolyIOP<Fr> as RationalSumcheckSlow<Fr>>::verify(&proof, &aux_info, &mut transcript)?;

    assert_eq!(
        &proof.sum_check_proof.point,
        &subclaim.sum_check_sub_claim.point
    );

    // Zerocheck subclaim
    let mut sum = Fr::zero();
    for (p, q, q_inv, coeff1, coeff2) in izip!(
        p_polys.iter(),
        q_polys.iter(),
        q_inv_polys.iter(),
        subclaim.coeffs.iter(),
        subclaim.coeffs[10..].iter()
    ) {
        let p_eval = d_evaluate_mle(p, Some(&subclaim.sum_check_sub_claim.point)).unwrap();
        let q_eval = d_evaluate_mle(q, Some(&subclaim.sum_check_sub_claim.point)).unwrap();
        let q_inv_eval = d_evaluate_mle(q_inv, Some(&subclaim.sum_check_sub_claim.point)).unwrap();
        sum += *coeff1
            * (q_eval * q_inv_eval - Fr::one())
            * eq_eval(&subclaim.sum_check_sub_claim.point, &subclaim.zerocheck_r)?
            + *coeff2 * (p_eval * q_inv_eval);
    }
    assert_eq!(sum, subclaim.sum_check_sub_claim.expected_evaluation);

    Ok(())
}

fn main() {
    common::network_run(|| {
        test_rational_sumcheck().unwrap();
    });
}

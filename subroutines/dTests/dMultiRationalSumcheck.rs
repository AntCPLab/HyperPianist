mod common;

use ark_bls12_381::Fr;
use ark_ff::batch_inversion;
use ark_poly::DenseMultilinearExtension;
use ark_std::{UniformRand, Zero};
use common::{d_evaluate_mle, test_rng};
use rand::RngCore;
use std::sync::Arc;
use subroutines::{MultiRationalSumcheck, PolyIOP, PolyIOPErrors};
use arithmetic::eq_eval;

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

fn create_polys<R: RngCore>(
    num_vars: usize,
    num_polys: usize,
    fx: &[Fr],
    rng: &mut R,
) -> (
    Vec<Arc<DenseMultilinearExtension<Fr>>>,
    Arc<DenseMultilinearExtension<Fr>>,
) {
    let gx = std::iter::repeat_with(|| {
        let evals_g = std::iter::repeat_with(|| {
            let mut val = Fr::zero();
            while val == Fr::zero() {
                val = Fr::rand(rng);
            }
            val
        })
        .take(1 << num_vars)
        .collect();

        Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars, evals_g,
        ))
    })
    .take(num_polys)
    .collect::<Vec<_>>();

    let inv = gx
        .iter()
        .map(|poly| {
            let mut evals = poly.evaluations.clone();
            batch_inversion(&mut evals);
            evals
        })
        .collect::<Vec<_>>();

    let h_evals = (0..(1 << num_vars))
        .map(|i| {
            fx.iter()
                .zip(inv.iter())
                .map(|(f, inv)| *f * inv[i])
                .sum::<Fr>()
        })
        .collect();

    (
        gx,
        Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars, h_evals,
        )),
    )
}

fn test_multi_rational_sumcheck() -> Result<(), PolyIOPErrors> {
    let num_vars = 5;
    let num_polys = 7;

    let mut rng = test_rng();

    let fx = if Net::am_master() {
        let fx = std::iter::repeat_with(|| Fr::rand(&mut rng))
            .take(num_polys)
            .collect::<Vec<_>>();
        Net::recv_from_master_uniform(Some(fx.clone()));
        fx
    } else {
        Net::recv_from_master_uniform(None)
    };

    let (gx, h) = create_polys(num_vars, num_polys, &fx, &mut rng);
    let mut claimed_sum = h.evaluations.iter().sum::<Fr>();
    let all_claimed_sums = Net::send_to_master(&claimed_sum);
    if Net::am_master() {
        let all_claimed_sums = all_claimed_sums.unwrap();
        claimed_sum = all_claimed_sums.iter().sum::<Fr>();
    }

    let mut transcript = <PolyIOP<Fr> as MultiRationalSumcheck<Fr>>::init_transcript();
    let proof = <PolyIOP<Fr> as MultiRationalSumcheck<Fr>>::d_prove(
        &fx,
        gx.iter()
            .map(|poly| Arc::new(DenseMultilinearExtension::clone(&poly)))
            .collect(),
        Arc::new(DenseMultilinearExtension::clone(&h)),
        claimed_sum,
        &mut transcript,
    )?;

    if !Net::am_master() {
        for poly in &gx {
            d_evaluate_mle(poly, None);
        }
        d_evaluate_mle(&h, None);
        return Ok(());
    }

    let (proof, _) = proof.unwrap();
    let mut transcript = <PolyIOP<Fr> as MultiRationalSumcheck<Fr>>::init_transcript();
    let subclaim = <PolyIOP<Fr> as MultiRationalSumcheck<Fr>>::verify(&proof, &mut transcript)?;

    let eq = eq_eval(&subclaim.sumcheck_point, &subclaim.zerocheck_r)?;
    let g_evals = gx
        .iter()
        .map(|poly| d_evaluate_mle(poly, Some(&subclaim.sumcheck_point)).unwrap())
        .collect::<Vec<_>>();
    let h_eval = d_evaluate_mle(&h, Some(&subclaim.sumcheck_point)).unwrap();
    let mut sum = h_eval * g_evals.iter().product::<Fr>();
    for i in 0..num_polys {
        let mut product_others = fx[i];
        for j in 0..num_polys {
            if j != i {
                product_others *= g_evals[j];
            }
        }
        sum -= product_others;
    }
    assert_eq!(
        h_eval + subclaim.coeff * eq * sum,
        subclaim.sumcheck_expected_evaluation
    );
    Ok(())
}

fn main() {
    common::network_run(|| {
        test_multi_rational_sumcheck().unwrap();
    });
}

mod common;

use arithmetic::{math::Math, VirtualPolynomial};
use ark_bls12_381::Fr;
use ark_ff::Zero;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use common::{d_evaluate, d_evaluate_mle, test_rng};
use std::sync::Arc;
use subroutines::{PolyIOP, PolyIOPErrors, SumCheck, SumcheckInstanceProof};
use transcript::IOPTranscript;

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

fn test_sumcheck(
    nv: usize,
    num_multiplicands_range: (usize, usize),
    num_products: usize,
) -> Result<(), PolyIOPErrors> {
    let mut rng = test_rng();
    let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();

    let (poly, mut asserted_sum) =
        VirtualPolynomial::rand_fixed_coeff(nv, num_multiplicands_range, num_products, &mut rng)?;
    if let Some(sums) = Net::send_to_master(&asserted_sum) {
        asserted_sum = sums.iter().sum();
    }

    let proof = <PolyIOP<Fr> as SumCheck<Fr>>::d_prove(poly.deep_copy(), &mut transcript)?;

    if Net::am_master() {
        let proof = proof.unwrap();
        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let mut poly_info = poly.aux_info.clone();
        poly_info.num_variables += Net::n_parties().log_2() as usize;
        let subclaim = <PolyIOP<Fr> as SumCheck<Fr>>::verify(
            asserted_sum,
            &proof,
            &poly_info,
            &mut transcript,
        )?;
        assert_eq!(&proof.point, &subclaim.point);
        assert!(
            d_evaluate(&poly, Some(&subclaim.point)).unwrap() == subclaim.expected_evaluation,
            "wrong subclaim"
        );
    } else {
        d_evaluate(&poly, None);
    }

    Ok(())
}

fn test_trivial_polynomial() -> Result<(), PolyIOPErrors> {
    let nv = 1;
    let num_multiplicands_range = (7, 8);
    let num_products = 5;

    test_sumcheck(nv, num_multiplicands_range, num_products)
}

fn test_normal_polynomial() -> Result<(), PolyIOPErrors> {
    let nv = 12;
    let num_multiplicands_range = (6, 7);
    let num_products = 5;

    test_sumcheck(nv, num_multiplicands_range, num_products)
}

fn test_generic_sumcheck(nv: usize) -> Result<(), PolyIOPErrors> {
    const NUM_POLYS: usize = 5;

    let mut rng = test_rng();
    let mut transcript = IOPTranscript::new(b"test transcript");

    let mut polys =
        std::iter::repeat_with(|| Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)))
            .take(NUM_POLYS)
            .collect::<Vec<_>>();
    let polys_clone = polys
        .iter()
        .map(|poly| Arc::new(DenseMultilinearExtension::clone(poly)))
        .collect::<Vec<_>>();

    let combine_func = |evals: &[Fr]| evals[0] * evals[1] + evals[2] * evals[3] * evals[4];

    let claim = (0..polys[0].evaluations.len())
        .map(|eval_index| {
            let evals = polys
                .iter()
                .map(|poly| poly.evaluations[eval_index])
                .collect::<Vec<_>>();
            combine_func(&evals)
        })
        .sum::<Fr>();
    let all_claims = Net::send_to_master(&claim);

    let claim = if Net::am_master() {
        all_claims.as_ref().unwrap().iter().sum()
    } else {
        Fr::zero()
    };
    let proof_result = SumcheckInstanceProof::d_prove_arbitrary(
        &claim,
        nv,
        &mut polys,
        combine_func,
        3,
        &mut transcript,
    );

    if Net::am_master() {
        let all_claims = all_claims.unwrap();

        let (proof, r_prover, _) = proof_result.unwrap();

        let mut transcript = IOPTranscript::new(b"test transcript");

        let num_party_vars = Net::n_parties().log_2();
        let (claimed_eval, r_verifier) = proof
            .verify(
                all_claims.iter().sum(),
                nv + num_party_vars,
                3,
                &mut transcript,
            )
            .unwrap();

        assert_eq!(r_prover, r_verifier);

        let evals = polys_clone
            .iter()
            .map(|poly| d_evaluate_mle(poly, Some(&r_verifier)).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(combine_func(&evals), claimed_eval);
    } else {
        for poly in &polys_clone {
            d_evaluate_mle(poly, None);
        }
    }

    Ok(())
}

fn test_generic_trivial() -> Result<(), PolyIOPErrors> {
    test_generic_sumcheck(1)
}

fn test_generic_normal() -> Result<(), PolyIOPErrors> {
    test_generic_sumcheck(5)
}

fn main() {
    common::network_run(|| {
        // test_trivial_polynomial().unwrap();
        test_normal_polynomial().unwrap();
        test_generic_trivial().unwrap();
        test_generic_normal().unwrap();
    });
}

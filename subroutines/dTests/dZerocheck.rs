mod common;

use crate::common::d_evaluate_mle;
use arithmetic::{math::Math, VirtualPolynomial};
use ark_bls12_381::Fr;
use ark_ff::{Zero, UniformRand};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use common::test_rng;
use std::sync::Arc;
use subroutines::{PolyIOP, PolyIOPErrors, ZeroCheck, ZerocheckInstanceProof};
use transcript::IOPTranscript;

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

fn test_zerocheck(
    nv: usize,
    num_multiplicands_range: (usize, usize),
    num_products: usize,
) -> Result<(), PolyIOPErrors> {
    let mut rng = test_rng();

    // good path: zero virtual poly
    let poly = VirtualPolynomial::rand_zero_fixed_coeff(
        nv,
        num_multiplicands_range,
        num_products,
        &mut rng,
    )?;

    let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
    transcript.append_message(b"testing", b"initializing transcript for testing")?;
    let proof = <PolyIOP<Fr> as ZeroCheck<Fr>>::d_prove(poly.deep_copy(), &mut transcript)?;

    if Net::am_master() {
        let proof = proof.unwrap();
        let mut poly_info = poly.aux_info.clone();
        poly_info.num_variables += Net::n_parties().log_2();
        let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;
        let zero_subclaim =
            <PolyIOP<Fr> as ZeroCheck<Fr>>::verify(&proof, &poly_info, &mut transcript)?;
        assert_eq!(&proof.point, &zero_subclaim.point);
        assert!(
            common::d_evaluate(&poly, Some(&zero_subclaim.point)).unwrap()
                == zero_subclaim.expected_evaluation,
            "wrong subclaim"
        );
    } else {
        common::d_evaluate(&poly, None);
    }

    Ok(())
}

fn test_small_polynomial() -> Result<(), PolyIOPErrors> {
    let nv = 2;
    let num_multiplicands_range = (7, 8);
    let num_products = 5;

    test_zerocheck(nv, num_multiplicands_range, num_products)
}

fn test_normal_polynomial() -> Result<(), PolyIOPErrors> {
    let nv = 12;
    let num_multiplicands_range = (6, 7);
    let num_products = 5;

    test_zerocheck(nv, num_multiplicands_range, num_products)
}

fn test_generic_zerocheck(nv: usize) -> Result<(), PolyIOPErrors> {
    const NUM_POLYS: usize = 5;

    let mut rng = test_rng();
    let mut transcript = IOPTranscript::new(b"test transcript");

    let mut polys =
        std::iter::repeat_with(|| Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)))
            .take(NUM_POLYS - 1)
            .collect::<Vec<_>>();
    polys.push(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
        nv,
        (0..(1 << nv))
            .map(|i| -polys.iter().map(|poly| poly.evaluations[i]).product::<Fr>())
            .collect(),
    )));
    let polys_clone = polys
        .iter()
        .map(|poly| Arc::new(DenseMultilinearExtension::clone(poly)))
        .collect::<Vec<_>>();

    let combine_func = |evals: &[Fr]| evals[0] * evals[1] * evals[2] * evals[3] + evals[4];

    let num_party_vars = Net::n_parties().log_2();
    let zerocheck_r = if Net::am_master() {
        let zerocheck_r = std::iter::repeat_with(|| Fr::rand(&mut rng)).take(nv + num_party_vars).collect::<Vec<_>>();
        Net::recv_from_master_uniform(Some(zerocheck_r))
    } else {
        Net::recv_from_master_uniform(None)
    };

    let proof_result = ZerocheckInstanceProof::d_prove_arbitrary(
        &Fr::zero(),
        nv,
        &mut polys,
        &zerocheck_r,
        combine_func,
        4,
        &mut transcript,
    );

    if Net::am_master() {
        let (proof, r_prover, _) = proof_result.unwrap();

        let mut transcript = IOPTranscript::new(b"test transcript");

        let num_party_vars = Net::n_parties().log_2();
        let (claimed_eval, r_verifier) = proof
            .verify(
                Fr::zero(),
                nv + num_party_vars,
                4,
                &zerocheck_r,
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

fn main() {
    common::network_run(|| {
        test_generic_zerocheck(6).unwrap();
        test_small_polynomial().unwrap();
        test_normal_polynomial().unwrap();
    });
}

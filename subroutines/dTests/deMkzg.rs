use arithmetic::math::Math;
use ark_bls12_381::Bls12_381;
use ark_ec::pairing::Pairing;
use ark_ff::UniformRand;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use rand::Rng;
use std::{iter::zip, sync::Arc};
use subroutines::pcs::prelude::{ 
    MultilinearProverParam, MultilinearUniversalParams, MultilinearVerifierParam, DeMkzg, DeMkzgSRS, PCSError, MultilinearKzgProof, PolynomialCommitmentScheme
};
use transcript::IOPTranscript;

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

mod common;
use common::{d_evaluate_mle, test_rng};

fn test_single_helper<E: Pairing, R: Rng>(
    params: &DeMkzgSRS<E>,
    poly: &Arc<DenseMultilinearExtension<E::ScalarField>>,
    rng: &mut R,
) -> Result<(), PCSError> {
    let nv = poly.num_vars;
    assert_ne!(nv, 0);
    let num_party_vars = Net::n_parties().log_2();
    let (ck, vk) = DeMkzg::trim(params, None, Some(nv + num_party_vars))?;
    let point = if Net::am_master() {
        let point: Vec<_> = (0..(nv + num_party_vars))
            .map(|_| E::ScalarField::rand(rng))
            .collect();
        Net::recv_from_master_uniform(Some(point))
    } else {
        Net::recv_from_master_uniform(None)
    };

    let (com, advice) = DeMkzg::d_commit(&ck, poly)?;
    let proof = DeMkzg::open(&ck, poly, &advice, &point)?;
    if Net::am_master() {
        let com = com.unwrap();

        let value = d_evaluate_mle(poly, Some(&point)).unwrap();
        assert!(DeMkzg::verify(&vk, &com, &point, &value, &proof)?);

        let value = E::ScalarField::rand(rng);
        assert!(!DeMkzg::verify(&vk, &com, &point, &value, &proof)?);
    } else {
        d_evaluate_mle(poly, None);
    }

    Ok(())
}

fn test_single_commit<E: Pairing>() -> Result<(), PCSError> {
    let mut rng = test_rng();

    let params = if Net::am_master() {
        let params = DeMkzg::<E>::gen_srs_for_testing(&mut rng, 4 + Net::n_parties().log_2())?;
        let pp = match &params {
            DeMkzgSRS::Unprocessed(pp) => pp,
            _ => panic!("Unexpected processed"),
        };
        Net::recv_from_master_uniform(Some(pp.clone()));
        params
    } else {
        DeMkzgSRS::Unprocessed(Net::recv_from_master_uniform(None))
    };

    // normal polynomials
    let poly1 = Arc::new(DenseMultilinearExtension::rand(4, &mut rng));
    test_single_helper(&params, &poly1, &mut rng)?;

    Ok(())
}

fn test_multi_helper<E: Pairing, R: Rng>(
    params: &DeMkzgSRS<E>,
    polys: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
    rng: &mut R,
) -> Result<(), PCSError> {
    let nv = polys[0].num_vars;
    let num_party_vars = Net::n_parties().log_2();
    let (ck, vk) = DeMkzg::trim(params, None, Some(nv + num_party_vars))?;

    let points = if Net::am_master() {
        let mut points = Vec::new();
        for poly in polys.iter() {
            let point = (0..(poly.num_vars() + num_party_vars))
                .map(|_| E::ScalarField::rand(rng))
                .collect::<Vec<E::ScalarField>>();
            points.push(point);
        }
        Net::recv_from_master_uniform(Some(points))
    } else {
        Net::recv_from_master_uniform(None)
    };

    let mut evals = vec![];
    for (poly, point) in zip(polys, &points) {
        if Net::am_master() {
            evals.push(d_evaluate_mle(poly, Some(point)).unwrap());
        } else {
            d_evaluate_mle(poly, None);
        }
    }
    evals = if Net::am_master() {
        Net::recv_from_master_uniform(Some(evals))
    } else {
        Net::recv_from_master_uniform(None)
    };

    let (commitments, advices): (Vec<_>, Vec<_>) = polys
        .iter()
        .map(|poly| DeMkzg::d_commit(&ck, poly).unwrap())
        .unzip();

    let mut transcript = IOPTranscript::new("test transcript".as_ref());
    let proof = DeMkzg::d_multi_open(&ck, polys, &advices, &points, &evals, &mut transcript)?;

    // good path
    let mut transcript = IOPTranscript::new("test transcript".as_ref());
    if Net::am_master() {
        let commitments = commitments.iter().map(|x| x.unwrap()).collect::<Vec<_>>();
        let proof = proof.unwrap();
        assert!(DeMkzg::batch_verify(
            &vk,
            &commitments,
            &points,
            &proof,
            &mut transcript
        )?);
    }

    Ok(())
}

fn test_multi<E: Pairing>() -> Result<(), PCSError> {
    let mut rng = test_rng();

    let params = if Net::am_master() {
        let params = DeMkzg::<E>::gen_srs_for_testing(&mut rng, 6 + Net::n_parties().log_2())?;
        let pp  = match &params {
            DeMkzgSRS::Unprocessed(pp) => pp,
            _ => panic!("Unexpected processed"),
        };
        Net::recv_from_master_uniform(Some(pp.clone()));
        params
    } else {
        DeMkzgSRS::Unprocessed(Net::recv_from_master_uniform(None))
    };
    for num_poly in 4..6 {
        for nv in 4..6 {
            let polys1: Vec<_> = (0..num_poly)
                .map(|_| Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)))
                .collect();
            test_multi_helper(&params, &polys1, &mut rng)?;
        }
    }

    Ok(())
}

fn main() {
    common::network_run(|| {
        // test_single_commit::<Bls12_381>().unwrap();
        test_multi::<Bls12_381>().unwrap();
    });
}

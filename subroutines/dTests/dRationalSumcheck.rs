mod common;

use arithmetic::{math::Math, Fraction};
use ark_bls12_381::Fr;
use ark_poly::DenseMultilinearExtension;
use ark_std::{UniformRand, Zero};
use common::{d_evaluate_mle, test_rng};
use rand_core::RngCore;
use std::{iter::zip, mem::take, sync::Arc};
use subroutines::{BatchedDenseRationalSum, BatchedRationalSum, BatchedSparseRationalSum};
use transcript::IOPTranscript;

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

const LAYER_SIZE: usize = 1 << 8;
const BATCH_SIZE: usize = 4;
const C: usize = 2;

fn test_rational_sumcheck_helper<
    const IS_DENSE: bool,
    MainCircuit: BatchedRationalSum<Fr, CompanionCircuit = CompanionCircuit>,
    CompanionCircuit: BatchedRationalSum<Fr>,
>(
    mut leaves_p: Vec<Vec<Fr>>,
    mut leaves_q: Vec<Vec<Fr>>,
    batched_circuit: &mut MainCircuit,
    companion_circuit: &mut Option<CompanionCircuit>,
) {
    let mut transcript = IOPTranscript::<Fr>::new(b"test_transcript");

    let expected_claims: Vec<Fraction<Fr>> = (0..BATCH_SIZE * C)
        .map(|i| {
            let layer_p = if IS_DENSE {
                &leaves_p[i]
            } else {
                &leaves_p[i % C]
            };
            let layer_q = &leaves_q[i / C];
            zip(layer_p.iter(), layer_q.iter())
                .map(|(p, q)| Fraction { p: *p, q: *q })
                .reduce(Fraction::rational_add)
                .unwrap()
        })
        .collect();
    let all_claims = Net::send_to_master(&expected_claims);

    // I love the rust type system
    let mut claims = Vec::new();
    if Net::am_master() {
        let all_claims = all_claims.unwrap();
        let expected_claims =
            all_claims
                .iter()
                .fold(vec![Fraction::zero(); all_claims[0].len()], |acc, x| {
                    zip(&acc, x)
                        .map(|(a, b)| Fraction::rational_add(*a, *b))
                        .collect()
                });

        claims = companion_circuit.as_ref().unwrap().claims();
        assert_eq!(expected_claims, claims);
    }

    let proof_ret =
        batched_circuit.d_prove_rational_sum(companion_circuit.as_mut(), &mut transcript);

    let polys_p = leaves_p
        .iter_mut()
        .map(|leave| {
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                LAYER_SIZE.log_2(),
                take(leave),
            ))
        })
        .collect::<Vec<_>>();

    let polys_q = leaves_q
        .iter_mut()
        .map(|leave| {
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                LAYER_SIZE.log_2(),
                take(leave),
            ))
        })
        .collect::<Vec<_>>();

    if Net::am_master() {
        let (proof, r_prover) = proof_ret.unwrap();
        let mut transcript = IOPTranscript::new(b"test_transcript");
        let (final_claims, r_verifier) =
            MainCircuit::verify_rational_sum(&proof, &claims, &mut transcript);
        assert_eq!(r_prover, r_verifier);
        for i in 0..(BATCH_SIZE * C) {
            let p_idx = if IS_DENSE { i } else { i % C };
            assert_eq!(
                d_evaluate_mle(&polys_p[p_idx], Some(&r_verifier)).unwrap(),
                final_claims[i].p
            );
            assert_eq!(
                d_evaluate_mle(&polys_q[i / C], Some(&r_verifier)).unwrap(),
                final_claims[i].q
            );
        }
    } else {
        for i in 0..(BATCH_SIZE * C) {
            let p_idx = if IS_DENSE { i } else { i % C };
            d_evaluate_mle(&polys_p[p_idx], None);
            d_evaluate_mle(&polys_q[i / C], None);
        }
    }
}

fn dense_prove_verify() {
    let mut rng = test_rng();
    let leaves_p: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
        std::iter::repeat_with(|| Fr::rand(&mut rng))
            .take(LAYER_SIZE)
            .collect()
    })
    .take(BATCH_SIZE * C)
    .collect();

    let leaves_q: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
        std::iter::repeat_with(|| Fr::rand(&mut rng))
            .take(LAYER_SIZE)
            .collect()
    })
    .take(BATCH_SIZE)
    .collect();

    let (mut batched_circuit, mut companion_circuit) =
        <BatchedDenseRationalSum<Fr, C> as BatchedRationalSum<Fr>>::d_construct((
            leaves_p.clone(),
            leaves_q.clone(),
        ));
    test_rational_sumcheck_helper::<true, _, _>(
        leaves_p,
        leaves_q,
        &mut batched_circuit,
        &mut companion_circuit,
    );
}

fn sparse_prove_verify() {
    let mut rng = test_rng();

    let baseline: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
        std::iter::repeat_with(|| Fr::rand(&mut rng))
            .take(LAYER_SIZE)
            .collect()
    })
    .take(BATCH_SIZE)
    .collect();

    let mut indices_p = vec![vec![]; C];
    let mut values_p = vec![vec![]; C];

    let leaves_p: Vec<Vec<Fr>> = (0..C)
        .map(|layer_idx| {
            (0..LAYER_SIZE)
                .map(|index| {
                    if rng.next_u32() % 4 == 0 {
                        let mut p = Fr::rand(&mut rng);
                        while p == Fr::zero() {
                            p = Fr::rand(&mut rng);
                        }
                        indices_p[layer_idx].push(index);
                        values_p[layer_idx].push(p);
                        p
                    } else {
                        Fr::zero()
                    }
                })
                .collect()
        })
        .collect();

    let (mut batched_circuit, mut companion_circuit) =
        <BatchedSparseRationalSum<Fr, C> as BatchedRationalSum<Fr>>::d_construct((
            indices_p,
            values_p,
            baseline.clone(),
        ));
    test_rational_sumcheck_helper::<false, _, _>(
        leaves_p,
        baseline,
        &mut batched_circuit,
        &mut companion_circuit,
    );
}

fn main() {
    common::network_run(|| {
        dense_prove_verify();
        sparse_prove_verify();
    });
}

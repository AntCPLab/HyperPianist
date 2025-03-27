#![feature(macro_metavar_expr)]

use arithmetic::math::Math;
use ark_bls12_381::Bls12_381;
use ark_ec::pairing::Pairing;
use ark_std::{log2, One, UniformRand};
use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};
use hyperpianist::{
    jolt_lookup,
    prelude::{
        CustomizedGates, HyperPlonkErrors, HyperPlonkIndex, HyperPlonkParams, SelectorColumn,
        WitnessColumn,
    },
    HyperPlonkSNARK,
};
use rand::RngCore;
use subroutines::{
    instruction::{and::ANDInstruction, or::ORInstruction, xor::XORInstruction}, pcs::prelude::DeDory, DeDorySRS, PolyIOP, PolynomialCommitmentScheme
};

mod common;
use common::test_rng;

fn test_hyperplonk_helper<E: Pairing>(gate_func: CustomizedGates) -> Result<(), HyperPlonkErrors> {
    let mut rng = test_rng();
    let num_constraints = 4;
    let num_pub_input = 2;
    let num_witnesses = 2;

    let nv = log2(num_constraints) as usize;
    let num_party_vars = Net::n_parties().log_2();
    let pcs_srs = if Net::am_master() {
        let pcs_srs = DeDory::<E>::gen_srs_for_testing(&mut rng, nv + num_party_vars)?;
        let pp = match &pcs_srs {
            DeDorySRS::Unprocessed(pp) => pp,
            _ => panic!("Unexpected processed"),
        };
        Net::recv_from_master_uniform(Some(pp.clone()));
        pcs_srs
    } else {
        DeDorySRS::Unprocessed(Net::recv_from_master_uniform(None))
    };

    // generate index
    let params = HyperPlonkParams {
        num_constraints,
        num_lookup_constraints: vec![],
        num_pub_input,
        gate_func: gate_func.clone(),
    };

    let party_id = Net::party_id();
    let offset = num_constraints * Net::n_parties();
    let permutation = (party_id * num_constraints..(party_id + 1) * num_constraints)
        .map(|i| E::ScalarField::from(i as u64))
        .chain(
            (party_id * num_constraints..(party_id + 1) * num_constraints)
                .map(|i| E::ScalarField::from((i + offset) as u64)),
        )
        .collect();

    let q1 = SelectorColumn(vec![
        E::ScalarField::one(),
        E::ScalarField::one(),
        E::ScalarField::one(),
        E::ScalarField::one(),
    ]);
    let index = HyperPlonkIndex {
        params,
        permutation,
        selectors: vec![q1],
    };

    // generate pk and vks
    let (pk, vk) =
        <PolyIOP<E::ScalarField> as HyperPlonkSNARK<E, DeDory<E>>>::d_preprocess(&index, &pcs_srs)?;

    // w1 := [0, 1, 2, 3]
    let w1 = WitnessColumn(
        std::iter::repeat_with(|| E::ScalarField::rand(&mut rng))
            .take(num_constraints)
            .collect(),
    );
    // w2 := [0^5, 1^5, 2^5, 3^5]
    let w2 = WitnessColumn(w1.0.iter().map(|&w| -w * w * w * w * w).collect());
    // public input = w1
    let pi = w1.0[..num_pub_input].to_vec();

    // generate a proof and verify
    let proof = <PolyIOP<E::ScalarField> as HyperPlonkSNARK<E, DeDory<E>>>::d_prove(
        &pk,
        &pi,
        &[w1.clone(), w2.clone()],
        &(),
    )?;

    let all_pi = Net::send_to_master(&pi);

    if Net::am_master() {
        let vk = vk.unwrap();
        let pi = all_pi.unwrap().concat();
        let proof = proof.unwrap();
        assert!(<PolyIOP<E::ScalarField> as HyperPlonkSNARK<
            E,
            DeDory<E>,
        >>::verify(&vk, &pi, &proof)?);
    }

    Ok(())
}

fn test_hyperplonk_e2e() -> Result<(), HyperPlonkErrors> {
    // Example:
    //     q_L(X) * W_1(X)^5 - W_2(X) = 0
    // is represented as
    // vec![
    //     ( 1,    Some(id_qL),    vec![id_W1, id_W1, id_W1, id_W1, id_W1]),
    //     (-1,    None,           vec![id_W2])
    // ]
    //
    // 4 public input
    // 1 selector,
    // 2 witnesses,
    // 2 variables for MLE,
    // 4 wires,
    let gates = CustomizedGates {
        gates: vec![(1, Some(0), vec![0, 0, 0, 0, 0]), (1, None, vec![1])],
    };
    test_hyperplonk_helper::<Bls12_381>(gates)
}

const C: usize = 2;
const M: usize = 1 << 8;

jolt_lookup! { LookupPlugin, C, M ;
    XORInstruction,
    ORInstruction,
    ANDInstruction
}

fn test_hyperplonk_lookup_helper<E: Pairing>(
    gate_func: CustomizedGates,
) -> Result<(), HyperPlonkErrors> {
    let mut rng = test_rng();
    let pcs_srs = if Net::am_master() {
        let pcs_srs = DeDory::<E>::gen_srs_for_testing(&mut rng, 8)?;
        let pp = match &pcs_srs {
            DeDorySRS::Unprocessed(pp) => pp,
            _ => panic!("Unexpected processed"),
        };
        Net::recv_from_master_uniform(Some(pp.clone()));
        pcs_srs
    } else {
        DeDorySRS::Unprocessed(Net::recv_from_master_uniform(None))
    };

    let num_constraints = 4;
    let num_pub_input = 4;
    let nv = log2(num_constraints) as usize;
    let num_witnesses = 2;

    // generate index
    let params = HyperPlonkParams {
        num_constraints,
        num_lookup_constraints: vec![5, 0, 3],
        num_pub_input,
        gate_func,
    };

    let party_id = Net::party_id() as usize;
    let mut offset = num_constraints * Net::n_parties();
    let mut permutation = (party_id * num_constraints..(party_id + 1) * num_constraints)
        .map(|i| E::ScalarField::from(i as u64))
        .chain(
            (party_id * num_constraints..(party_id + 1) * num_constraints)
                .map(|i| E::ScalarField::from((i + offset) as u64)),
        )
        .collect::<Vec<_>>();
    offset += num_constraints * Net::n_parties();
    for &lookup_constraints in &params.num_lookup_constraints {
        if lookup_constraints == 0 {
            continue;
        }
        let num_vars = log2(lookup_constraints) as usize;
        let length = num_vars.pow2();
        // 3 witness columns
        for _ in 0..3 {
            permutation.extend(
                (party_id * length..(party_id + 1) * length)
                    .map(|i| E::ScalarField::from((i + offset) as u64)),
            );
            offset += length * Net::n_parties();
        }
    }

    let q1 = SelectorColumn(vec![
        E::ScalarField::one(),
        E::ScalarField::one(),
        E::ScalarField::one(),
        E::ScalarField::one(),
    ]);
    let index = HyperPlonkIndex {
        params,
        permutation,
        selectors: vec![q1],
    };

    // generate pk and vks
    let ops = (
        Some(
            std::iter::repeat_with(|| {
                XORInstruction((rng.next_u32() as u64) % 256, (rng.next_u32() as u64) % 256)
            })
            .take(5)
            .collect(),
        ),
        None,
        Some(
            std::iter::repeat_with(|| {
                ANDInstruction((rng.next_u32() as u64) % 256, (rng.next_u32() as u64) % 256)
            })
            .take(3)
            .collect(),
        ),
    );
    let (pk, vk) =
        <PolyIOP<E::ScalarField> as HyperPlonkSNARK<E, DeDory<E>, LookupPlugin>>::d_preprocess(
            &index, &pcs_srs,
        )?;

    // w1 := [0, 1, 2, 3]
    let w1 = WitnessColumn(
        std::iter::repeat_with(|| E::ScalarField::rand(&mut rng))
            .take(num_constraints)
            .collect(),
    );
    // w2 := [0^5, 1^5, 2^5, 3^5]
    let w2 = WitnessColumn(w1.0.iter().map(|&w| w * w * w * w * w).collect());
    // public input = w1
    let pi = w1.clone();

    // generate a proof and verify
    let proof = <PolyIOP<E::ScalarField> as HyperPlonkSNARK<E, DeDory<E>, LookupPlugin>>::d_prove(
        &pk,
        &pi.0,
        &[w1.clone(), w2.clone()],
        &ops,
    )?;

    let all_pi = Net::send_to_master(&pi.0);

    if Net::am_master() {
        let vk = vk.unwrap();
        let pi = all_pi.unwrap().concat();
        let proof = proof.unwrap();
        assert!(<PolyIOP<E::ScalarField> as HyperPlonkSNARK<
            E,
            DeDory<E>,
            LookupPlugin,
        >>::verify(&vk, &pi, &proof)?);
    }

    Ok(())
}

fn test_hyperplonk_lookup() -> Result<(), HyperPlonkErrors> {
    // Example:
    //     q_L(X) * W_1(X)^5 - W_2(X) = 0
    // is represented as
    // vec![
    //     ( 1,    Some(id_qL),    vec![id_W1, id_W1, id_W1, id_W1, id_W1]),
    //     (-1,    None,           vec![id_W2])
    // ]
    //
    // 4 public input
    // 1 selector,
    // 2 witnesses,
    // 2 variables for MLE,
    // 4 wires,
    let gates = CustomizedGates {
        gates: vec![(1, Some(0), vec![0, 0, 0, 0, 0]), (-1, None, vec![1])],
    };
    test_hyperplonk_lookup_helper::<Bls12_381>(gates)
}

fn main() {
    common::network_run(|| {
        test_hyperplonk_e2e().unwrap();
        // test_hyperplonk_lookup().unwrap();
    });
}

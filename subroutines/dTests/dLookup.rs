mod common;

use ark_bls12_381::Bls12_381;
use ark_ff::{UniformRand, Zero};
use transcript::IOPTranscript;

use arithmetic::{evaluate_opt, math::Math};
use ark_ec::pairing::Pairing;
use ark_poly::DenseMultilinearExtension;
use common::{d_evaluate_mle, test_rng};
use rand_core::RngCore;
use std::{iter::zip, mem::take, sync::Arc};
use subroutines::{
    instruction::xor::XORInstruction, pcs::prelude::DummyPCS, JoltInstruction, LookupCheck,
    PolyIOP, PolynomialCommitmentScheme,
};

use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

fn test_lookup<
    E: Pairing,
    Instruction: JoltInstruction + Default,
    const C: usize,
    const M: usize,
>(
    ops: &[Instruction],
    rng: &mut impl RngCore,
) {
    let mut transcript = IOPTranscript::new(b"test_transcript");
    let preprocessing =
        <PolyIOP<E::ScalarField> as LookupCheck<E, DummyPCS<E>, Instruction, C, M>>::preprocess();

    let srs = DummyPCS::<E>::gen_srs_for_testing(rng, 10).unwrap();
    let (pcs_param, _) = DummyPCS::<E>::trim(&srs, None, Some(10)).unwrap();

    let (alpha, tau) = if Net::am_master() {
        let alpha = E::ScalarField::rand(rng);
        let tau = E::ScalarField::rand(rng);
        Net::recv_from_master_uniform(Some((alpha, tau)))
    } else {
        Net::recv_from_master_uniform(None)
    };

    let witnesses = <PolyIOP<E::ScalarField> as LookupCheck<
        E,
        DummyPCS<E>,
        Instruction,
        C,
        M,
    >>::construct_witnesses(ops);
    let mut poly = <PolyIOP<E::ScalarField> as LookupCheck<
        E,
        DummyPCS<E>,
        Instruction,
        C,
        M,
    >>::construct_polys(&preprocessing, ops, &alpha);

    poly.collect_m_polys();

    let proof_ret =
        <PolyIOP<E::ScalarField> as LookupCheck<E, DummyPCS<E>, Instruction, C, M>>::d_prove(
            &preprocessing,
            &pcs_param,
            &mut poly,
            &alpha,
            &tau,
            &mut transcript,
        )
        .unwrap();

    #[cfg(feature = "rational_sumcheck_piop")]
    let (proof, _advices, r_f, r_g, r_z, r_primary_sumcheck, f_inv, g_inv) = proof_ret;

    #[cfg(not(feature = "rational_sumcheck_piop"))]
    let (proof, _advices, r_f, r_g, r_z, r_primary_sumcheck) = proof_ret;

    if Net::am_master() {
        let mut transcript = IOPTranscript::new(b"test_transcript");
        let subclaim =
            <PolyIOP<E::ScalarField> as LookupCheck<E, DummyPCS<E>, Instruction, C, M>>::verify(
                &proof.unwrap(),
                &mut transcript,
            )
            .unwrap();

        assert_eq!(subclaim.r_primary_sumcheck, r_primary_sumcheck);
        assert_eq!(subclaim.r_z, r_z);
        #[cfg(feature = "rational_sumcheck_piop")]
        {
            assert_eq!(
                subclaim
                    .logup_checking
                    .f_subclaims
                    .sum_check_sub_claim
                    .point,
                r_f
            );
            assert_eq!(
                subclaim
                    .logup_checking
                    .g_subclaims
                    .sum_check_sub_claim
                    .point,
                r_g
            );
        }
        #[cfg(not(feature = "rational_sumcheck_piop"))]
        {
            assert_eq!(subclaim.logup_checking.point_f, r_f);
            assert_eq!(subclaim.logup_checking.point_g, r_g);
        }

        let m_openings = poly
            .m
            .iter()
            .map(|poly| d_evaluate_mle(poly, Some(&r_f)).unwrap())
            .collect::<Vec<_>>();
        let dim_openings = poly
            .dim
            .iter()
            .map(|poly| d_evaluate_mle(poly, Some(&r_g)).unwrap())
            .collect::<Vec<_>>();
        let E_openings = poly
            .E_polys
            .iter()
            .map(|poly| d_evaluate_mle(poly, Some(&r_g)).unwrap())
            .chain(
                poly.E_polys
                    .iter()
                    .map(|poly| d_evaluate_mle(poly, Some(&r_z)).unwrap()),
            )
            .collect::<Vec<_>>();
        let witness_openings = witnesses
            .iter()
            .map(|poly| d_evaluate_mle(poly, Some(&r_primary_sumcheck)).unwrap())
            .collect::<Vec<_>>();
        #[cfg(feature = "rational_sumcheck_piop")]
        {
            let f_inv_openings = f_inv
                .iter()
                .map(|poly| d_evaluate_mle(poly, Some(&r_f)).unwrap())
                .collect::<Vec<_>>();
            let g_inv_openings = g_inv
                .iter()
                .map(|poly| d_evaluate_mle(poly, Some(&r_g)).unwrap())
                .collect::<Vec<_>>();

            <PolyIOP<E::ScalarField> as LookupCheck<E, DummyPCS<E>, Instruction, C, M>>::check_openings(&subclaim, &dim_openings, &E_openings, &m_openings, &witness_openings, &f_inv_openings, &g_inv_openings, &alpha, &tau).unwrap();
        }

        #[cfg(not(feature = "rational_sumcheck_piop"))]
        <PolyIOP<E::ScalarField> as LookupCheck<E, DummyPCS<E>,
        Instruction, C, M>>::check_openings(&subclaim, &dim_openings,
        &E_openings, &m_openings, &witness_openings, &alpha, &tau).unwrap();
    } else {
        for poly in &poly.m {
            d_evaluate_mle(poly, None);
        }
        for poly in &poly.dim {
            d_evaluate_mle(poly, None);
        }
        for poly in &poly.E_polys {
            d_evaluate_mle(poly, None);
        }
        for poly in &poly.E_polys {
            d_evaluate_mle(poly, None);
        }
        for poly in &witnesses {
            d_evaluate_mle(poly, None);
        }
        #[cfg(feature = "rational_sumcheck_piop")]
        {
            for poly in &f_inv {
                d_evaluate_mle(poly, None);
            }
            for poly in &g_inv {
                d_evaluate_mle(poly, None);
            }
        }
    }
}

fn generate_ops<const M: usize>(rng: &mut impl RngCore, num_ops: usize) -> Vec<XORInstruction> {
    let operand_size = (M * M) as u64;
    std::iter::repeat_with(|| {
        XORInstruction(
            (rng.next_u32() as u64) % operand_size,
            (rng.next_u32() as u64) % operand_size,
        )
    })
    .take(num_ops)
    .collect()
}

fn test_e2e() {
    const C: usize = 4;
    const M: usize = 1 << 8;
    let mut rng = test_rng();
    let ops = generate_ops::<M>(&mut rng, 16);
    test_lookup::<Bls12_381, XORInstruction, C, M>(&ops, &mut rng);
}

fn test_e2e_non_pow_2() {
    const C: usize = 4;
    const M: usize = 1 << 8;
    let mut rng = test_rng();
    let ops = generate_ops::<M>(&mut rng, 25);
    test_lookup::<Bls12_381, XORInstruction, C, M>(&ops, &mut rng);
}

fn main() {
    common::network_run(|| {
        test_e2e();
        test_e2e_non_pow_2();
    });
}

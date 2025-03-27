use ark_ff::{UniformRand, Zero};
use ark_std::test_rng;
use transcript::IOPTranscript;
use ark_bn254::Bn254;

use arithmetic::{evaluate_opt, math::Math};
use ark_ec::pairing::Pairing;
use rand_core::RngCore;
use std::{iter::zip, mem::take, sync::Arc, time::Instant, error::Error};
use subroutines::{
    instruction::xor::XORInstruction, pcs::prelude::MultilinearKzgPCS, JoltInstruction, LookupCheck, MultilinearProverParam, MultilinearVerifierParam, PolyIOP, PolynomialCommitmentScheme
};
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};

fn read_mkzg_srs() -> Result<(MultilinearProverParam<Bn254>, MultilinearVerifierParam<Bn254>), Box<dyn Error>> {
    let sub_prover_setup_filepath = format!(
        "mkzg-max{}.paras",
        MAX_NUM_VARS
    );
    let verifier_setup_filepath = format!(
        "mkzgVerifier-max{}.paras",
        MAX_NUM_VARS
    );
    let prover_setup = {
        let file = std::fs::File::open(sub_prover_setup_filepath)?;
        MultilinearProverParam::deserialize_uncompressed_unchecked(std::io::BufReader::new(file))?
    };

    let verifier_setup = {
        let file = std::fs::File::open(verifier_setup_filepath)?;
        MultilinearVerifierParam::deserialize_uncompressed_unchecked(std::io::BufReader::new(file))?
    };
    Ok((prover_setup, verifier_setup))
}

fn write_mkzg_srs(
    prover: &MultilinearProverParam<Bn254>,
    verifier: &MultilinearVerifierParam<Bn254>,
) -> Result<(), Box<dyn Error>> {
    let sub_prover_setup_filepath = format!(
        "mkzg-max{}.paras",
        MAX_NUM_VARS
    );
    let verifier_setup_filepath = format!(
        "mkzgVerifier-max{}.paras",
        MAX_NUM_VARS
    );

    let file = std::fs::File::create(sub_prover_setup_filepath)?;
    prover.serialize_uncompressed(std::io::BufWriter::new(file))?;

    let file = std::fs::File::create(verifier_setup_filepath)?;
    verifier.serialize_uncompressed(std::io::BufWriter::new(file))?;
    Ok(())
}

fn test_lookup<
    E: Pairing,
    Instruction: JoltInstruction + Default,
    const C: usize,
    const M: usize,
>(
    pcs_srs: &MultilinearProverParam<E>,
    ops: &[Instruction],
    rng: &mut impl RngCore,
) {
    let start = Instant::now();
    let repetition = 10;
    for _ in 0..repetition {
        let _ =
            <PolyIOP<E::ScalarField> as LookupCheck<E, MultilinearKzgPCS<E>, Instruction, C, M>>::preprocess(
            );
    }
    println!(
        "preprocessing: {} us",
        start.elapsed().as_micros() / repetition as u128
    );

    let preprocessing =
        <PolyIOP<E::ScalarField> as LookupCheck<E, MultilinearKzgPCS<E>, Instruction, C, M>>::preprocess();

    let alpha = E::ScalarField::rand(rng);
    let tau = E::ScalarField::rand(rng);

    let witnesses = <PolyIOP<E::ScalarField> as LookupCheck<
        E,
        MultilinearKzgPCS<E>,
        Instruction,
        C,
        M,
    >>::construct_witnesses(ops);

    let start = Instant::now();
    for _ in 0..repetition {
        let mut poly = <PolyIOP<E::ScalarField> as LookupCheck<
            E,
            MultilinearKzgPCS<E>,
            Instruction,
            C,
            M,
        >>::construct_polys(&preprocessing, ops, &alpha);
        let mut transcript = IOPTranscript::new(b"test_transcript");
        let _ = <PolyIOP<E::ScalarField> as LookupCheck<E, MultilinearKzgPCS<E>, Instruction, C, M>>::prove(
            &preprocessing,
            &pcs_srs,
            &mut poly,
            &alpha,
            &tau,
            &mut transcript,
        )
        .unwrap();
    }
    println!(
        "prove: {} us",
        start.elapsed().as_micros() / repetition as u128
    );
    let mut transcript = IOPTranscript::new(b"test_transcript");
    let mut poly = <PolyIOP<E::ScalarField> as LookupCheck<
            E,
            MultilinearKzgPCS<E>,
            Instruction,
            C,
            M,
        >>::construct_polys(&preprocessing, ops, &alpha);
    let proof_ret =
        <PolyIOP<E::ScalarField> as LookupCheck<E, MultilinearKzgPCS<E>, Instruction, C, M>>::prove(
            &preprocessing,
            &pcs_srs,
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

    let mut buf = Vec::new();
    proof.serialize_compressed(&mut buf).unwrap();
    println!("Proof size compressed: {}", buf.len());

    let mut buf = Vec::new();
    proof.serialize_uncompressed(&mut buf).unwrap();
    println!("Proof size uncompressed: {}", buf.len());

    let start = Instant::now();
    for _ in 0..(repetition * 5) {
        let mut transcript = IOPTranscript::new(b"test_transcript");
        let _ =
            <PolyIOP<E::ScalarField> as LookupCheck<E, MultilinearKzgPCS<E>, Instruction, C, M>>::verify(
                &proof,
                &mut transcript,
            )
            .unwrap();
    }
    println!(
        "verify: {} us",
        start.elapsed().as_micros() / (repetition * 5) as u128
    );

    let mut transcript = IOPTranscript::new(b"test_transcript");
    let subclaim =
        <PolyIOP<E::ScalarField> as LookupCheck<E, MultilinearKzgPCS<E>, Instruction, C, M>>::verify(
            &proof,
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
        .map(|poly| evaluate_opt(poly, &r_f))
        .collect::<Vec<_>>();
    let dim_openings = poly
        .dim
        .iter()
        .map(|poly| evaluate_opt(poly, &r_g))
        .collect::<Vec<_>>();
    let E_openings = poly
        .E_polys
        .iter()
        .map(|poly| evaluate_opt(poly, &r_g))
        .chain(poly.E_polys.iter().map(|poly| evaluate_opt(poly, &r_z)))
        .collect::<Vec<_>>();
    let witness_openings = witnesses
        .iter()
        .map(|poly| evaluate_opt(poly, &r_primary_sumcheck))
        .collect::<Vec<_>>();
    #[cfg(feature = "rational_sumcheck_piop")]
    {
        let f_inv_openings = f_inv
            .iter()
            .map(|poly| evaluate_opt(poly, &r_f))
            .collect::<Vec<_>>();
        let g_inv_openings = g_inv
            .iter()
            .map(|poly| evaluate_opt(poly, &r_g))
            .collect::<Vec<_>>();

        <PolyIOP<E::ScalarField> as LookupCheck<E, MultilinearKzgPCS<E>, Instruction, C, M>>::check_openings(&subclaim, &dim_openings, &E_openings, &m_openings, &witness_openings, &f_inv_openings, &g_inv_openings, &alpha, &tau).unwrap();
    }

    #[cfg(not(feature = "rational_sumcheck_piop"))]
    <PolyIOP<E::ScalarField> as LookupCheck<E, MultilinearKzgPCS<E>, Instruction, C, M>>::check_openings(
        &subclaim,
        &dim_openings,
        &E_openings,
        &m_openings,
        &witness_openings,
        &alpha,
        &tau,
    )
    .unwrap();
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

const MIN_NUM_VARS: usize = 20;
const MAX_NUM_VARS: usize = 24;

fn bench<E: Pairing>(pcs_srs: &MultilinearProverParam<E>) {
    const C: usize = 4;
    const M: usize = 1 << 16;
    let mut rng = test_rng();
    for nv in MIN_NUM_VARS..=MAX_NUM_VARS {
        let ops = generate_ops::<M>(&mut rng, 1 << nv);
        test_lookup::<E, XORInstruction, C, M>(pcs_srs, &ops, &mut rng);
    }
}

fn main() {
    let mkzg_pcs_srs = {
        match read_mkzg_srs() {
            Ok(p) => p,
            Err(_e) => {
                let mut srs_rng = ark_std::test_rng();
                let srs = MultilinearKzgPCS::<Bn254>::gen_srs_for_testing(&mut srs_rng, 24).unwrap();
                let (prover, verifier) = MultilinearKzgPCS::trim(&srs, None, Some(24)).unwrap();
                write_mkzg_srs(&prover, &verifier).unwrap();
                (prover, verifier)
            },
        }
    };

    bench(&mkzg_pcs_srs.0);
}

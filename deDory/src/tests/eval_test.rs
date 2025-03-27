use super::super::{
    setup::{ProverSetup, PublicParameters, VerifierSetup},
    DoryCommitment,
};
use crate::eval::{compute_evaluation_vector, generate_eval_proof, verify_eval_proof};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ff::Field;
use ark_std::UniformRand;
use transcript::IOPTranscript;
use std::{mem, time::Instant};

use ark_bls12_381::Bls12_381;

fn test_random_commitment_evaluation_proof_helper<E: Pairing>() {
    let num_vars = 12;
    let mut rng = ark_std::test_rng();

    // Setup
    let time = Instant::now();
    let public_parameters = PublicParameters::<E>::rand(num_vars, &mut rng);
    println!("Generating public parameter time: {:?}", time.elapsed());
    let time = Instant::now();
    let prover_setup = ProverSetup::from(&public_parameters);
    println!("Prover precomputation time: {:?}", time.elapsed());
    let time = Instant::now();
    let verifier_setup = VerifierSetup::from(&public_parameters);
    println!("Verifier precomputation time: {:?}", time.elapsed());

    // Generate random polynomial and evaluation points
    let time = Instant::now();
    let coeff_len = 1usize << num_vars;
    let f_coeffs = core::iter::repeat_with(|| E::ScalarField::rand(&mut rng))
        .take(coeff_len)
        .collect::<Vec<_>>();
    let b_point = core::iter::repeat_with(|| E::ScalarField::rand(&mut rng))
        .take(num_vars)
        .collect::<Vec<_>>();
    println!(
        "Generating random poly and points time: {:?}",
        time.elapsed()
    );


    // Commit
    let time = Instant::now();
    let f_comm = DoryCommitment::commit(&f_coeffs, &prover_setup);
    println!("Commiting time: {:?}", time.elapsed());

    // Compute evaluation vector and product
    let time = Instant::now();
    let mut b = vec![E::ScalarField::ZERO; coeff_len];
    compute_evaluation_vector(&mut b, &b_point);
    let product = f_coeffs.iter().zip(b.iter()).map(|(f, b)| *f * *b).sum();
    println!(
        "Computing evaluation vector and product time: {:?}",
        time.elapsed()
    );

    // Prove
    let time = Instant::now();
    let mut prover_transcript = IOPTranscript::new(b"Dory Evaluation Proof");
    let mut eval_proof = generate_eval_proof(
        &mut prover_transcript,
        &f_coeffs,
        &f_comm.T_vec_prime,
        &b_point,
        &prover_setup,
    );
    println!("Proving time: {:?}", time.elapsed());

    // Verify
    let time = Instant::now();
    let mut verifier_transcript = IOPTranscript::new(b"Dory Evaluation Proof");
    let r = verify_eval_proof(
        &mut verifier_transcript,
        &mut eval_proof,
        &f_comm.comm,
        product,
        &b_point,
        &verifier_setup,
    );
    println!("Verification time: {:?}", time.elapsed());

    assert!(r.is_ok());
}

#[test]
fn test_random_commitment_evaluation_proof() {
    test_random_commitment_evaluation_proof_helper::<Bls12_381>();
}

fn test_evaluation_proof_size_helper<E: Pairing>() {
    let num_vars = 10;
    let mut rng = ark_std::test_rng();

    let public_parameters = PublicParameters::<E>::rand(num_vars, &mut rng);
    let prover_setup = ProverSetup::from(&public_parameters);
    let verifier_setup = VerifierSetup::from(&public_parameters);

    let coeff_len = 1usize << num_vars;
    let f_coeffs = core::iter::repeat_with(|| E::ScalarField::rand(&mut rng))
        .take(coeff_len)
        .collect::<Vec<_>>();
    let b_point = core::iter::repeat_with(|| E::ScalarField::rand(&mut rng))
        .take(num_vars)
        .collect::<Vec<_>>();

    let mut prover_transcript = IOPTranscript::new(b"Dory Evaluation Proof");

    let f_comm = DoryCommitment::commit(&f_coeffs, &prover_setup);

    let mut b = vec![E::ScalarField::ZERO; coeff_len];
    compute_evaluation_vector(&mut b, &b_point);

    let mut eval_proof = generate_eval_proof(
        &mut prover_transcript,
        &f_coeffs,
        &f_comm.T_vec_prime,
        &b_point,
        &prover_setup,
    );

    let proof_size = eval_proof.get_size();
    println!("Proof size: {:?}", proof_size);

    let g1_len = eval_proof.G1_messages.len();
    let g2_len = eval_proof.G2_messages.len();
    let gt_len = eval_proof.GT_messages.len();
    let g1_size = mem::size_of_val(&eval_proof.G1_messages[0]);
    let g2_size = mem::size_of_val(&eval_proof.G2_messages[0]);
    let gt_size = mem::size_of_val(&eval_proof.GT_messages[0]);
    println!("g1_len: {:?}", g1_len);
    println!("g2_len: {:?}", g2_len);
    println!("gt_len: {:?}", gt_len);

    println!("g1_size: {:?}", g1_size);
    println!("g2_size: {:?}", g2_size);
    println!("gt_size: {:?}", gt_size);

    let g1_aff_size = mem::size_of_val(&(E::G1Affine::rand(&mut rng)));
    let g1_proj_size = mem::size_of_val(&(E::G1::rand(&mut rng)));
    let g2_aff_size = mem::size_of_val(&(E::G2Affine::rand(&mut rng)));
    let g2_proj_size = mem::size_of_val(&(E::G2::rand(&mut rng)));
    let gt_size = mem::size_of_val(&(PairingOutput::<E>::rand(&mut rng)));

    println!("g1_aff_size: {:?}", g1_aff_size);
    println!("g1_proj_size: {:?}", g1_proj_size);
    println!("g2_aff_size: {:?}", g2_aff_size);
    println!("g2_proj_size: {:?}", g2_proj_size);
    println!("gt_size: {:?}", gt_size);
}

#[test]
fn test_evaluation_proof_size() {
    test_evaluation_proof_size_helper::<Bls12_381>();
}

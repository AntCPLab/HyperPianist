use super::{compute_evaluation_vector, DoryError, DoryEvalProof};
use crate::{
    base::pairings, eval::utility::unsafe_allocate_zero_vec, setup::VerifierSetup, DeferredG1,
    DeferredG2, DeferredGT,
};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ff::Zero;
use transcript::IOPTranscript;

pub fn verify_eval_proof<E: Pairing>(
    transcript: &mut IOPTranscript<E::ScalarField>,
    eval_proof: &mut DoryEvalProof<E>,
    comm: &PairingOutput<E>,
    product: E::ScalarField,
    b_point: &Vec<E::ScalarField>,
    setup: &VerifierSetup<E>,
) -> Result<(), DoryError> {
    let f_comm = DeferredGT::from(*comm);

    let num_vars = b_point.len();
    let half_num_vars = num_vars >> 1;

    // Assume the matrix is well-formed.
    let n_mat = 1usize << half_num_vars;

    if half_num_vars > setup.max_num {
        return Err(DoryError::SmallSetup(setup.max_num, num_vars));
    }

    let mut R_vec = unsafe_allocate_zero_vec::<E::ScalarField>(n_mat);
    let mut L_vec = unsafe_allocate_zero_vec::<E::ScalarField>(n_mat);

    compute_evaluation_vector(&mut L_vec, &b_point[..half_num_vars]);
    compute_evaluation_vector(&mut R_vec, &b_point[half_num_vars..]);

    if eval_proof.GT_messages.len() < 2 || eval_proof.G1_messages.is_empty() {
        Err(DoryError::VerificationError)?;
    }

    let mut C: DeferredGT<E> = eval_proof.read_GT_message(transcript).into();
    let mut D_2: DeferredGT<E> = eval_proof.read_GT_message(transcript).into();
    let mut E_1: DeferredG1<E> = eval_proof.read_G1_message(transcript).into();

    let mut D_1 = f_comm;
    let mut E_2 = DeferredG2::<E>::from(setup.Gamma_2_fin) * product;
    let mut s1 = R_vec;
    let mut s2 = L_vec;

    let mut n = half_num_vars;

    for _ in 0..half_num_vars {
        let D_1L = eval_proof.read_GT_message(transcript);
        let D_1R = eval_proof.read_GT_message(transcript);
        let D_2L = eval_proof.read_GT_message(transcript);
        let D_2R = eval_proof.read_GT_message(transcript);
        let E_1beta = eval_proof.read_G1_message(transcript);
        let E_2beta = eval_proof.read_G2_message(transcript);
        let (beta, beta_inv) = eval_proof.get_challenge_scalar(transcript);

        let C_plus = eval_proof.read_GT_message(transcript);
        let C_minus = eval_proof.read_GT_message(transcript);
        let E_1plus = eval_proof.read_G1_message(transcript);
        let E_1minus = eval_proof.read_G1_message(transcript);
        let E_2plus = eval_proof.read_G2_message(transcript);
        let E_2minus = eval_proof.read_G2_message(transcript);
        let (alpha, alpha_inv) = eval_proof.get_challenge_scalar(transcript);

        C += D_2.clone() * beta
            + D_1.clone() * beta_inv
            + DeferredGT::from(C_plus) * alpha
            + DeferredGT::from(C_minus) * alpha_inv
            + setup.chi[n];

        D_1 = DeferredGT::from(D_1L) * alpha
            + D_1R
            + DeferredGT::from(setup.Delta_1L[n]) * beta * alpha
            + DeferredGT::from(setup.Delta_1R[n]) * beta;
        D_2 = DeferredGT::from(D_2L) * alpha_inv
            + D_2R
            + DeferredGT::from(setup.Delta_2L[n]) * beta_inv * alpha_inv
            + DeferredGT::from(setup.Delta_2R[n]) * beta_inv;

        E_1 += DeferredG1::<E>::from(E_1beta) * beta
            + DeferredG1::<E>::from(E_1plus) * alpha
            + DeferredG1::<E>::from(E_1minus) * alpha_inv;
        E_2 += DeferredG2::<E>::from(E_2beta) * beta_inv
            + DeferredG2::<E>::from(E_2plus) * alpha
            + DeferredG2::<E>::from(E_2minus) * alpha_inv;

        let half_length = 1usize << (n - 1);

        let (s_1L, s_1R) = s1.split_at(half_length);
        let (s_2L, s_2R) = s2.split_at(half_length);

        s1 = s_1L
            .iter()
            .zip(s_1R)
            .map(|(s_L, s_R)| *s_L * alpha + s_R)
            .collect::<Vec<E::ScalarField>>();
        s2 = s_2L
            .iter()
            .zip(s_2R)
            .map(|(s_L, s_R)| *s_L * alpha_inv + s_R)
            .collect::<Vec<E::ScalarField>>();

        n -= 1;
    }

    assert_eq!(n, 0);

    let (gamma, gamma_inv) = eval_proof.get_challenge_scalar(transcript);

    C += DeferredGT::from(setup.H_T) * s1[0] * s2[0]
        + DeferredGT::from(pairings::pairing::<E>(setup.H_1, E_2.compute::<E::G2MSM>())) * gamma
        + DeferredGT::from(pairings::pairing::<E>(E_1.compute::<E::G1MSM>(), setup.H_2))
            * gamma_inv;
    D_1 += pairings::pairing::<E>(setup.H_1, setup.Gamma_2_0 * s1[0] * gamma);
    D_2 += pairings::pairing::<E>(setup.Gamma_1_0 * s2[0] * gamma_inv, setup.H_2);

    let E_1 = eval_proof.read_G1_message(transcript);
    let E_2 = eval_proof.read_G2_message(transcript);
    let (d, d_inv) = eval_proof.get_challenge_scalar(transcript);
    let lhs = pairings::pairing::<E>(E_1 + setup.Gamma_1_0 * d, E_2 + setup.Gamma_2_0 * d_inv);
    let rhs: PairingOutput<E> = (C + setup.chi[0] + D_2 * d + D_1 * d_inv).compute();
    if lhs != rhs {
        Err(DoryError::VerificationError)?;
    }

    Ok(())
}

pub fn verify_batched_eval_proof<E: Pairing>(
    transcript: &mut IOPTranscript<E::ScalarField>,
    eval_proof: &mut DoryEvalProof<E>,
    comms_batch: &Vec<PairingOutput<E>>,
    batch_factors: &Vec<E::ScalarField>,
    product: E::ScalarField,
    b_point: &Vec<E::ScalarField>,
    setup: &VerifierSetup<E>,
) -> Result<(), DoryError> {
    let f_comms = DeferredGT::new(
        comms_batch.iter().map(|c| *c),
        batch_factors.iter().map(|f| *f),
    );

    let num_vars = b_point.len();
    let half_num_vars = num_vars >> 1;

    // Assume the matrix is well-formed.
    let n_mat = 1usize << half_num_vars;

    if half_num_vars > setup.max_num {
        return Err(DoryError::SmallSetup(setup.max_num, num_vars));
    }

    let mut R_vec = unsafe_allocate_zero_vec::<E::ScalarField>(n_mat);
    let mut L_vec = unsafe_allocate_zero_vec::<E::ScalarField>(n_mat);

    compute_evaluation_vector(&mut L_vec, &b_point[..half_num_vars]);
    compute_evaluation_vector(&mut R_vec, &b_point[half_num_vars..]);

    if eval_proof.GT_messages.len() < 2 || eval_proof.G1_messages.is_empty() {
        Err(DoryError::VerificationError)?;
    }

    let mut C: DeferredGT<E> = eval_proof.read_GT_message(transcript).into();
    let mut D_2: DeferredGT<E> = eval_proof.read_GT_message(transcript).into();
    let mut E_1: DeferredG1<E> = eval_proof.read_G1_message(transcript).into();

    let mut D_1 = f_comms;
    let mut E_2 = DeferredG2::<E>::from(setup.Gamma_2_fin) * product;
    let mut s1 = R_vec;
    let mut s2 = L_vec;

    let mut n = half_num_vars;

    for _ in 0..half_num_vars {
        let D_1L = eval_proof.read_GT_message(transcript);
        let D_1R = eval_proof.read_GT_message(transcript);
        let D_2L = eval_proof.read_GT_message(transcript);
        let D_2R = eval_proof.read_GT_message(transcript);
        let E_1beta = eval_proof.read_G1_message(transcript);
        let E_2beta = eval_proof.read_G2_message(transcript);
        let (beta, beta_inv) = eval_proof.get_challenge_scalar(transcript);

        let C_plus = eval_proof.read_GT_message(transcript);
        let C_minus = eval_proof.read_GT_message(transcript);
        let E_1plus = eval_proof.read_G1_message(transcript);
        let E_1minus = eval_proof.read_G1_message(transcript);
        let E_2plus = eval_proof.read_G2_message(transcript);
        let E_2minus = eval_proof.read_G2_message(transcript);
        let (alpha, alpha_inv) = eval_proof.get_challenge_scalar(transcript);

        C += D_2.clone() * beta
            + D_1.clone() * beta_inv
            + DeferredGT::from(C_plus) * alpha
            + DeferredGT::from(C_minus) * alpha_inv
            + setup.chi[n];

        D_1 = DeferredGT::from(D_1L) * alpha
            + D_1R
            + DeferredGT::from(setup.Delta_1L[n]) * beta * alpha
            + DeferredGT::from(setup.Delta_1R[n]) * beta;
        D_2 = DeferredGT::from(D_2L) * alpha_inv
            + D_2R
            + DeferredGT::from(setup.Delta_2L[n]) * beta_inv * alpha_inv
            + DeferredGT::from(setup.Delta_2R[n]) * beta_inv;

        E_1 += DeferredG1::<E>::from(E_1beta) * beta
            + DeferredG1::<E>::from(E_1plus) * alpha
            + DeferredG1::<E>::from(E_1minus) * alpha_inv;
        E_2 += DeferredG2::<E>::from(E_2beta) * beta_inv
            + DeferredG2::<E>::from(E_2plus) * alpha
            + DeferredG2::<E>::from(E_2minus) * alpha_inv;

        let half_length = 1usize << (n - 1);

        let (s_1L, s_1R) = s1.split_at(half_length);
        let (s_2L, s_2R) = s2.split_at(half_length);

        s1 = s_1L
            .iter()
            .zip(s_1R)
            .map(|(s_L, s_R)| *s_L * alpha + s_R)
            .collect::<Vec<E::ScalarField>>();
        s2 = s_2L
            .iter()
            .zip(s_2R)
            .map(|(s_L, s_R)| *s_L * alpha_inv + s_R)
            .collect::<Vec<E::ScalarField>>();

        n -= 1;
    }

    assert_eq!(n, 0);

    let (gamma, gamma_inv) = eval_proof.get_challenge_scalar(transcript);

    C += DeferredGT::from(setup.H_T) * s1[0] * s2[0]
        + DeferredGT::from(pairings::pairing::<E>(setup.H_1, E_2.compute::<E::G2MSM>())) * gamma
        + DeferredGT::from(pairings::pairing::<E>(E_1.compute::<E::G1MSM>(), setup.H_2))
            * gamma_inv;
    D_1 += pairings::pairing::<E>(setup.H_1, setup.Gamma_2_0 * s1[0] * gamma);
    D_2 += pairings::pairing::<E>(setup.Gamma_1_0 * s2[0] * gamma_inv, setup.H_2);

    let E_1 = eval_proof.read_G1_message(transcript);
    let E_2 = eval_proof.read_G2_message(transcript);
    let (d, d_inv) = eval_proof.get_challenge_scalar(transcript);
    let lhs = pairings::pairing::<E>(E_1 + setup.Gamma_1_0 * d, E_2 + setup.Gamma_2_0 * d_inv);
    let rhs: PairingOutput<E> = (C + setup.chi[0] + D_2 * d + D_1 * d_inv).compute();
    if lhs != rhs {
        Err(DoryError::VerificationError)?;
    }

    Ok(())
}

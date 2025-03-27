use super::{compute_evaluation_vector, DoryError, DoryEvalProof};
use crate::{
    base::pairings, eval::utility::unsafe_allocate_zero_vec, DeferredG1, DeferredG2, DeferredGT,
    VerifierSetup,
};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ff::{Field, Zero};
use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use transcript::IOPTranscript;

pub fn verify_de_eval_proof<E: Pairing>(
    transcript: &mut IOPTranscript<E::ScalarField>,
    eval_proof: &mut DoryEvalProof<E>,
    comm: &PairingOutput<E>,
    product: E::ScalarField,
    b_point: &Vec<E::ScalarField>,
    n: usize,
    m: usize,
    setup: &VerifierSetup<E>,
) -> Result<(), DoryError> {
    // Size of full matrix: 2^{(n+m)/2} \times 2^{(n+m)/2}, (2^{(n-m)/2} * 2^m)
    let sub_num_vars = (n - m) / 2;
    let full_num_vars = (n + m) / 2;
    let sub_mat_len = 1usize << sub_num_vars;
    let full_mat_len = 1usize << full_num_vars;
    assert_eq!(n, b_point.len());
    let M = 1usize << m;
    assert_eq!(M, Net::n_parties());

    if full_num_vars > setup.max_num {
        return Err(DoryError::SmallSetup(setup.max_num, full_num_vars));
    }

    let f_zero = E::ScalarField::ZERO;
    let f_one = E::ScalarField::ONE;

    // ======== Compute L_vec and R_vec ========

    let mut sub_R_vec = unsafe_allocate_zero_vec::<E::ScalarField>(sub_mat_len);
    let mut sub_L_vec = unsafe_allocate_zero_vec::<E::ScalarField>(sub_mat_len);
    compute_evaluation_vector(&mut sub_L_vec, &b_point[..(n - m) / 2]);
    compute_evaluation_vector(&mut sub_R_vec, &b_point[(n - m) / 2..(n - m)]);
    // All sub-provers' R_vecs are exactly the same
    // Multiply sub_L_vec accordingly to bin(sub_prover_id)

    let mut L_vec: Vec<E::ScalarField> = vec![];
    let mut R_vec: Vec<E::ScalarField> = vec![];

    let mut suffix_vec = unsafe_allocate_zero_vec::<E::ScalarField>(1usize << m);
    compute_evaluation_vector(&mut suffix_vec, &b_point[(n - m)..]);

    for i in 0..sub_mat_len {
        L_vec.extend(
            suffix_vec
                .iter()
                .map(|sfx| *sfx * sub_L_vec[i])
                .collect::<Vec<E::ScalarField>>(),
        );
        R_vec.extend(std::iter::repeat_with(|| sub_R_vec[i]).take(1usize << m));
    }

    if eval_proof.GT_messages.len() < 2 || eval_proof.G1_messages.is_empty() {
        Err(DoryError::VerificationError)?;
    }

    let f_comm = DeferredGT::from(*comm);
    let mut C: DeferredGT<E> = eval_proof.read_GT_message(transcript).into();
    let mut D_2: DeferredGT<E> = eval_proof.read_GT_message(transcript).into();
    let mut E_1: DeferredG1<E> = eval_proof.read_G1_message(transcript).into();

    let mut D_1 = f_comm;
    let mut E_2 = DeferredG2::<E>::from(setup.Gamma_2_fin) * product;
    let mut s1 = R_vec;
    let mut s2 = L_vec;

    let mut k = full_num_vars;

    for _ in 0..full_num_vars {
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
            + setup.chi[k];

        let half_len = (1usize << k) / 2;

        rayon::join(
            || {
                rayon::join(
                    || {
                        D_1 = DeferredGT::from(D_1L) * alpha
                            + D_1R
                            + DeferredGT::from(setup.Delta_1L[k]) * beta * alpha
                            + DeferredGT::from(setup.Delta_1R[k]) * beta;
                        D_2 = DeferredGT::from(D_2L) * alpha_inv
                            + D_2R
                            + DeferredGT::from(setup.Delta_2L[k]) * beta_inv * alpha_inv
                            + DeferredGT::from(setup.Delta_2R[k]) * beta_inv;
                    },
                    || {
                        let (s_1L, s_1R) = s1.split_at(half_len);
                        s1 = s_1L
                            .par_iter()
                            .zip(s_1R.par_iter())
                            .map(|(s_L, s_R)| *s_L * alpha + s_R)
                            .collect::<Vec<E::ScalarField>>();
                    },
                )
            },
            || {
                rayon::join(
                    || {
                        E_1 += DeferredG1::<E>::from(E_1beta) * beta
                            + DeferredG1::<E>::from(E_1plus) * alpha
                            + DeferredG1::<E>::from(E_1minus) * alpha_inv;
                        E_2 += DeferredG2::<E>::from(E_2beta) * beta_inv
                            + DeferredG2::<E>::from(E_2plus) * alpha
                            + DeferredG2::<E>::from(E_2minus) * alpha_inv;
                    },
                    || {
                        let (s_2L, s_2R) = s2.split_at(half_len);

                        s2 = s_2L
                            .par_iter()
                            .zip(s_2R.par_iter())
                            .map(|(s_L, s_R)| *s_L * alpha_inv + s_R)
                            .collect::<Vec<E::ScalarField>>();
                    },
                )
            },
        );

        k -= 1;
    }

    assert_eq!(k, 0);

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

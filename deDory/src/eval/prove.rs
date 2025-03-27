use super::{compute_evaluation_vector, DoryEvalProof};
use crate::{base::pairings, eval::utility::unsafe_allocate_zero_vec, ProverSetup};
use ark_ec::{pairing::Pairing, AffineRepr, VariableBaseMSM};
use ark_ff::Field;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use transcript::IOPTranscript;

pub fn generate_eval_proof<E: Pairing>(
    transcript: &mut IOPTranscript<E::ScalarField>,
    f_witness_mat: &[E::ScalarField],
    T_vec_prime: &Vec<E::G1Affine>,
    b_point: &Vec<E::ScalarField>,
    setup: &ProverSetup<E>,
) -> DoryEvalProof<E> {
    // Assume the matrix is well-formed.
    let num_vars = b_point.len();
    let half_num_vars = num_vars >> 1;
    let is_padded = if f_witness_mat.len() == 1 << num_vars {
        false
    } else if f_witness_mat.len() == 1 << (num_vars - 1) {
        true
    } else {
        panic!("Sub-witness matrix has invalid size");
    };

    let n_mat = 1 << half_num_vars;

    let mut R_vec = unsafe_allocate_zero_vec::<E::ScalarField>(n_mat);
    let mut L_vec = unsafe_allocate_zero_vec::<E::ScalarField>(n_mat);

    // This is the scenario where we need more than a single row, but we don't need
    // to pad the columns.
    rayon::join(
        || compute_evaluation_vector(&mut L_vec, &b_point[..half_num_vars]),
        || compute_evaluation_vector(&mut R_vec, &b_point[half_num_vars..]),
    );

    let len = if is_padded { n_mat / 2 } else { n_mat };
    let mut v_vec = unsafe_allocate_zero_vec::<E::ScalarField>(len);
    v_vec.par_iter_mut().enumerate().for_each(|(j, out)| {
        for i in 0..n_mat {
            *out += f_witness_mat[j * n_mat + i] * L_vec[i];
        }
    });

    let (C, (D_2, E_1)) = rayon::join(
        || {
            pairings::pairing::<E>(
                E::G1MSM::msm_unchecked(&T_vec_prime, &v_vec),
                setup.Gamma_2_fin,
            )
        },
        || {
            rayon::join(
                || {
                    pairings::pairing::<E>(
                        E::G1MSM::msm_unchecked(&setup.Gamma_1[half_num_vars], &v_vec),
                        setup.Gamma_2_fin,
                    )
                },
                || E::G1MSM::msm_unchecked(&T_vec_prime, &L_vec),
            )
        },
    );

    let mut eval_proof = DoryEvalProof::new();

    eval_proof.write_GT_message(transcript, C);
    eval_proof.write_GT_message(transcript, D_2);
    eval_proof.write_G1_message(transcript, E_1);

    let mut v2 = v_vec
        .iter()
        .map(|v| (setup.Gamma_2_fin * v).into())
        .collect::<Vec<E::G2Affine>>();

    let mut s1 = R_vec;
    let mut s2 = L_vec;
    let mut v1 = T_vec_prime.clone();

    let mut n = half_num_vars;

    for _ in 0..half_num_vars {
        // length = 2^n, half_length = 2 ^{n-1}
        let len = 1 << n;
        let half_length = 1usize << (n - 1);
        let (v_1L, v_1R) = v1.split_at(half_length);
        let (v_2L, v_2R) = v2.split_at(half_length);
        let ((D_1L, D_1R, D_2L, D_2R), (E_1beta, E_2beta)): (_, (E::G1Affine, E::G2Affine)) =
            rayon::join(
                || {
                    pairings::multi_pairing_4(
                        (v_1L, &setup.Gamma_2[n - 1]), // Gamma_2[n - 1] stroes Gamma_2[..2^{n-1}]
                        (v_1R, &setup.Gamma_2[n - 1][..v_1R.len()]),
                        (&setup.Gamma_1[n - 1], v_2L),
                        (&setup.Gamma_1[n - 1][..v_2R.len()], v_2R),
                    )
                },
                || {
                    rayon::join(
                        || E::G1MSM::msm_unchecked(&setup.Gamma_1[n], &s2).into(),
                        || E::G2MSM::msm_unchecked(&setup.Gamma_2[n], &s1).into(),
                    )
                },
            );

        eval_proof.write_GT_message(transcript, D_1L);
        eval_proof.write_GT_message(transcript, D_1R);
        eval_proof.write_GT_message(transcript, D_2L);
        eval_proof.write_GT_message(transcript, D_2R);
        eval_proof.write_G1_message(transcript, E_1beta);
        eval_proof.write_G2_message(transcript, E_2beta);

        let (beta, beta_inv) = eval_proof.get_challenge_scalar(transcript);

        /// * v_1 <- v_1 + beta * Gamma_1
        /// * v_2 <- v_2 + beta_inv * Gamma_2
        rayon::join(
            || {
                v1.resize(len, E::G1Affine::zero());
                v1.par_iter_mut()
                    .zip(&setup.Gamma_1[n])
                    .for_each(|(v, &g)| *v = (*v + g * beta).into());
            },
            || {
                v2.resize(len, E::G2Affine::zero());
                v2.par_iter_mut()
                    .zip(&setup.Gamma_2[n])
                    .for_each(|(v, &g)| *v = (*v + g * beta_inv).into())
            },
        );

        let (v_1L, v_1R) = v1.split_at(half_length);
        let (v_2L, v_2R) = v2.split_at(half_length);
        let (s_1L, s_1R) = s1.split_at(half_length);
        let (s_2L, s_2R) = s2.split_at(half_length);

        let ((C_plus, C_minus), E_1plus, E_1minus, E_2plus, E_2minus) = par_join_5!(
            || pairings::multi_pairing_2((v_1L, v_2R), (v_1R, v_2L)),
            || E::G1MSM::msm_unchecked(v_1L, s_2R).into(),
            || E::G1MSM::msm_unchecked(v_1R, s_2L).into(),
            || E::G2MSM::msm_unchecked(v_2R, s_1L).into(),
            || E::G2MSM::msm_unchecked(v_2L, s_1R).into()
        );

        eval_proof.write_GT_message(transcript, C_plus);
        eval_proof.write_GT_message(transcript, C_minus);
        eval_proof.write_G1_message(transcript, E_1plus);
        eval_proof.write_G1_message(transcript, E_1minus);
        eval_proof.write_G2_message(transcript, E_2plus);
        eval_proof.write_G2_message(transcript, E_2minus);

        let (alpha, alpha_inv) = eval_proof.get_challenge_scalar(transcript);

        rayon::join(
            || {
                rayon::join(
                    || {
                        let (v_1L, v_1R) = v1.split_at(half_length);
                        v1 = v_1L
                            .par_iter()
                            .zip(v_1R)
                            .map(|(v_L, v_R)| (*v_L * alpha + v_R).into())
                            .collect::<Vec<E::G1Affine>>()
                    },
                    || {
                        let (v_2L, v_2R) = v2.split_at(half_length);
                        v2 = v_2L
                            .par_iter()
                            .zip(v_2R)
                            .map(|(v_L, v_R)| (*v_L * alpha_inv + v_R).into())
                            .collect::<Vec<E::G2Affine>>()
                    },
                )
            },
            || {
                rayon::join(
                    || {
                        let (s_1L, s_1R) = s1.split_at(half_length);
                        s1 = s_1L
                            .iter()
                            .zip(s_1R)
                            .map(|(s_L, s_R)| *s_L * alpha + s_R)
                            .collect::<Vec<E::ScalarField>>()
                    },
                    || {
                        let (s_2L, s_2R) = s2.split_at(half_length);
                        s2 = s_2L
                            .iter()
                            .zip(s_2R)
                            .map(|(s_L, s_R)| *s_L * alpha_inv + s_R)
                            .collect::<Vec<E::ScalarField>>()
                    },
                )
            },
        );
        n -= 1;
    }

    assert_eq!(n, 0);

    let (gamma, gamma_inv) = eval_proof.get_challenge_scalar(transcript);

    // E_1 = v1[0] is a single element.
    let E_1: E::G1Affine = (v1[0] + setup.H_1 * s1[0] * gamma).into();
    // E_2 = v2[0] is a single element.
    let E_2: E::G2Affine = (v2[0] + setup.H_2 * s2[0] * gamma_inv).into();

    eval_proof.write_G1_message(transcript, E_1);
    eval_proof.write_G2_message(transcript, E_2);

    eval_proof
}

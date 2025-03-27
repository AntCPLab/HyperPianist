use super::{compute_evaluation_vector, DoryEvalProof};
use crate::{
    base::{impl_serde_for_ark_serde_checked, pairings},
    eval::utility::unsafe_allocate_zero_vec,
    SubProverSetup,
};
use ark_ec::{
    pairing::{Pairing, PairingOutput},
    AffineRepr, VariableBaseMSM,
};
use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{end_timer, start_timer, Zero};
use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};
use itertools::MultiUnzip;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use transcript::IOPTranscript;

#[derive(Default, Clone, CanonicalSerialize, CanonicalDeserialize, PartialEq, Eq, Debug)]
pub struct SentToMasterData<E: Pairing> {
    pub F_vec: Vec<E::ScalarField>,
    pub G1_vec: Vec<E::G1Affine>,
    pub G2_vec: Vec<E::G2Affine>,
    pub GT_vec: Vec<PairingOutput<E>>,
}

impl<E: Pairing> SentToMasterData<E> {
    pub(super) fn new() -> Self {
        Self {
            F_vec: Vec::new(),
            G1_vec: Vec::new(),
            G2_vec: Vec::new(),
            GT_vec: Vec::new(),
        }
    }
}

impl_serde_for_ark_serde_checked!(SentToMasterData);

// Note: You are supposed to feed the flattened array of transpose of the
// subwitness matrix here. This happens to be equal to the evaluations array of
// the MLE poly.
pub fn de_generate_eval_proof<E: Pairing>(
    sub_prover_id: usize,
    transcript: &mut IOPTranscript<E::ScalarField>,
    sub_witness_evals: &[E::ScalarField],
    sub_T_prime_vec: &Vec<E::G1Affine>,
    b_point: &Vec<E::ScalarField>,
    n: usize,
    m: usize,
    setup: &SubProverSetup<E>,
) -> Option<DoryEvalProof<E>> {
    let timer = start_timer!(|| "deDory deProve");

    // Assume the matrix is well-formed.
    // Size of sub-prover's witness matrix should be 2^{(n-m)/2} times 2^{(n-m)/2.
    let sub_mat_len: usize = 1 << ((n - m) / 2);

    let is_padded = if sub_witness_evals.len() == 1 << (n - m) {
        false
    } else if sub_witness_evals.len() == 1 << (n - m - 1) {
        true
    } else {
        panic!("Sub-witness matrix has invalid size");
    };

    assert!(sub_mat_len > 0);
    assert_eq!(n, b_point.len());
    let M = 1usize << m;
    assert_eq!(M, Net::n_parties());
    // Size of full matrix: 2^{(n+m)/2} \times 2^{(n+m)/2}, (2^{(n-m)/2} * 2^m)
    let full_mat_len = sub_mat_len << m;
    let sub_num_vars = (n - m) / 2;
    let full_num_vars = (n + m) / 2;
    // Check: 2^{(k-m)/2} * 2^m < 2^max_num
    assert!(full_mat_len <= (1usize << setup.max_num));
    // Check: 2^{(n + m)/2} = sub_mat_len * 2^m
    assert_eq!((1usize << full_num_vars), full_mat_len);

    let (SubGamma_1, SubGamma_2): (Vec<E::G1Affine>, Vec<E::G2Affine>) = (0..sub_mat_len)
        .map(|i| {
            (
                setup.Gamma_1[sub_prover_id + (i << m)].clone(),
                setup.Gamma_2[sub_prover_id + (i << m)].clone(),
            )
        })
        .unzip();

    let g1_zero = E::G1Affine::zero();
    let g2_zero = E::G2Affine::zero();
    let gt_zero = PairingOutput::zero();
    let f_zero = E::ScalarField::ZERO;
    let f_one = E::ScalarField::ONE;

    // ======== Compute sub_L_vec and sub_R_vec ========
    // It's sufficient to set the size of sub-prover's evaluation vectors to be
    // sub_mat_len
    let mut sub_R_vec = unsafe_allocate_zero_vec::<E::ScalarField>(sub_mat_len);
    let mut sub_L_vec = unsafe_allocate_zero_vec::<E::ScalarField>(sub_mat_len);
    let step = start_timer!(|| "compute evaluation vectors");
    // b_point: little endian
    rayon::join(
        || compute_evaluation_vector(&mut sub_R_vec, &b_point[(n - m) / 2..(n - m)]),
        || {
            compute_evaluation_vector(&mut sub_L_vec, &b_point[..(n - m) / 2]);
            let mut c = f_one;
            for i in 0..m {
                if sub_prover_id & (1usize << i) == 0 {
                    c *= f_one - b_point[n - m + i];
                } else {
                    c *= b_point[n - m + i];
                };
            }
            sub_L_vec.par_iter_mut().for_each(|l| *l *= c);
        },
    );
    // All sub-provers' R_vecs are exactly the same
    // Multiply sub_L_vec accordingly to bin(sub_prover_id)
    end_timer!(step);

    let step = start_timer!(|| "Compute sub_v_vec");

    // ======== Compute sub_v_vec ========
    let len = if is_padded {
        sub_mat_len / 2
    } else {
        sub_mat_len
    };
    let mut sub_v_vec = unsafe_allocate_zero_vec::<E::ScalarField>(len);
    sub_v_vec.par_iter_mut().enumerate().for_each(|(j, out)| {
        for i in 0..sub_mat_len {
            *out += sub_witness_evals[j * sub_mat_len + i] * sub_L_vec[i];
        }
    });

    end_timer!(step);

    let step = start_timer!(|| "Compute sub commitments and sub-product");
    // ======== Compute sub-commitments and sub-product ========
    let ((sub_C, sub_E_1), sub_D_2) = rayon::join(
        || {
            rayon::join(
                || {
                    pairings::pairing::<E>(
                        E::G1MSM::msm_unchecked(&sub_T_prime_vec, &sub_v_vec),
                        setup.Gamma_2_fin,
                    )
                },
                || E::G1MSM::msm_unchecked(&sub_T_prime_vec, &sub_L_vec).into(),
            )
        },
        || {
            pairings::pairing::<E>(
                E::G1MSM::msm_unchecked(&SubGamma_1, &sub_v_vec),
                setup.Gamma_2_fin,
            )
        },
    );
    end_timer!(step);

    // ======== Aggregate commitments and product ========

    let step = start_timer!(|| "Aggregate commitments and product");
    let mut eval_proof = DoryEvalProof::new();
    let am_master = Net::am_master();

    let sub_data = SentToMasterData {
        F_vec: vec![],
        G1_vec: vec![sub_E_1],
        G2_vec: vec![],
        GT_vec: vec![sub_C, sub_D_2],
    };

    let sub_data_vec = Net::send_to_master(&sub_data);
    let (C, D_2, E_1) = if am_master {
        let sub_data_vec = sub_data_vec.unwrap();
        (
            sub_data_vec
                .iter()
                .map(|sub_data_i| sub_data_i.GT_vec[0])
                .sum(),
            sub_data_vec
                .iter()
                .map(|sub_data_i| sub_data_i.GT_vec[1])
                .sum(),
            sub_data_vec
                .iter()
                .fold(E::G1MSM::zero(), |acc, sub_data_i| {
                    acc + &sub_data_i.G1_vec[0]
                })
                .into(),
        )
    } else {
        (gt_zero, gt_zero, g1_zero)
    };

    if am_master {
        eval_proof.write_GT_message(transcript, C);
        eval_proof.write_GT_message(transcript, D_2);
        eval_proof.write_G1_message(transcript, E_1);
    }

    end_timer!(step);

    // ======== Start Recursion ========
    let step = start_timer!(|| "Recursive steps (distributed)");
    let mut sub_s1 = sub_R_vec;
    let mut sub_s2 = sub_L_vec;
    let mut sub_v1 = sub_T_prime_vec.clone();
    let mut sub_v2 = sub_v_vec
        .iter()
        .map(|v| (setup.Gamma_2_fin * v).into())
        .collect::<Vec<E::G2Affine>>();

    let mut k = sub_num_vars;

    for _ in 0..sub_num_vars {
        let len = 1usize << k;
        let half_len = len / 2;

        let step = start_timer!(|| "Computing D, E");
        let ((sub_D_1L, sub_D_1R, sub_D_2L, sub_D_2R), (sub_E_1beta, sub_E_2beta)) = rayon::join(
            || {
                let (sub_v_1L, sub_v_1R) = sub_v1.split_at(half_len);
                let (sub_v_2L, sub_v_2R) = sub_v2.split_at(half_len);
                pairings::multi_pairing_4(
                    (sub_v_1L, &SubGamma_2[..half_len]),
                    (sub_v_1R, &SubGamma_2[..sub_v_1R.len()]),
                    (&SubGamma_1[..half_len], sub_v_2L),
                    (&SubGamma_1[..sub_v_2R.len()], sub_v_2R),
                )
            },
            || {
                rayon::join(
                    || E::G1MSM::msm_unchecked(&SubGamma_1[..len], &sub_s2).into(),
                    || E::G2MSM::msm_unchecked(&SubGamma_2[..len], &sub_s1).into(),
                )
            },
        );
        end_timer!(step);

        let step = start_timer!(|| "Communciation round");

        let sub_data = SentToMasterData {
            F_vec: vec![],
            G1_vec: vec![sub_E_1beta],
            G2_vec: vec![sub_E_2beta],
            GT_vec: vec![sub_D_1L, sub_D_1R, sub_D_2L, sub_D_2R],
        };

        let sub_data_vec = Net::send_to_master(&sub_data);
        let (D_1L, D_1R, D_2L, D_2R, E_1beta, E_2beta) = if am_master {
            let step = start_timer!(|| "Master accumulation");
            let sub_data_vec = sub_data_vec.unwrap();
            let ret = (
                sub_data_vec
                    .iter()
                    .map(|sub_data_i| sub_data_i.GT_vec[0])
                    .sum(),
                sub_data_vec
                    .iter()
                    .map(|sub_data_i| sub_data_i.GT_vec[1])
                    .sum(),
                sub_data_vec
                    .iter()
                    .map(|sub_data_i| sub_data_i.GT_vec[2])
                    .sum(),
                sub_data_vec
                    .iter()
                    .map(|sub_data_i| sub_data_i.GT_vec[3])
                    .sum(),
                sub_data_vec
                    .iter()
                    .fold(E::G1MSM::zero(), |acc, sub_data_i| {
                        acc + &sub_data_i.G1_vec[0]
                    })
                    .into(),
                sub_data_vec
                    .iter()
                    .fold(E::G2MSM::zero(), |acc, sub_data_i| {
                        acc + &sub_data_i.G2_vec[0]
                    })
                    .into(),
            );
            end_timer!(step);
            ret
        } else {
            (gt_zero, gt_zero, gt_zero, gt_zero, g1_zero, g2_zero)
        };

        end_timer!(step);

        let step = start_timer!(|| "Challenge betas");

        let beta_vec = Net::recv_from_master_uniform(if am_master {
            eval_proof.write_GT_message(transcript, D_1L);
            eval_proof.write_GT_message(transcript, D_1R);
            eval_proof.write_GT_message(transcript, D_2L);
            eval_proof.write_GT_message(transcript, D_2R);
            eval_proof.write_G1_message(transcript, E_1beta);
            eval_proof.write_G2_message(transcript, E_2beta);

            let betas = eval_proof.get_challenge_scalar(transcript);
            Some(betas)
        } else {
            None
        });

        rayon::join(
            || {
                sub_v1.resize(len, E::G1Affine::zero());
                sub_v1
                    .par_iter_mut()
                    .zip(&SubGamma_1[..len])
                    .for_each(|(v, g)| *v = (*v + *g * beta_vec.0).into())
            },
            || {
                sub_v2.resize(len, E::G2Affine::zero());
                sub_v2
                    .par_iter_mut()
                    .zip(&SubGamma_2[..len])
                    .for_each(|(v, g)| *v = (*v + *g * beta_vec.1).into())
            },
        );

        end_timer!(step);

        let step = start_timer!(|| "Compute s, v");

        let (sub_s_1L, sub_s_1R) = sub_s1.split_at(half_len);
        let (sub_s_2L, sub_s_2R) = sub_s2.split_at(half_len);

        let (sub_v_1L, sub_v_1R) = sub_v1.split_at(half_len);
        let (sub_v_2L, sub_v_2R) = sub_v2.split_at(half_len);

        /// * v_1 <- v_1 + beta * Gamma_1
        /// * v_2 <- v_2 + beta_inv * Gamma_2
        let ((sub_C_plus, sub_C_minus), sub_E_1plus, sub_E_1minus, sub_E_2plus, sub_E_2minus) = par_join_5!(
            || {
                let (sub_v_1L, sub_v_1R) = sub_v1.split_at(half_len);
                let (sub_v_2L, sub_v_2R) = sub_v2.split_at(half_len);
                pairings::multi_pairing_2((sub_v_1L, sub_v_2R), (sub_v_1R, sub_v_2L))
            },
            || E::G1MSM::msm_unchecked(sub_v_1L, sub_s_2R).into(),
            || E::G1MSM::msm_unchecked(sub_v_1R, sub_s_2L).into(),
            || E::G2MSM::msm_unchecked(sub_v_2R, sub_s_1L).into(),
            || E::G2MSM::msm_unchecked(sub_v_2L, sub_s_1R).into()
        );

        end_timer!(step);

        let step = start_timer!(|| "Communication round");

        let sub_data = SentToMasterData {
            F_vec: vec![],
            G1_vec: vec![sub_E_1plus, sub_E_1minus],
            G2_vec: vec![sub_E_2plus, sub_E_2minus],
            GT_vec: vec![sub_C_plus, sub_C_minus],
        };

        let sub_data_vec = Net::send_to_master(&sub_data);
        let (C_plus, C_minus, E_1plus, E_1minus, E_2plus, E_2minus) = if am_master {
            let step = start_timer!(|| "Master accumulation");
            let sub_data_vec = sub_data_vec.unwrap();
            let ret = (
                sub_data_vec
                    .iter()
                    .map(|sub_data_i| sub_data_i.GT_vec[0])
                    .sum(),
                sub_data_vec
                    .iter()
                    .map(|sub_data_i| sub_data_i.GT_vec[1])
                    .sum(),
                sub_data_vec
                    .iter()
                    .fold(E::G1MSM::zero(), |acc, sub_data_i| {
                        acc + &sub_data_i.G1_vec[0]
                    })
                    .into(),
                sub_data_vec
                    .iter()
                    .fold(E::G1MSM::zero(), |acc, sub_data_i| {
                        acc + &sub_data_i.G1_vec[1]
                    })
                    .into(),
                sub_data_vec
                    .iter()
                    .fold(E::G2MSM::zero(), |acc, sub_data_i| {
                        acc + &sub_data_i.G2_vec[0]
                    })
                    .into(),
                sub_data_vec
                    .iter()
                    .fold(E::G2MSM::zero(), |acc, sub_data_i| {
                        acc + &sub_data_i.G2_vec[1]
                    })
                    .into(),
            );
            end_timer!(step);
            ret
        } else {
            (gt_zero, gt_zero, g1_zero, g1_zero, g2_zero, g2_zero)
        };

        end_timer!(step);

        let step = start_timer!(|| "Challenge alphas");

        let alpha_vec = Net::recv_from_master_uniform(if am_master {
            eval_proof.write_GT_message(transcript, C_plus);
            eval_proof.write_GT_message(transcript, C_minus);
            eval_proof.write_G1_message(transcript, E_1plus);
            eval_proof.write_G1_message(transcript, E_1minus);
            eval_proof.write_G2_message(transcript, E_2plus);
            eval_proof.write_G2_message(transcript, E_2minus);

            let alphas = eval_proof.get_challenge_scalar(transcript);

            Some(alphas)
        } else {
            None
        });

        rayon::join(
            || {
                rayon::join(
                    || {
                        sub_v1 = sub_v1[..half_len]
                            .par_iter()
                            .zip(&sub_v1[half_len..])
                            .map(|(v_L, v_R)| (*v_L * alpha_vec.0 + v_R).into())
                            .collect::<Vec<E::G1Affine>>();
                    },
                    || {
                        sub_v2 = sub_v2[..half_len]
                            .par_iter()
                            .zip(&sub_v2[half_len..])
                            .map(|(v_L, v_R)| (*v_L * alpha_vec.1 + v_R).into())
                            .collect::<Vec<E::G2Affine>>();
                    },
                )
            },
            || {
                rayon::join(
                    || {
                        sub_s1 = sub_s1[..half_len]
                            .par_iter()
                            .zip(&sub_s1[half_len..])
                            .map(|(s_L, s_R)| *s_L * alpha_vec.0 + s_R)
                            .collect::<Vec<E::ScalarField>>()
                    },
                    || {
                        sub_s2 = sub_s2[..half_len]
                            .par_iter()
                            .zip(&sub_s2[half_len..])
                            .map(|(s_L, s_R)| *s_L * alpha_vec.1 + s_R)
                            .collect::<Vec<E::ScalarField>>()
                    },
                )
            },
        );

        end_timer!(step);

        k -= 1;
    }
    assert_eq!(k, 0);

    end_timer!(step);

    let sub_data = SentToMasterData::<E> {
        F_vec: vec![sub_s1[0], sub_s2[0]],
        G1_vec: vec![sub_v1[0]],
        G2_vec: vec![sub_v2[0]],
        GT_vec: vec![],
    };

    let sub_data_vec = Net::send_to_master(&sub_data);

    if am_master {
        let step = start_timer!(|| "Recursive steps (master)");
        let sub_data_vec = sub_data_vec.unwrap();
        let (mut v1, mut v2, mut s1, mut s2): (
            Vec<E::G1Affine>,
            Vec<E::G2Affine>,
            Vec<E::ScalarField>,
            Vec<E::ScalarField>,
        ) = sub_data_vec
            .iter()
            .map(|sub_data_i| {
                (
                    sub_data_i.G1_vec[0],
                    sub_data_i.G2_vec[0],
                    sub_data_i.F_vec[0],
                    sub_data_i.F_vec[1],
                )
            })
            .multiunzip();

        k = m;

        for _ in 0..m {
            let len = 1usize << k;
            // length = 2^k, half_len = 2^{k-1}
            let half_len = len / 2;
            let (v_1L, v_1R) = v1.split_at(half_len);
            let (v_2L, v_2R) = v2.split_at(half_len);

            let ((D_1L, D_1R, D_2L, D_2R), (E_1beta, E_2beta)) = rayon::join(
                || {
                    pairings::multi_pairing_4(
                        (v_1L, &setup.Gamma_2[..half_len]),
                        (v_1R, &setup.Gamma_2[..half_len]),
                        (&setup.Gamma_1[..half_len], v_2L),
                        (&setup.Gamma_1[..half_len], v_2R),
                    )
                },
                || {
                    rayon::join(
                        || E::G1MSM::msm_unchecked(&setup.Gamma_1[..len], &s2).into(),
                        || E::G2MSM::msm_unchecked(&setup.Gamma_2[..len], &s1).into(),
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

            rayon::join(
                || {
                    v1.par_iter_mut()
                        .zip(&setup.Gamma_1[..len])
                        .for_each(|(v, g)| *v = (*v + *g * beta).into())
                },
                || {
                    v2.par_iter_mut()
                        .zip(&setup.Gamma_2[..len])
                        .for_each(|(v, g)| *v = (*v + *g * beta_inv).into())
                },
            );

            let (v_1L, v_1R) = v1.split_at(half_len);
            let (v_2L, v_2R) = v2.split_at(half_len);

            let ((C_plus, C_minus), E_1plus, E_1minus, E_2plus, E_2minus) = par_join_5!(
                || pairings::multi_pairing_2((v_1L, v_2R), (v_1R, v_2L)),
                || E::G1MSM::msm_unchecked(v_1L, &s2[half_len..]).into(),
                || E::G1MSM::msm_unchecked(v_1R, &s2[..half_len]).into(),
                || E::G2MSM::msm_unchecked(v_2R, &s1[..half_len]).into(),
                || E::G2MSM::msm_unchecked(v_2L, &s1[half_len..]).into()
            );

            /// * v_1 <- v_1 + beta * Gamma_1
            /// * v_2 <- v_2 + beta_inv * Gamma_2
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
                            v1 = v1[..half_len]
                                .par_iter()
                                .zip(&v1[half_len..])
                                .map(|(v_L, v_R)| (*v_L * alpha + v_R).into())
                                .collect::<Vec<E::G1Affine>>()
                        },
                        || {
                            v2 = v2[..half_len]
                                .par_iter()
                                .zip(&v2[half_len..])
                                .map(|(v_L, v_R)| (*v_L * alpha_inv + v_R).into())
                                .collect::<Vec<E::G2Affine>>()
                        },
                    )
                },
                || {
                    rayon::join(
                        || {
                            s1 = s1[..half_len]
                                .par_iter()
                                .zip(&s1[half_len..])
                                .map(|(s_L, s_R)| *s_L * alpha + s_R)
                                .collect::<Vec<E::ScalarField>>()
                        },
                        || {
                            s2 = s2[..half_len]
                                .par_iter()
                                .zip(&s2[half_len..])
                                .map(|(s_L, s_R)| *s_L * alpha_inv + s_R)
                                .collect::<Vec<E::ScalarField>>()
                        },
                    )
                },
            );

            k -= 1;
        }

        assert_eq!(k, 0);
        let (gamma, gamma_inv) = eval_proof.get_challenge_scalar(transcript);

        // E_1 = v1[0] is a single element.
        let E_1: E::G1Affine = (v1[0] + setup.H_1 * s1[0] * gamma).into();
        // E_2 = v2[0] is a single element.
        let E_2: E::G2Affine = (v2[0] + setup.H_2 * s2[0] * gamma_inv).into();

        eval_proof.write_G1_message(transcript, E_1);
        eval_proof.write_G2_message(transcript, E_2);

        end_timer!(step);
        end_timer!(timer);

        Some(eval_proof)
    } else {
        end_timer!(timer);
        None
    }
}

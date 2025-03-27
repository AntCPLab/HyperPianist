//! Module containing the `DoryCommitment` type and its implementation.

use crate::{base::pairings, SubProverSetup};
use ark_ec::{
    pairing::{Pairing, PairingOutput},
    VariableBaseMSM,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{start_timer, end_timer};
use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};
use num_traits::One;
use std::iter::zip;

use rayon::iter::{IntoParallelIterator, ParallelIterator, IntoParallelRefIterator};

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct DeDoryCommitment<E: Pairing> {
    pub sub_mat_comm: PairingOutput<E>,
    pub sub_T_prime_vec: Vec<E::G1Affine>,
}

/// The default for GT is the the additive identity, but should be the
/// multiplicative identity.
impl<E: Pairing> Default for DeDoryCommitment<E> {
    fn default() -> Self {
        Self {
            sub_mat_comm: PairingOutput(One::one()),
            sub_T_prime_vec: Vec::new(),
        }
    }
}

impl<E: Pairing> DeDoryCommitment<E> {
    pub fn sub_commit(
        // sub-prover id
        sub_prover_id: usize,
        sub_witness_vec: &[<E as Pairing>::ScalarField],
        m: usize,
        n: usize,
        setup: &SubProverSetup<E>,
    ) -> Self
    {
        // Assume the matrix is well-formed.
        // Size of sub-prover's witness matrix should be 2^{(n-m)/2} times 2^{(n-m)/2.
        let timer = start_timer!(|| "deCommit");
        let sub_mat_len: usize = 1usize << ((n - m) / 2);
        // assert!(sub_mat_len > 0);
        // Check: 2^{(n-m)/2} * 2^m < 2^max_num
        // assert!((sub_mat_len << m) <= (1usize << setup.max_num));

        let step = start_timer!(|| "Calculate subgamma");
        let (SubGamma_1, SubGamma_2): (Vec<E::G1Affine>, Vec<E::G2Affine>) = (0..sub_mat_len)
        .map(|i| {
            (setup.Gamma_1[sub_prover_id + (i << m)].clone(), setup.Gamma_2[sub_prover_id + (i << m)].clone())
        }).unzip();
        end_timer!(step);

        // let r_rows_i = (0..n_rows_each_sp).map(|_| F::rand(rng)
        // ).collect::<Vec<F>>(); let r_fin = F::rand(rng);

        // Compute commitments for the rows.
        let step = start_timer!(|| "Calculate sub row comms");
        let total_rows = if sub_witness_vec.len() == 1usize << (n - m) {
            sub_mat_len
        } else {
            sub_mat_len / 2
        };
        let sub_row_comms = (0..total_rows)
            .into_par_iter()
            .map(|i| {
                let start = i * sub_mat_len;
                let end = (i + 1) * sub_mat_len;
                E::G1MSM::msm_unchecked(
                    &SubGamma_1, 
                    &sub_witness_vec[start..end],
                ) // + setup.H_1 * r_rows_i[i]
            })
            .collect::<Vec<E::G1MSM>>();
        end_timer!(step);

        // Compute the commitment to the entire sub-matrix.
        let step = start_timer!(|| "Sub mat comm");
        let sub_mat_comm = pairings::multi_pairing(
            &sub_row_comms,
            &SubGamma_2[..sub_row_comms.len()],
        ); // + pairings::pairing(setup.H_1, setup.H_2) * r_fin;
        end_timer!(step);

        let step = start_timer!(|| "Sub T prime vec");
        let sub_T_prime_vec = sub_row_comms
            .par_iter()
            .map(|T| (*T).into())
            .collect::<Vec<E::G1Affine>>();
        end_timer!(step);

        end_timer!(timer);

        Self {
            sub_mat_comm,
            sub_T_prime_vec,
        }
    }

    // Distributed Commit
    pub fn deCommit(
        sub_prover_id: usize,
        sub_witness_vec: &[<E as Pairing>::ScalarField],
        m: usize,
        n: usize,
        setup: &SubProverSetup<E>,
    ) -> (Option<PairingOutput<E>>, Vec<E::G1Affine>)
    {
        let sub_comm = Self::sub_commit(sub_prover_id, sub_witness_vec, m, n, setup);
        let sub_comms = Net::send_to_master(&sub_comm.sub_mat_comm);
        if Net::am_master() {
            (Some(sub_comms.unwrap().par_iter().sum()), sub_comm.sub_T_prime_vec)
        } else {
            (None, sub_comm.sub_T_prime_vec)
        }
    }

    pub fn multi_deCommit(
        sub_prover_id: usize,
        sub_witness_vec: Vec<&[E::ScalarField]>,
        m: usize,
        n: usize,
        setup: &SubProverSetup<E>,
    ) -> (Vec<Option<PairingOutput<E>>>, Vec<Vec<E::G1Affine>>)
    {
        let (sub_comms, sub_T_prime_vecs) : (Vec<_>, Vec<_>) = sub_witness_vec.par_iter()
            .map(|sub_witness_vec| {
                let comm = Self::sub_commit(sub_prover_id, sub_witness_vec, m, n, setup);
                (comm.sub_mat_comm, comm.sub_T_prime_vec)
            })
            .unzip();
        let all_sub_comms = Net::send_to_master(&sub_comms);
        if Net::am_master() {
            let all_sub_comms = all_sub_comms.unwrap();
            let sub_comms = all_sub_comms.into_iter().reduce(
                |a, b| zip(a, b).map(|(x, y)| x + y).collect()
            ).unwrap();
            (sub_comms.into_iter().map(|comm| Some(comm)).collect(), sub_T_prime_vecs)
        } else {
            (vec![], sub_T_prime_vecs)
        }
    }
}

//! Module containing the `DoryCommitment` type and its implementation.

use crate::{base::pairings, ProverSetup};
use ark_ec::{
    pairing::{Pairing, PairingOutput},
    VariableBaseMSM,
};
use num_traits::One;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub struct DoryCommitment<E: Pairing> {
    pub comm: PairingOutput<E>,
    pub T_vec_prime: Vec<E::G1Affine>,
}

/// The default for GT is the the additive identity, but should be the
/// multiplicative identity.
impl<E: Pairing> Default for DoryCommitment<E> {
    fn default() -> Self {
        Self {
            comm: PairingOutput(One::one()),
            T_vec_prime: Vec::new(),
        }
    }
}

impl<E: Pairing> DoryCommitment<E> {
    pub fn commit(
        witness_matrix: &[E::ScalarField],
        setup: &ProverSetup<E>,
        n: usize,
    ) -> Self
    {
        // Assume the matrix is well-formed.
        let n_mat = 1usize << (n / 2);
        let n_rows = if witness_matrix.len() == 1usize << n {
            n_mat
        } else {
            n_mat / 2
        };
        assert!(n_rows <= (1usize << setup.max_num));

        // Compute commitments for the rows.
        let row_comms = (0..n_rows)
            .into_par_iter()
            .map(|i| {
                // assert_eq!(n_rows, witness_matrix[i].len());
                E::G1MSM::msm_unchecked(
                    &setup.Gamma_1.last().unwrap()[..n_mat], /* Gamma_1.last() is the full
                                                               * vector of Gamma_1 */
                    &witness_matrix[i * n_mat..(i + 1) * n_mat],
                ) // + setup.H_1 * r_rows[i]
            })
            .collect::<Vec<E::G1MSM>>();

        // Compute the commitment for the entire matrix.
        let comm = pairings::multi_pairing(
            &row_comms,
            &setup.Gamma_2.last().unwrap()[..n_rows], /* Gamma_2.last() is the full vector of
                                                       * Gamma_2 */
        ); // + pairings::pairing(setup.H_1, setup.H_2) * r_fin;

        let T_vec_prime = row_comms
            .iter()
            .map(|T| (*T).into())
            .collect::<Vec<E::G1Affine>>();

        Self { comm, T_vec_prime }
    }
}

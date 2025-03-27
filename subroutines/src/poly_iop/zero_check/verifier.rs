// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Verifier subroutines for a ZeroCheck protocol.

use super::{ZeroCheckSubClaim, ZeroCheckVerifier};
use crate::poly_iop::{
    errors::PolyIOPErrors,
    structs::{IOPProverMessage, IOPVerifierState}, sum_check::SumCheckVerifier,
};
use arithmetic::{VPAuxInfo, interpolate_uni_poly};
use ark_ff::PrimeField;
use ark_std::{end_timer, start_timer};
use transcript::IOPTranscript;

#[cfg(feature = "parallel")]
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

pub struct ZeroCheckVerifierState<F: PrimeField> {
    pub iop: IOPVerifierState<F>,
    pub(crate) zerocheck_r: Vec<F>,
}

impl<F: PrimeField> ZeroCheckVerifier<F> for ZeroCheckVerifierState<F> {
    type VPAuxInfo = VPAuxInfo<F>;
    type ProverMessage = IOPProverMessage<F>;
    type Challenge = F;
    type Transcript = IOPTranscript<F>;
    type ZeroCheckSubClaim = ZeroCheckSubClaim<F>;

    /// Initialize the verifier's state.
    fn verifier_init(index_info: &Self::VPAuxInfo, zerocheck_r: Vec<F>) -> Self {
        let start = start_timer!(|| "zero check verifier init");
        let res = Self {
            iop: IOPVerifierState {
                round: 1,
                num_vars: index_info.num_variables,
                max_degree: index_info.max_degree,
                finished: false,
                polynomials_received: Vec::with_capacity(index_info.num_variables),
                challenges: Vec::with_capacity(index_info.num_variables),
            },
            zerocheck_r,
        };
        end_timer!(start);
        res
    }

    /// Run verifier for the current round, given a prover message.
    ///
    /// Note that `verify_round_and_update_state` only samples and stores
    /// challenges; and update the verifier's state accordingly. The actual
    /// verifications are deferred (in batch) to `check_and_generate_subclaim`
    /// at the last step.
    fn verify_round_and_update_state(
        &mut self,
        prover_msg: &Self::ProverMessage,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::Challenge, PolyIOPErrors> {
        self.iop.verify_round_and_update_state(prover_msg, transcript)
    }

    /// This function verifies the deferred checks in the interactive version of
    /// the protocol; and generate the subclaim. Returns an error if the
    /// proof failed to verify.
    ///
    /// If the asserted sum is correct, then the multilinear polynomial
    /// evaluated at `subclaim.point` will be `subclaim.expected_evaluation`.
    /// Otherwise, it is highly unlikely that those two will be equal.
    /// Larger field size guarantees smaller soundness error.
    fn check_and_generate_subclaim(
        &self,
        asserted_sum: &F,
    ) -> Result<Self::ZeroCheckSubClaim, PolyIOPErrors> {
        let start = start_timer!(|| "zero check check and generate subclaim");
        if !self.iop.finished {
            return Err(PolyIOPErrors::InvalidVerifier(
                "Incorrect verifier state: Verifier has not finished.".to_string(),
            ));
        }

        if self.iop.polynomials_received.len() != self.iop.num_vars {
            return Err(PolyIOPErrors::InvalidVerifier(
                "insufficient rounds".to_string(),
            ));
        }

        // the deferred check during the interactive phase:
        // 2. set `expected` to P(r)`
        #[cfg(feature = "parallel")]
        let mut expected_vec = self
            .iop.polynomials_received
            .clone()
            .into_par_iter()
            .zip(self.iop.challenges.clone().into_par_iter())
            .map(|(evaluations, challenge)| {
                if evaluations.len() != self.iop.max_degree + 1 {
                    return Err(PolyIOPErrors::InvalidVerifier(format!(
                        "incorrect number of evaluations: {} vs {}",
                        evaluations.len(),
                        self.iop.max_degree + 1
                    )));
                }
                Ok(interpolate_uni_poly::<F>(&evaluations, challenge))
            })
            .collect::<Result<Vec<_>, PolyIOPErrors>>()?;

        #[cfg(not(feature = "parallel"))]
        let mut expected_vec = self
            .iop.polynomials_received
            .clone()
            .into_iter()
            .zip(self.iop.challenges.clone().into_iter())
            .map(|(evaluations, challenge)| {
                if evaluations.len() != self.iop.max_degree + 1 {
                    return Err(PolyIOPErrors::InvalidVerifier(format!(
                        "incorrect number of evaluations: {} vs {}",
                        evaluations.len(),
                        self.max_degree + 1
                    )));
                }
                Ok(interpolate_uni_poly::<F>(&evaluations, challenge))
            })
            .collect::<Result<Vec<_>, PolyIOPErrors>>()?;

        // insert the asserted_sum to the first position of the expected vector
        expected_vec.insert(0, *asserted_sum);

        let first_evaluations = &self.iop.polynomials_received[0];
        if first_evaluations[0] != F::zero() || first_evaluations[1] != F::zero() {
            return Err(PolyIOPErrors::InvalidProof(
                "First polynomial is not 0 on both sides.".to_string(),
            ));
        }

        for (round, (evaluations, &expected)) in self
            .iop.polynomials_received
            .iter()
            .zip(expected_vec.iter())
            .take(self.iop.num_vars)
            .enumerate()
            .skip(1)
        {
            // the deferred check during the interactive phase:
            // 1. check if the received 'P(0) + P(1) = expected`.
            let alpha = self.zerocheck_r[round];
            if evaluations[0] + alpha * (evaluations[1] - evaluations[0]) != expected {
                return Err(PolyIOPErrors::InvalidProof(
                    "Prover message is not consistent with the claim.".to_string(),
                ));
            }
        }
        end_timer!(start);
        Ok(ZeroCheckSubClaim {
            point: self.iop.challenges.clone(),
            // the last expected value (not checked within this function) will be included in the
            // subclaim
            expected_evaluation: expected_vec[self.iop.num_vars],
            init_challenge: self.zerocheck_r.clone(),
        })
    }
}

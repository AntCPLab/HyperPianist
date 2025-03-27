// Copyright (c) Jolt Project
// Copyright (c) 2023 HyperPlonk Project
// Copyright (c) 2024 HyperPianist Project

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! This module implements batched cubic sumchecks

use arithmetic::{
    bind_poly_var_bot, bit_decompose, build_eq_table, eq_eval,
    interpolate_uni_poly,
    math::Math, unipoly::UniPoly,
};
use ark_ff::{batch_inversion, PrimeField};
use ark_poly::DenseMultilinearExtension;
use rayon::iter::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use transcript::IOPTranscript;

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

use crate::drop_in_background_thread;

use super::generic_sumcheck::ZerocheckInstanceProof;

// A cubic sumcheck instance that is not represented as virtual polynomials.
// Instead the struct itself can hold arbitrary state as long as it can bind
// varaibles and produce a cubic polynomial on demand.
// Used by the layered circuit implementation for rational sumcheck
pub trait BatchedCubicSumcheckInstance<F: PrimeField>: Sync {
    fn num_rounds(&self) -> usize;
    fn bind(&mut self, r: &F);
    // Returns evals at 0, 2, 3
    fn compute_cubic(&self, coeffs: &[F], eq: &[F], lambda: &F) -> (F, F);
    fn final_claims(&self) -> Vec<Vec<F>>;
    fn compute_cubic_direct(
        &self,
        coeffs: &[F],
        evaluations: &[Vec<DenseMultilinearExtension<F>>],
        eq: &[F],
        lambda: &F,
    ) -> (F, F);

    // #[tracing::instrument(skip_all, name =
    // "BatchedCubicSumcheck::prove_sumcheck")]
    fn prove_sumcheck(
        &mut self,
        claim: &F,
        coeffs: &[F],
        zerocheck_r: &[F],
        transcript: &mut IOPTranscript<F>,
        lambda: &F,
    ) -> (ZerocheckInstanceProof<F>, Vec<F>, Vec<Vec<F>>) {
        let mut previous_claim = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut cubic_polys = Vec::new();

        let mut zerocheck_r = zerocheck_r.to_vec();
        zerocheck_r.reverse();
        let mut zerocheck_r_inv = zerocheck_r.to_vec();
        batch_inversion(&mut zerocheck_r_inv);

        let eq_table = build_eq_table(&zerocheck_r, F::one());

        for round in 0..self.num_rounds() {
            let evals = self.compute_cubic(coeffs, &eq_table[round], lambda);
            let evals = vec![
                evals.0,
                zerocheck_r_inv[round]
                    * (previous_claim - (F::one() - zerocheck_r[round]) * evals.0),
                evals.1,
            ];
            let poly = UniPoly::from_evals(&evals);
            // append the prover's message to the transcript
            transcript
                .append_serializable_element(b"poly", &poly)
                .unwrap();
            // derive the verifier's challenge for the next round
            let r_j = transcript
                .get_and_append_challenge(b"challenge_nextround")
                .unwrap();

            r.push(r_j);
            // bind polynomials to verifier's challenge
            self.bind(&r_j);

            previous_claim = poly.evaluate(&r_j);
            cubic_polys.push(poly);
        }
        drop_in_background_thread(eq_table);

        (
            ZerocheckInstanceProof::new(cubic_polys),
            r,
            self.final_claims(),
        )
    }

    fn d_prove_sumcheck(
        &mut self,
        claim: &F,
        coeffs: &[F],
        zerocheck_r: &[F],
        transcript: &mut IOPTranscript<F>,
        lambda: &F,
    ) -> Option<(ZerocheckInstanceProof<F>, Vec<F>, Vec<Vec<Vec<F>>>)> {
        let num_party_vars = Net::n_parties().log_2();
        let length = zerocheck_r.len() - num_party_vars;

        let mut previous_claim = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut cubic_polys = Vec::new();

        // rev() because this is big endian
        let index_vec: Vec<F> = bit_decompose(Net::party_id() as u64, num_party_vars)
            .into_iter()
            .rev()
            .map(|x| F::from(x))
            .collect();
        let eq_coeff = eq_eval(&index_vec, &zerocheck_r[..num_party_vars]).unwrap();

        let mut zerocheck_r = zerocheck_r.to_vec();
        zerocheck_r.reverse();
        let mut zerocheck_r_inv = zerocheck_r.to_vec();
        batch_inversion(&mut zerocheck_r_inv);

        let eq_table = build_eq_table(&zerocheck_r[..length], eq_coeff);
        for round in 0..self.num_rounds() {
            let evals = self.compute_cubic(coeffs, &eq_table[round], lambda);
            let messages = Net::send_to_master(&evals);

            let mut poly = None;
            let r_j = if Net::am_master() {
                let evals = messages
                    .unwrap()
                    .iter()
                    .fold((F::zero(), F::zero()), |acc, x| (acc.0 + x.0, acc.1 + x.1));
                let evals = vec![
                    evals.0,
                    zerocheck_r_inv[round]
                        * (previous_claim - (F::one() - zerocheck_r[round]) * evals.0),
                    evals.1,
                ];
                poly = Some(UniPoly::from_evals(&evals));
                // append the prover's message to the transcript
                transcript
                    .append_serializable_element(b"poly", poly.as_ref().unwrap())
                    .unwrap();
                // derive the verifier's challenge for the next round
                let r_j = transcript
                    .get_and_append_challenge(b"challenge_nextround")
                    .unwrap();
                Net::recv_from_master_uniform(Some(r_j))
            } else {
                Net::recv_from_master_uniform(None)
            };

            r.push(r_j);
            // bind polynomials to verifier's challenge
            self.bind(&r_j);

            if Net::am_master() {
                let poly = poly.unwrap();
                previous_claim = poly.evaluate(&r_j);
                cubic_polys.push(poly);
            }
        }
        drop_in_background_thread(eq_table);

        // Dimensions: (party, poly_id, batch_index)
        let final_claims = Net::send_to_master(&self.final_claims());
        if !Net::am_master() {
            return None;
        }

        let final_claims = final_claims.unwrap();
        let mut polys = (0..final_claims[0].len())
            .into_par_iter()
            .map(|poly_id| {
                (0..final_claims[0][poly_id].len())
                    .map(|batch_index| {
                        DenseMultilinearExtension::from_evaluations_vec(
                            num_party_vars,
                            final_claims
                                .iter()
                                .map(|claim| claim[poly_id][batch_index])
                                .collect(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let eq_table = build_eq_table(&zerocheck_r[length..], F::one());
        for round in 0..num_party_vars {
            let evals = self.compute_cubic_direct(coeffs, &polys, &eq_table[round], lambda);
            let evals = vec![
                evals.0,
                zerocheck_r_inv[length + round]
                    * (previous_claim - (F::one() - zerocheck_r[length + round]) * evals.0),
                evals.1,
            ];
            let poly = UniPoly::from_evals(&evals);
            // append the prover's message to the transcript
            transcript
                .append_serializable_element(b"poly", &poly)
                .unwrap();
            // derive the verifier's challenge for the next round
            let r_j = transcript
                .get_and_append_challenge(b"challenge_nextround")
                .unwrap();

            r.push(r_j);
            
            polys.par_iter_mut().for_each(|polys| {
                for poly in polys {
                    bind_poly_var_bot(poly, &r_j);
                }
            });

            previous_claim = poly.evaluate(&r_j);
            cubic_polys.push(poly);
        }

        Some((ZerocheckInstanceProof::new(cubic_polys), r, final_claims))
    }
}

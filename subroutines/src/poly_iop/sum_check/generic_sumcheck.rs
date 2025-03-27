use crate::poly_iop::errors::PolyIOPErrors;
use arithmetic::{
    bind_poly_var_bot_par, bit_decompose, build_eq_table, eq_eval,
    math::Math,
    unipoly::{CompressedUniPoly, UniPoly},
};
use ark_ff::{batch_inversion, PrimeField};
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{end_timer, start_timer};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use std::{fmt::Debug, iter::zip, sync::Arc};
use transcript::IOPTranscript;

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct SumcheckInstanceProof<F: PrimeField> {
    compressed_polys: Vec<CompressedUniPoly<F>>,
}

impl<F: PrimeField> SumcheckInstanceProof<F> {
    pub fn new(compressed_polys: Vec<CompressedUniPoly<F>>) -> SumcheckInstanceProof<F> {
        SumcheckInstanceProof { compressed_polys }
    }

    fn prove_arbitrary_step<Func>(
        polys: &mut Vec<Arc<DenseMultilinearExtension<F>>>,
        comb_func: &Func,
        combined_degree: usize,
    ) -> Vec<F>
    where
        Func: Fn(&[F]) -> F + std::marker::Sync,
    {
        let start = start_timer!(|| "sum check step");
        // Vector storing evaluations of combined polynomials g(x) = P_0(x) * ...
        // P_{num_polys} (x) for points {0, ..., |g(x)|}
        let mut eval_points = vec![F::zero(); combined_degree + 1];

        let mle_half = polys[0].evaluations.len() / 2;

        let accum: Vec<Vec<F>> = (0..mle_half)
            .into_par_iter()
            .map(|poly_term_i| {
                let mut accum = vec![F::zero(); combined_degree + 1];
                // Evaluate P({0, ..., |g(r)|})

                // TODO(#28): Optimize
                // Tricks can be used here for low order bits {0,1} but general premise is a
                // running sum for each of the m terms in the Dense multilinear
                // polynomials. Formula is: half = | D_{n-1} | / 2
                // D_n(index, r) = D_{n-1}[half + index] + r * (D_{n-1}[half + index] -
                // D_{n-1}[index])

                // eval 0: bound_func is A(low)
                let params_zero: Vec<F> = polys.iter().map(|poly| poly[2 * poly_term_i]).collect();
                accum[0] += comb_func(&params_zero);

                // TODO(#28): Can be computed from prev_round_claim - eval_point_0
                let params_one: Vec<F> =
                    polys.iter().map(|poly| poly[2 * poly_term_i + 1]).collect();
                // accum[1] += comb_func(&params_one);

                // D_n(index, r) = D_{n-1}[half + index] + r * (D_{n-1}[half + index] -
                // D_{n-1}[index]) D_n(index, 0) = D_{n-1}[LOW]
                // D_n(index, 1) = D_{n-1}[HIGH]
                // D_n(index, 2) = D_{n-1}[HIGH] + (D_{n-1}[HIGH] - D_{n-1}[LOW])
                // D_n(index, 3) = D_{n-1}[HIGH] + (D_{n-1}[HIGH] - D_{n-1}[LOW]) +
                // (D_{n-1}[HIGH] - D_{n-1}[LOW]) ...
                let mut existing_term = params_one;
                for eval_i in 2..(combined_degree + 1) {
                    let mut poly_evals = vec![F::zero(); polys.len()];
                    for poly_i in 0..polys.len() {
                        let poly = &polys[poly_i];
                        poly_evals[poly_i] = existing_term[poly_i] + poly[2 * poly_term_i + 1]
                            - poly[2 * poly_term_i];
                    }

                    accum[eval_i] += comb_func(&poly_evals);
                    existing_term = poly_evals;
                }
                accum
            })
            .collect();

        eval_points
            .par_iter_mut()
            .enumerate()
            .for_each(|(poly_i, eval_point)| {
                *eval_point = accum
                    .par_iter()
                    .take(mle_half)
                    .map(|mle| mle[poly_i])
                    .sum::<F>();
            });
        end_timer!(start);
        eval_points
    }

    /// Create a sumcheck proof for polynomial(s) of arbitrary degree.
    ///
    /// Params
    /// - `claim`: Claimed sumcheck evaluation (note: currently unused)
    /// - `num_rounds`: Number of rounds of sumcheck, or number of variables to
    ///   bind
    /// - `polys`: Dense polynomials to combine and sumcheck
    /// - `comb_func`: Function used to combine each polynomial evaluation
    /// - `transcript`: Fiat-shamir transcript
    ///
    /// Returns (SumcheckInstanceProof, r_eval_point, final_evals)
    /// - `r_eval_point`: Final random point of evaluation
    /// - `final_evals`: Each of the polys evaluated at `r_eval_point`
    // #[tracing::instrument(skip_all, name = "Sumcheck.prove")]
    pub fn prove_arbitrary<Func>(
        claim: &F,
        num_rounds: usize,
        polys: &mut Vec<Arc<DenseMultilinearExtension<F>>>,
        comb_func: Func,
        combined_degree: usize,
        transcript: &mut IOPTranscript<F>,
    ) -> (Self, Vec<F>, Vec<F>)
    where
        Func: Fn(&[F]) -> F + std::marker::Sync,
    {
        let mut r: Vec<F> = Vec::new();
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        let mut previous_claim = claim.clone();

        for _round in 0..num_rounds {
            let mut eval_points = Self::prove_arbitrary_step(polys, &comb_func, combined_degree);
            eval_points[1] = previous_claim - eval_points[0];
            let step = start_timer!(|| "from evals");
            let round_uni_poly = UniPoly::from_evals(&eval_points);
            end_timer!(step);

            // append the prover's message to the transcript
            transcript
                .append_serializable_element(b"poly", &round_uni_poly)
                .unwrap();
            let r_j = transcript
                .get_and_append_challenge(b"challenge_nextround")
                .unwrap();
            r.push(r_j);

            // bound all tables to the verifier's challenege
            let step = start_timer!(|| "bind polys");
            let concurrency = (rayon::current_num_threads() * 2 + polys.len() - 1) / polys.len();
            polys.par_iter_mut().for_each(|poly| {
                bind_poly_var_bot_par(Arc::get_mut(poly).unwrap(), &r_j, concurrency)
            });
            compressed_polys.push(round_uni_poly.compress());
            previous_claim = round_uni_poly.evaluate(&r_j);
            end_timer!(step);
        }

        let final_evals = polys.iter().map(|poly| poly[0]).collect();

        (SumcheckInstanceProof::new(compressed_polys), r, final_evals)
    }

    pub fn d_prove_arbitrary<Func>(
        claim: &F,
        num_rounds: usize,
        polys: &mut Vec<Arc<DenseMultilinearExtension<F>>>,
        comb_func: Func,
        combined_degree: usize,
        transcript: &mut IOPTranscript<F>,
    ) -> Option<(Self, Vec<F>, Vec<F>)>
    where
        Func: Fn(&[F]) -> F + std::marker::Sync,
    {
        let mut r: Vec<F> = Vec::new();
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();
        let mut previous_claim = claim.clone();

        for _round in 0..num_rounds {
            let eval_points = Self::prove_arbitrary_step(polys, &comb_func, combined_degree);
            let all_eval_points = Net::send_to_master(&eval_points);

            // append the prover's message to the transcript
            let r_j = if Net::am_master() {
                let all_eval_points = all_eval_points.unwrap();
                let mut eval_points = all_eval_points
                    .iter()
                    .fold(vec![F::zero(); all_eval_points[0].len()], |acc, x| {
                        zip(&acc, x).map(|(acc, x)| *acc + *x).collect()
                    });
                eval_points[1] = previous_claim - eval_points[0];

                let step = start_timer!(|| "from evals");
                let round_uni_poly = UniPoly::from_evals(&eval_points);
                end_timer!(step);

                transcript
                    .append_serializable_element(b"poly", &round_uni_poly)
                    .unwrap();
                compressed_polys.push(round_uni_poly.compress());

                let r_j = transcript
                    .get_and_append_challenge(b"challenge_nextround")
                    .unwrap();
                previous_claim = round_uni_poly.evaluate(&r_j);
                Net::recv_from_master_uniform(Some(r_j))
            } else {
                Net::recv_from_master_uniform(None)
            };
            r.push(r_j);

            // bound all tables to the verifier's challenege
            let step = start_timer!(|| "bind polys");
            let concurrency = (rayon::current_num_threads() * 2 + polys.len() - 1) / polys.len();
            polys.par_iter_mut().for_each(|poly| {
                bind_poly_var_bot_par(Arc::get_mut(poly).unwrap(), &r_j, concurrency)
            });
            end_timer!(step);
        }

        let final_evals = polys.iter().map(|poly| poly[0]).collect::<Vec<_>>();
        let all_final_evals = Net::send_to_master(&final_evals);

        if !Net::am_master() {
            return None;
        }

        let num_party_vars = Net::n_parties().log_2();
        let all_final_evals = all_final_evals.unwrap();
        let mut polys = (0..all_final_evals[0].len())
            .into_par_iter()
            .map(|poly_id| {
                Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                    num_party_vars,
                    all_final_evals
                        .iter()
                        .map(|party_evals| party_evals[poly_id])
                        .collect(),
                ))
            })
            .collect::<Vec<_>>();
        let (mut proof, mut r_final, final_evals) = Self::prove_arbitrary(
            &previous_claim,
            num_party_vars,
            &mut polys,
            comb_func,
            combined_degree,
            transcript,
        );
        compressed_polys.append(&mut proof.compressed_polys);
        r.append(&mut r_final);

        Some((SumcheckInstanceProof::new(compressed_polys), r, final_evals))
    }

    /// Verify this sumcheck proof.
    /// Note: Verification does not execute the final check of sumcheck
    /// protocol: g_v(r_v) = oracle_g(r), as the oracle is not passed in.
    /// Expected that the caller will implement.
    ///
    /// Params
    /// - `claim`: Claimed evaluation
    /// - `num_rounds`: Number of rounds of sumcheck, or number of variables to
    ///   bind
    /// - `degree_bound`: Maximum allowed degree of the combined univariate
    ///   polynomial
    /// - `transcript`: Fiat-shamir transcript
    ///
    /// Returns (e, r)
    /// - `e`: Claimed evaluation at random point
    /// - `r`: Evaluation point
    /// #[tracing::instrument(skip_all, name = "Sumcheck::verify")]
    pub fn verify(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut IOPTranscript<F>,
    ) -> Result<(F, Vec<F>), PolyIOPErrors> {
        let mut e = claim;
        let mut r: Vec<F> = Vec::new();

        // verify that there is a univariate polynomial for each round
        assert_eq!(self.compressed_polys.len(), num_rounds);
        for i in 0..self.compressed_polys.len() {
            let poly = self.compressed_polys[i].decompress(&e);

            // verify degree bound
            if poly.degree() != degree_bound {
                return Err(PolyIOPErrors::InvalidProof(format!(
                    "degree_bound = {}, poly.degree() = {}",
                    degree_bound,
                    poly.degree(),
                )));
            }

            // append the prover's message to the transcript
            transcript.append_serializable_element(b"poly", &poly)?;

            // derive the verifier's challenge for the next round
            let r_i = transcript.get_and_append_challenge(b"challenge_nextround")?;

            r.push(r_i);

            // evaluate the claimed degree-ell polynomial at r_i
            e = poly.evaluate(&r_i);
        }

        Ok((e, r))
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct ZerocheckInstanceProof<F: PrimeField> {
    pub(crate) polys: Vec<UniPoly<F>>,
}

impl<F: PrimeField> ZerocheckInstanceProof<F> {
    pub fn new(polys: Vec<UniPoly<F>>) -> ZerocheckInstanceProof<F> {
        ZerocheckInstanceProof { polys }
    }

    /// Verify this sumcheck proof.
    /// Note: Verification does not execute the final check of sumcheck
    /// protocol: g_v(r_v) = oracle_g(r), as the oracle is not passed in.
    /// Expected that the caller will implement.
    ///
    /// Params
    /// - `claim`: Claimed evaluation
    /// - `num_rounds`: Number of rounds of sumcheck, or number of variables to
    ///   bind
    /// - `degree_bound`: Maximum allowed degree of the combined univariate
    ///   polynomial
    /// - `transcript`: Fiat-shamir transcript
    ///
    /// Returns (e, r)
    /// - `e`: Claimed evaluation at random point
    /// - `r`: Evaluation point
    /// #[tracing::instrument(skip_all, name = "Sumcheck::verify")]
    pub fn verify(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        zerocheck_r: &[F],
        transcript: &mut IOPTranscript<F>,
    ) -> Result<(F, Vec<F>), PolyIOPErrors> {
        let mut e = claim;
        let mut r: Vec<F> = Vec::new();

        // verify that there is a univariate polynomial for each round
        assert_eq!(self.polys.len(), num_rounds);
        for i in 0..self.polys.len() {
            let poly = &self.polys[i];
            // append the prover's message to the transcript
            transcript.append_serializable_element(b"poly", poly)?;

            // verify degree bound
            if poly.degree() != degree_bound {
                return Err(PolyIOPErrors::InvalidProof(format!(
                    "degree_bound = {}, poly.degree() = {}",
                    degree_bound,
                    poly.degree(),
                )));
            }

            if poly.coeffs[0] + zerocheck_r[i] * (poly.coeffs.iter().skip(1).sum::<F>()) != e {
                return Err(PolyIOPErrors::InvalidProof(
                    "Inconsistent message".to_string(),
                ));
            }

            // derive the verifier's challenge for the next round
            let r_i = transcript.get_and_append_challenge(b"challenge_nextround")?;

            r.push(r_i);

            // evaluate the claimed degree-ell polynomial at r_i
            e = poly.evaluate(&r_i);
        }

        Ok((e, r))
    }
}

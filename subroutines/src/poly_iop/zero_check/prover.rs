// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Prover subroutines for a ZeroCheck protocol.

use super::ZeroCheckProver;
use crate::{
    barycentric_weights, extrapolate,
    poly_iop::{
        errors::PolyIOPErrors,
        structs::{IOPProverMessage, IOPProverState},
        sum_check::SumCheckProver,
    },
};
use arithmetic::{
    bind_poly_var_bot, bind_poly_var_bot_par, build_eq_table, build_eq_x_r_vec, VirtualPolynomial,
};
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_std::{cfg_into_iter, end_timer, start_timer, vec::Vec};
use itertools::Itertools;
use rayon::{
    iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator},
    prelude::IntoParallelIterator,
};
use std::{mem::take, sync::Arc};

pub struct ZeroCheckProverState<F: PrimeField> {
    pub iop: IOPProverState<F>,
    pub(crate) eq_table: Vec<Vec<F>>,
}

impl<F: PrimeField> ZeroCheckProver<F> for ZeroCheckProverState<F> {
    type VirtualPolynomial = VirtualPolynomial<F>;
    type ProverMessage = IOPProverMessage<F>;

    /// Initialize the prover state to argue for the sum of the input polynomial
    /// over {0,1}^`num_vars`.
    fn prover_init(
        polynomial: Self::VirtualPolynomial,
        zerocheck_r: &[F],
        coeff: F,
    ) -> Result<Self, PolyIOPErrors> {
        let start = start_timer!(|| "sum check prover init");
        if polynomial.aux_info.num_variables == 0 {
            return Err(PolyIOPErrors::InvalidParameters(
                "Attempt to prove a constant.".to_string(),
            ));
        }
        end_timer!(start);

        let max_degree = polynomial.aux_info.max_degree;
        let (extrapolation_aux, eq_table) = rayon::join(
            || {
                (1..max_degree)
                    .map(|degree| {
                        let points = (0..1 + degree as u64).map(F::from).collect::<Vec<_>>();
                        let weights = barycentric_weights(&points);
                        (points, weights)
                    })
                    .collect()
            },
            || build_eq_table(&zerocheck_r, coeff),
        );

        Ok(Self {
            iop: IOPProverState {
                challenges: Vec::with_capacity(polynomial.aux_info.num_variables),
                round: 0,
                poly: polynomial,
                extrapolation_aux,
            },
            eq_table,
        })
    }

    /// Receive message from verifier, generate prover message, and proceed to
    /// next round.
    ///
    /// Main algorithm used is from section 3.2 of [XZZPS19](https://eprint.iacr.org/2019/317.pdf#subsection.3.2).
    fn prove_round_and_update_state(
        &mut self,
        challenge: &Option<F>,
    ) -> Result<Self::ProverMessage, PolyIOPErrors> {
        let start = start_timer!(|| format!(
            "sum check prove {}-th round and update state",
            self.iop.round
        ));

        if self.iop.round >= self.iop.poly.aux_info.num_variables {
            return Err(PolyIOPErrors::InvalidProver(
                "Prover is not active".to_string(),
            ));
        }

        let fix_argument = start_timer!(|| "fix argument");

        // Step 1:
        // fix argument and evaluate f(x) over x_m = r; where r is the challenge
        // for the current round, and m is the round number, indexed from 1
        //
        // i.e.:
        // at round m <= n, for each mle g(x_1, ... x_n) within the flattened_mle
        // which has already been evaluated to
        //
        //    g(r_1, ..., r_{m-1}, x_m ... x_n)
        //
        // eval g over r_m, and mutate g to g(r_1, ... r_m,, x_{m+1}... x_n)

        if let Some(chal) = challenge {
            if self.iop.round == 0 {
                return Err(PolyIOPErrors::InvalidProver(
                    "first round should be prover first.".to_string(),
                ));
            }
            self.iop.challenges.push(*chal);

            let r = self.iop.challenges[self.iop.round - 1];
            let concurrency = (rayon::current_num_threads() * 2
                + self.iop.poly.flattened_ml_extensions.len()
                - 1)
                / self.iop.poly.flattened_ml_extensions.len();
            self.iop
                .poly
                .flattened_ml_extensions
                .par_iter_mut()
                .for_each(|mle| bind_poly_var_bot_par(Arc::get_mut(mle).unwrap(), &r, concurrency));
        } else if self.iop.round > 0 {
            return Err(PolyIOPErrors::InvalidProver(
                "verifier message is empty".to_string(),
            ));
        }
        end_timer!(fix_argument);

        self.iop.round += 1;

        let products_list = self.iop.poly.products.clone();
        let mut products_sum = vec![F::zero(); self.iop.poly.aux_info.max_degree + 1];

        // Step 2: generate sum for the partial evaluated polynomial:
        // f(r_1, ... r_m,, x_{m+1}... x_n)

        let step = start_timer!(|| "products");

        products_list.iter().for_each(|(coefficient, products)| {
            let mut sum =
                cfg_into_iter!(0..1 << (self.iop.poly.aux_info.num_variables - self.iop.round))
                    .fold(
                        || {
                            (
                                vec![(F::zero(), F::zero()); products.len()],
                                vec![F::zero(); products.len() + 1],
                            )
                        },
                        |(mut buf, mut acc), b| {
                            let eq_eval = self.eq_table[self.iop.round - 1][b];
                            buf.iter_mut().zip(products.iter()).enumerate().for_each(
                                |(i, ((eval, step), f))| {
                                    let table = &self.iop.poly.flattened_ml_extensions[*f];
                                    if i == 0 {
                                        *eval = table[b << 1] * eq_eval;
                                        *step = table[(b << 1) + 1] * eq_eval - *eval;
                                    } else {
                                        *eval = table[b << 1];
                                        *step = table[(b << 1) + 1] - table[b << 1];
                                    }
                                },
                            );
                            acc[0] += buf.iter().map(|(eval, _)| eval).product::<F>();
                            acc[1..].iter_mut().for_each(|acc| {
                                buf.iter_mut().for_each(|(eval, step)| *eval += step as &_);
                                *acc += buf.iter().map(|(eval, _)| eval).product::<F>();
                            });
                            (buf, acc)
                        },
                    )
                    .map(|(_, partial)| partial)
                    .reduce(
                        || vec![F::zero(); products.len() + 1],
                        |mut sum, partial| {
                            sum.iter_mut()
                                .zip(partial.iter())
                                .for_each(|(sum, partial)| *sum += partial);
                            sum
                        },
                    );
            sum.iter_mut().for_each(|sum| *sum *= coefficient);
            let extraploation =
                cfg_into_iter!(0..self.iop.poly.aux_info.max_degree - products.len())
                    .map(|i| {
                        let (points, weights) = &self.iop.extrapolation_aux[products.len() - 1];
                        let at = F::from((products.len() + 1 + i) as u64);
                        extrapolate(points, weights, &sum, &at)
                    })
                    .collect::<Vec<_>>();
            products_sum
                .iter_mut()
                .zip(sum.iter().chain(extraploation.iter()))
                .for_each(|(products_sum, sum)| *products_sum += sum);
        });
        end_timer!(step);
        end_timer!(start);

        Ok(IOPProverMessage {
            evaluations: products_sum,
        })
    }

    fn get_final_mle_evaluations(&mut self, challenge: F) -> Result<Vec<F>, PolyIOPErrors> {
        self.iop.get_final_mle_evaluations(challenge)
    }
}

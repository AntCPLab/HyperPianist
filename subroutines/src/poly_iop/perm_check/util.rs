// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! This module implements useful functions for the permutation check protocol.

use crate::poly_iop::errors::PolyIOPErrors;
use arithmetic::math::Math;
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_std::{end_timer, start_timer};
use rayon::iter::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator};
use std::sync::Arc;

use deNetwork::{DeMultiNet as Net, DeNet};

/// Returns the evaluations of two list of MLEs:
/// - numerators = (a1, ..., ak)
/// - denominators = (b1, ..., bk)
///
///  where
///  - beta and gamma are challenges
///  - (f1, ..., fk), (g1, ..., gk),
///  - (s_id1, ..., s_idk), (perm1, ..., permk) are mle-s
///
/// - ai(x) is the MLE for `fi(x) + \beta s_id_i(x) + \gamma`
/// - bi(x) is the MLE for `gi(x) + \beta perm_i(x) + \gamma`
///
/// The caller is responsible for sanity-check
#[allow(clippy::type_complexity)]
pub(crate) fn compute_leaves<F: PrimeField, const DISTRIBUTED: bool>(
    beta: &F,
    gamma: &F,
    fxs: &[Arc<DenseMultilinearExtension<F>>],
    gxs: &[Arc<DenseMultilinearExtension<F>>],
    perms: &[Arc<DenseMultilinearExtension<F>>],
) -> Result<Vec<Vec<Vec<F>>>, PolyIOPErrors> {
    let timer = start_timer!(|| "compute numerators and denominators");

    let mut leaves = vec![];

    let mut shift = 0;

    let n_parties = if DISTRIBUTED {
        Net::n_parties() as u64
    } else {
        1
    };
    let mut start = 0;
    while start < fxs.len() {
        let num_vars = fxs[start].num_vars;
        let mut end = start + 1;
        while end < fxs.len() && fxs[end].num_vars == num_vars {
            end += 1;
        }

        let (mut numerators, mut denominators) = (start..end)
            .into_par_iter()
            .map(|l| {
                let eval_len = num_vars.pow2() as u64;

                let start = if DISTRIBUTED {
                    shift + eval_len * n_parties * ((l - start) as u64) + eval_len * (Net::party_id() as u64)
                } else {
                    shift + eval_len * ((l - start) as u64)
                };

                (
                    &fxs[l].evaluations,
                    &gxs[l].evaluations,
                    &perms[l].evaluations,
                )
                    .into_par_iter()
                    .enumerate()
                    .map(|(i, (&f_ev, &g_ev, &perm_ev))| {
                        let numerator = f_ev + *beta * F::from_u64(start + (i as u64)).unwrap() + gamma;
                        let denominator = g_ev + *beta * perm_ev + gamma;
                        (numerator, denominator)
                    })
                    .unzip::<_, _, Vec<_>, Vec<_>>()
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();
        numerators.append(&mut denominators);
        leaves.push(numerators);

        shift += ((end - start) as u64) * (num_vars.pow2() as u64) * n_parties;
        start = end;
    }

    end_timer!(timer);
    Ok(leaves)
}

// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! This module implements useful functions for the product check protocol.

#![allow(non_snake_case)]
use crate::poly_iop::{errors::PolyIOPErrors, structs::IOPProof, zero_check::ZeroCheck, lookup::instruction::JoltInstruction, PolyIOP};
use arithmetic::{get_index, VirtualPolynomial};
use ark_ff::{batch_inversion, PrimeField};
use ark_poly::DenseMultilinearExtension;
use ark_std::{end_timer, start_timer};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::sync::Arc;
use transcript::IOPTranscript;

pub trait SurgeCommons<F, Instruction, const C: usize, const M: usize>
where
    F: PrimeField,
    Instruction: JoltInstruction + Default,
{
    fn num_memories() -> usize {
        C * Self::num_subtables()
    }

    fn num_subtables() -> usize {
        // Add one for operand recovery
        Instruction::default().subtables::<F>(C, M).len() + 1
    }

    /// Maps an index [0, NUM_MEMORIES) -> [0, NUM_SUBTABLES)
    fn memory_to_subtable_index(i: usize) -> usize {
        i / C
    }

    /// Maps an index [0, NUM_MEMORIES) -> [0, C)
    fn memory_to_dimension_index(i: usize) -> usize {
        i % C
    }
}

impl<F, Instruction, const C: usize, const M: usize> SurgeCommons<F, Instruction, C, M> for PolyIOP<F>
where
    F: PrimeField,
    Instruction: JoltInstruction + Default, {}

// #[tracing::instrument(skip_all, name = "SurgeCommons::polys_from_evals")]
pub(super) fn polys_from_evals<F: PrimeField>(
    num_vars: usize,
    all_evals: &Vec<Vec<F>>,
) -> Vec<Arc<DenseMultilinearExtension<F>>> {
    all_evals
        .par_iter()
        .map(|evals| Arc::new(DenseMultilinearExtension::from_evaluations_vec(num_vars, evals.to_vec())))
        .collect()
}

// #[tracing::instrument(skip_all, name = "SurgeCommons::polys_from_evals_usize")]
pub(super) fn polys_from_evals_usize<F: PrimeField>(
    num_vars: usize,
    all_evals_usize: &Vec<Vec<usize>>,
) -> Vec<Arc<DenseMultilinearExtension<F>>> {
    all_evals_usize
        .par_iter()
        .map(|evals_usize| {
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                evals_usize.iter().map(|x| F::from_u64(*x as u64).unwrap()).collect(),
            ))
        })
        .collect()
}

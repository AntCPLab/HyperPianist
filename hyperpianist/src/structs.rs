// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Main module for the HyperPlonk PolyIOP.

use crate::{custom_gate::CustomizedGates, lookup::HyperPlonkLookupPlugin, prelude::HyperPlonkErrors, selectors::SelectorColumn, utils::PcsDynamicProof};
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use ark_std::{log2, Zero, One};
use std::sync::Arc;
use std::cmp::max;
use std::fmt::Debug;
use std::iter::zip;
use subroutines::{
    pcs::PolynomialCommitmentScheme,
    BatchProof, CombinedCheck,
};

/// The proof for the HyperPlonk PolyIOP, consists of the following:
///   - the commitments to all witness MLEs
///   - a batch opening to all the MLEs at certain index
///   - the zero-check proof for checking custom gate-satisfiability
///   - the permutation-check proof for checking the copy constraints
#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyperPlonkProof<E, CC, PCS>
where
    E: Pairing,
    CC: CombinedCheck<E, PCS>,
    PCS: PolynomialCommitmentScheme<E, BatchProof = BatchProof<E, PCS>>,
{
    // PCS commit for witnesses
    pub witness_commits: Vec<PCS::Commitment>,
    pub batch_openings: PcsDynamicProof<E, PCS>,
    // =======================================================================
    // IOP proofs
    // =======================================================================
    // the custom gate zerocheck proof
    pub combined_check_proof: CC::CombinedCheckProof,
}

/// The HyperPlonk instance parameters, consists of the following:
///   - the number of constraints
///   - number of public input columns
///   - the customized gate function
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HyperPlonkParams {
    /// the number of constraints for gate_func
    pub num_constraints: usize,
    pub num_lookup_constraints: Vec<usize>,
    /// number of public input
    // public input is only 1 column and is implicitly the first witness column.
    // this size must not exceed number of total constraints.
    // Beware that public input must be wired to regular gates. If public input
    // needs to be wired to lookup gates an no-op regular gate is necessary
    pub num_pub_input: usize,
    /// customized gate function
    pub gate_func: CustomizedGates,
}

impl HyperPlonkParams {
    /// Number of variables in a multilinear system
    pub fn num_variables(&self) -> usize {
        log2(self.num_constraints) as usize
    }

    pub fn max_num_variables<E: Pairing, PCS: PolynomialCommitmentScheme<E>, Lookup: HyperPlonkLookupPlugin<E, PCS>>(&self) -> usize {
        max(log2(max(self.num_constraints, *self.num_lookup_constraints.iter().max().unwrap_or(&0usize))) as usize, Lookup::max_num_variables())
    }

    /// number of selector columns
    pub fn num_selector_columns(&self) -> usize {
        self.gate_func.num_selector_columns()
    }

    /// number of witness columns
    pub fn num_witness_columns<E: Pairing, PCS: PolynomialCommitmentScheme<E>, Lookup: HyperPlonkLookupPlugin<E, PCS>>(&self) -> usize {
        let mut sum = self.gate_func.num_witness_columns();
        for (&num_constraints, &num_witnesses) in zip(self.num_lookup_constraints.iter(), Lookup::num_witness_columns().iter()) {
            if num_constraints != 0 {
                sum += num_witnesses;
            }
        }
        sum
    }

    /// evaluate the identical polynomial
    pub fn eval_id_oracle<E: Pairing, PCS: PolynomialCommitmentScheme<E>, Lookup: HyperPlonkLookupPlugin<E, PCS>>(&self, point: &[E::ScalarField]) -> Result<E::ScalarField, HyperPlonkErrors> {
        let len = self.num_variables() + (log2(self.num_witness_columns::<E, PCS, Lookup>()) as usize);
        if point.len() != len {
            return Err(HyperPlonkErrors::InvalidParameters(format!(
                "ID oracle point length = {}, expected {}",
                point.len(),
                len,
            )));
        }

        let mut res = E::ScalarField::zero();
        let mut base = E::ScalarField::one();
        for &v in point.iter() {
            res += base * v;
            base += base;
        }
        Ok(res)
    }
}

/// The HyperPlonk index, consists of the following:
///   - HyperPlonk parameters
///   - the wire permutation
///   - the selector vectors
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HyperPlonkIndex<F: PrimeField> {
    pub params: HyperPlonkParams,
    pub permutation: Vec<F>,
    pub selectors: Vec<SelectorColumn<F>>,
}

impl<F: PrimeField> HyperPlonkIndex<F> {
    /// Number of variables in a multilinear system
    pub fn num_variables(&self) -> usize {
        self.params.num_variables()
    }

    pub fn max_num_variables<E: Pairing, PCS: PolynomialCommitmentScheme<E>, Lookup: HyperPlonkLookupPlugin<E, PCS>>(&self) -> usize {
        self.params.max_num_variables::<E, PCS, Lookup>()
    }

    /// number of selector columns
    pub fn num_selector_columns(&self) -> usize {
        self.params.num_selector_columns()
    }

    /// number of witness columns
    pub fn num_witness_columns<E: Pairing, PCS: PolynomialCommitmentScheme<E>, Lookup: HyperPlonkLookupPlugin<E, PCS>>(&self) -> usize {
        self.params.num_witness_columns::<E, PCS, Lookup>()
    }
}

/// The HyperPlonk proving key, consists of the following:
///   - the hyperplonk instance parameters
///   - the preprocessed polynomials output by the indexer
///   - the commitment to the selectors and permutations
///   - the parameters for polynomial commitment
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HyperPlonkProvingKey<E: Pairing, PCS: PolynomialCommitmentScheme<E>, Lookup: HyperPlonkLookupPlugin<E, PCS>> {
    /// Hyperplonk instance parameters
    pub params: HyperPlonkParams,
    /// The preprocessed permutation polynomials
    pub permutation_oracles: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>,
    /// The preprocessed selector polynomials
    pub selector_oracles: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>,
    /// Commitments to the preprocessed selector polynomials
    pub selector_commitments: Vec<PCS::Commitment>,
    /// Commitments to the preprocessed permutation polynomials
    pub permutation_commitments: Vec<PCS::Commitment>,

    pub selector_advices: Vec<PCS::ProverCommitmentAdvice>,
    pub permutation_advices: Vec<PCS::ProverCommitmentAdvice>,
    
    /// The parameters for PCS commitment
    pub pcs_param: PCS::ProverParam,

    pub lookup_preprocessing: Lookup::Preprocessing,
}

/// The HyperPlonk verifying key, consists of the following:
///   - the hyperplonk instance parameters
///   - the commitments to the preprocessed polynomials output by the indexer
///   - the parameters for polynomial commitment
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HyperPlonkVerifyingKey<E: Pairing, PCS: PolynomialCommitmentScheme<E>, Lookup: HyperPlonkLookupPlugin<E, PCS>> {
    /// Hyperplonk instance parameters
    pub params: HyperPlonkParams,
    /// The parameters for PCS commitment
    pub pcs_param: PCS::VerifierParam,
    /// A commitment to the preprocessed selector polynomials
    pub selector_commitments: Vec<PCS::Commitment>,
    /// Permutation oracles' commitments
    pub perm_commitments: Vec<PCS::Commitment>,

    pub lookup_preprocessing: Lookup::Preprocessing,

    pub num_party_vars: usize,
}

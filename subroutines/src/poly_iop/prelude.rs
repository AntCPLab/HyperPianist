// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

pub use crate::poly_iop::{
    errors::PolyIOPErrors,
    lookup::{instruction, instruction::JoltInstruction, LookupCheck, LookupCheckProof},
    perm_check::PermutationCheck,
    // prod_check::ProductCheck,
    structs::IOPProof,
    sum_check::SumCheck,
    sum_check::generic_sumcheck::{SumcheckInstanceProof, ZerocheckInstanceProof},
    utils::*,
    zero_check::ZeroCheck,
    rational_sumcheck::layered_circuit::{BatchedRationalSum, BatchedRationalSumProof, BatchedDenseRationalSum, BatchedSparseRationalSum},
    multi_rational_sumcheck::{MultiRationalSumcheck, MultiRationalSumcheckProof},
    combined_check::CombinedCheck,
    PolyIOP,
};

#[cfg(feature = "rational_sumcheck_piop")]
pub use crate::poly_iop::rational_sumcheck::{RationalSumcheckSlow, RationalSumcheckProof};

#[cfg(not(feature = "rational_sumcheck_piop"))]
pub use crate::poly_iop::perm_check::{BatchedDenseGrandProduct, BatchedGrandProduct, BatchedGrandProductProof};

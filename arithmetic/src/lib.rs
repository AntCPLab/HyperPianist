// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

mod errors;
mod fraction;
mod gaussian_elimination;
mod multilinear_polynomial;
mod optimized_mul;
mod univariate_polynomial;
mod util;
mod virtual_polynomial;

pub mod eq_poly;
pub mod math;
pub mod unipoly;

pub use fraction::Fraction;
pub use optimized_mul::OptimizedMul;
pub use errors::ArithErrors;
pub use multilinear_polynomial::{
    bind_poly_var_bot, bind_poly_var_top, bind_poly_var_bot_par,
    evaluate_no_par, evaluate_opt, fix_last_variables, fix_last_variables_no_par, fix_variables,
    identity_permutation, identity_permutation_mle, identity_permutation_mles, merge_polynomials, random_mle_list,
    random_permutation, random_permutation_raw, random_permutation_u64, random_permutation_mles, random_zero_mle_list, DenseMultilinearExtension,
};
pub use univariate_polynomial::{build_l, get_uni_domain};
pub use util::{bit_decompose, gen_eval_point, get_batched_nv, get_index, products_except_self, unsafe_allocate_zero_vec};
pub use virtual_polynomial::{
    build_eq_x_r, build_eq_x_r_vec, build_eq_x_r_with_coeff, build_eq_x_r_vec_with_coeff, build_eq_table, eq_eval, VPAuxInfo, VirtualPolynomial,
};
pub use unipoly::interpolate_uni_poly;

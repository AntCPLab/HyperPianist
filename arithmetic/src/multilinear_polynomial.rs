// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

use crate::{unsafe_allocate_zero_vec, util::get_batched_nv, ArithErrors};
use ark_ff::{Field, PrimeField};
use ark_poly::MultilinearExtension;
use ark_std::{end_timer, rand::RngCore, start_timer};
#[cfg(feature = "parallel")]
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator, ParallelSliceMut,
};
use std::sync::Arc;

pub use ark_poly::DenseMultilinearExtension;

pub fn bind_poly_var_top<F: PrimeField>(poly: &mut DenseMultilinearExtension<F>, r: &F) {
    let n = poly.evaluations.len() / 2;
    let (left, right) = poly.evaluations.split_at_mut(n);

    left.iter_mut().zip(right.iter()).for_each(|(a, b)| {
        *a += *r * (*b - *a);
    });

    poly.num_vars -= 1;
    poly.evaluations.truncate(n);
}

pub fn bind_poly_var_bot<F: PrimeField>(poly: &mut DenseMultilinearExtension<F>, r: &F) {
    let n = poly.evaluations.len() / 2;
    for i in 0..n {
        poly.evaluations[i] =
            poly.evaluations[2 * i] + *r * (poly.evaluations[2 * i + 1] - poly.evaluations[2 * i]);
    }
    poly.num_vars -= 1;
    poly.evaluations.truncate(n);
}

pub fn bind_poly_var_bot_par<F: PrimeField>(poly: &mut DenseMultilinearExtension<F>, r: &F, concurrency: usize) {
    let n = poly.evaluations.len() / 2;
    let mut chunk_size = (poly.evaluations.len() + concurrency - 1) / concurrency;
    if chunk_size == 0 || chunk_size % 2 == 1 {
        chunk_size += 1;
    }
    let num_chunks = (poly.evaluations.len() + chunk_size - 1) / chunk_size;
    poly.evaluations
        .par_chunks_mut(chunk_size)
        .for_each(|chunk| {
            for i in 0..chunk.len() / 2 {
                chunk[i] = chunk[2 * i] + *r * (chunk[2 * i + 1] - chunk[2 * i]);
            }
        });
    for i in 1..num_chunks {
        let src_start = i * chunk_size;
        let dst_start = (i * chunk_size) / 2;
        let size = (std::cmp::min((i + 1) * chunk_size, poly.evaluations.len()) - src_start) / 2;
        unsafe {
            let data = poly.evaluations.as_mut_ptr();
            std::ptr::copy_nonoverlapping(data.add(src_start), data.add(dst_start), size);
        }
    }
    poly.num_vars -= 1;
    poly.evaluations.truncate(n);
}

/// Sample a random list of multilinear polynomials.
/// Returns
/// - the list of polynomials,
/// - its sum of polynomial evaluations over the boolean hypercube.
pub fn random_mle_list<F: PrimeField, R: RngCore>(
    nv: usize,
    degree: usize,
    rng: &mut R,
) -> (Vec<Arc<DenseMultilinearExtension<F>>>, F) {
    let start = start_timer!(|| "sample random mle list");
    let mut multiplicands = Vec::with_capacity(degree);
    for _ in 0..degree {
        multiplicands.push(Vec::with_capacity(1 << nv))
    }
    let mut sum = F::zero();

    for _ in 0..(1 << nv) {
        let mut product = F::one();

        for e in multiplicands.iter_mut() {
            let val = F::rand(rng);
            e.push(val);
            product *= val;
        }
        sum += product;
    }

    let list = multiplicands
        .into_iter()
        .map(|x| Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, x)))
        .collect();

    end_timer!(start);
    (list, sum)
}

// Build a randomize list of mle-s whose sum is zero.
pub fn random_zero_mle_list<F: PrimeField, R: RngCore>(
    nv: usize,
    degree: usize,
    rng: &mut R,
) -> Vec<Arc<DenseMultilinearExtension<F>>> {
    let start = start_timer!(|| "sample random zero mle list");

    let mut multiplicands = Vec::with_capacity(degree);
    for _ in 0..degree {
        multiplicands.push(Vec::with_capacity(1 << nv))
    }
    for _ in 0..(1 << nv) {
        let zero_pos = (rng.next_u32() as usize) % multiplicands.len();
        for (i, e) in multiplicands.iter_mut().enumerate() {
            if i == zero_pos {
                e.push(F::zero());
            } else {
                e.push(F::rand(rng));
            }
        }
    }

    let list = multiplicands
        .into_iter()
        .map(|x| Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, x)))
        .collect();

    end_timer!(start);
    list
}

pub fn identity_permutation<F: PrimeField>(num_vars: usize, num_chunks: usize) -> Vec<F> {
    let len = (num_chunks as u64) * (1u64 << num_vars);
    (0..len).map(F::from).collect()
}

pub fn identity_permutation_mle<F: PrimeField>(
    shift: u64,
    num_vars: usize,
) -> Arc<DenseMultilinearExtension<F>> {
    let s_id_vec = (shift..shift + (1u64 << num_vars)).map(F::from).collect();
    Arc::new(DenseMultilinearExtension::from_evaluations_vec(
        num_vars, s_id_vec,
    ))
}

/// A list of MLEs that represents an identity permutation
pub fn identity_permutation_mles<F: PrimeField>(
    num_vars: usize,
    num_chunks: usize,
) -> Vec<Arc<DenseMultilinearExtension<F>>> {
    let mut res = vec![];
    for i in 0..num_chunks {
        let shift = (i * (1 << num_vars)) as u64;
        res.push(identity_permutation_mle(shift, num_vars));
    }
    res
}

pub fn random_permutation_u64<R: RngCore>(len: usize, rng: &mut R) -> Vec<u64> {
    let mut s_id_vec: Vec<u64> = (0..len as u64).into_iter().collect();
    let mut s_perm_vec = vec![];
    for _ in 0..len {
        let index = rng.next_u64() as usize % s_id_vec.len();
        s_perm_vec.push(s_id_vec.remove(index));
    }
    s_perm_vec
}

pub fn random_permutation_raw<F: PrimeField, R: RngCore>(len: u64, rng: &mut R) -> Vec<F> {
    let mut s_id_vec: Vec<F> = (0..len).map(F::from).collect();
    let mut s_perm_vec = vec![];
    for _ in 0..len {
        let index = rng.next_u64() as usize % s_id_vec.len();
        s_perm_vec.push(s_id_vec.remove(index));
    }
    s_perm_vec
}

pub fn random_permutation<F: PrimeField, R: RngCore>(
    num_vars: usize,
    num_chunks: usize,
    rng: &mut R,
) -> Vec<F> {
    let len = (num_chunks as u64) * (1u64 << num_vars);
    let mut s_id_vec: Vec<F> = (0..len).map(F::from).collect();
    let mut s_perm_vec = vec![];
    for _ in 0..len {
        let index = rng.next_u64() as usize % s_id_vec.len();
        s_perm_vec.push(s_id_vec.remove(index));
    }
    s_perm_vec
}

/// A list of MLEs that represent a random permutation
pub fn random_permutation_mles<F: PrimeField, R: RngCore>(
    num_vars: usize,
    num_chunks: usize,
    rng: &mut R,
) -> Vec<Arc<DenseMultilinearExtension<F>>> {
    let s_perm_vec = random_permutation(num_vars, num_chunks, rng);
    let mut res = vec![];
    let n = 1 << num_vars;
    for i in 0..num_chunks {
        res.push(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            s_perm_vec[i * n..i * n + n].to_vec(),
        )));
    }
    res
}

pub fn evaluate_opt<F: PrimeField>(poly: &DenseMultilinearExtension<F>, point: &[F]) -> F {
    assert_eq!(poly.num_vars, point.len());
    fix_variables(poly, point).evaluations[0]
}

pub fn fix_variables<F: PrimeField>(
    poly: &DenseMultilinearExtension<F>,
    partial_point: &[F],
) -> DenseMultilinearExtension<F> {
    assert!(
        partial_point.len() <= poly.num_vars,
        "invalid size of partial point"
    );
    let nv = poly.num_vars;
    let mut poly = poly.evaluations.to_vec();
    let dim = partial_point.len();
    // evaluate single variable of partial point from left to right
    for (i, point) in partial_point.iter().enumerate().take(dim) {
        poly = fix_one_variable_helper(&poly, nv - i, point);
    }

    DenseMultilinearExtension::<F>::from_evaluations_slice(nv - dim, &poly[..(1 << (nv - dim))])
}

fn fix_one_variable_helper<F: PrimeField>(data: &[F], nv: usize, point: &F) -> Vec<F> {
    let mut res = unsafe_allocate_zero_vec::<F>(1 << (nv - 1));

    // evaluate single variable of partial point from left to right
    #[cfg(not(feature = "parallel"))]
    for i in 0..(1 << (nv - 1)) {
        res[i] = data[i] + (data[(i << 1) + 1] - data[i << 1]) * point;
    }

    #[cfg(feature = "parallel")]
    res.par_iter_mut().enumerate().for_each(|(i, x)| {
        *x = data[i << 1] + (data[(i << 1) + 1] - data[i << 1]) * point;
    });

    res
}

pub fn evaluate_no_par<F: Field>(poly: &DenseMultilinearExtension<F>, point: &[F]) -> F {
    assert_eq!(poly.num_vars, point.len());
    fix_variables_no_par(poly, point).evaluations[0]
}

fn fix_variables_no_par<F: Field>(
    poly: &DenseMultilinearExtension<F>,
    partial_point: &[F],
) -> DenseMultilinearExtension<F> {
    assert!(
        partial_point.len() <= poly.num_vars,
        "invalid size of partial point"
    );
    let nv = poly.num_vars;
    let mut poly = poly.evaluations.to_vec();
    let dim = partial_point.len();
    // evaluate single variable of partial point from left to right
    for i in 1..dim + 1 {
        let r = partial_point[i - 1];
        for b in 0..(1 << (nv - i)) {
            poly[b] = poly[b << 1] + (poly[(b << 1) + 1] - poly[b << 1]) * r;
        }
    }
    DenseMultilinearExtension::from_evaluations_slice(nv - dim, &poly[..(1 << (nv - dim))])
}

/// merge a set of polynomials. Returns an error if the
/// polynomials do not share a same number of nvs.
pub fn merge_polynomials<F: PrimeField>(
    polynomials: &[Arc<DenseMultilinearExtension<F>>],
) -> Result<Arc<DenseMultilinearExtension<F>>, ArithErrors> {
    let nv = polynomials[0].num_vars();
    for poly in polynomials.iter() {
        if nv != poly.num_vars() {
            return Err(ArithErrors::InvalidParameters(
                "num_vars do not match for polynomials".to_string(),
            ));
        }
    }

    let merged_nv = get_batched_nv(nv, polynomials.len());
    let mut scalars = vec![];
    for poly in polynomials.iter() {
        scalars.extend_from_slice(poly.to_evaluations().as_slice());
    }
    scalars.extend_from_slice(
        unsafe_allocate_zero_vec::<F>((1 << merged_nv) - scalars.len()).as_ref(),
    );
    Ok(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
        merged_nv, scalars,
    )))
}

pub fn fix_last_variables_no_par<F: PrimeField>(
    poly: &DenseMultilinearExtension<F>,
    partial_point: &[F],
) -> DenseMultilinearExtension<F> {
    let mut res = fix_last_variable_no_par(poly, partial_point.last().unwrap());
    for p in partial_point.iter().rev().skip(1) {
        res = fix_last_variable_no_par(&res, p);
    }
    res
}

fn fix_last_variable_no_par<F: PrimeField>(
    poly: &DenseMultilinearExtension<F>,
    partial_point: &F,
) -> DenseMultilinearExtension<F> {
    let nv = poly.num_vars();
    let half_len = 1 << (nv - 1);
    let mut res = unsafe_allocate_zero_vec::<F>(half_len);
    for (i, e) in res.iter_mut().enumerate().take(half_len) {
        *e = poly.evaluations[i]
            + *partial_point * (poly.evaluations[i + half_len] - poly.evaluations[i]);
    }
    DenseMultilinearExtension::from_evaluations_vec(nv - 1, res)
}
pub fn fix_last_variables<F: PrimeField>(
    poly: &DenseMultilinearExtension<F>,
    partial_point: &[F],
) -> DenseMultilinearExtension<F> {
    assert!(
        partial_point.len() <= poly.num_vars,
        "invalid size of partial point"
    );
    let nv = poly.num_vars;
    let mut poly = poly.evaluations.to_vec();
    let dim = partial_point.len();
    // evaluate single variable of partial point from left to right
    for (i, point) in partial_point.iter().rev().enumerate().take(dim) {
        poly = fix_last_variable_helper(&poly, nv - i, point);
    }

    DenseMultilinearExtension::<F>::from_evaluations_slice(nv - dim, &poly[..(1 << (nv - dim))])
}

fn fix_last_variable_helper<F: PrimeField>(data: &[F], nv: usize, point: &F) -> Vec<F> {
    let half_len = 1 << (nv - 1);
    let mut res = unsafe_allocate_zero_vec::<F>(half_len);

    // evaluate single variable of partial point from left to right
    #[cfg(not(feature = "parallel"))]
    for b in 0..half_len {
        res[b] = data[b] + (data[b + half_len] - data[b]) * point;
    }

    #[cfg(feature = "parallel")]
    res.par_iter_mut().enumerate().for_each(|(i, x)| {
        *x = data[i] + (data[i + half_len] - data[i]) * point;
    });

    res
}

#[cfg(test)]
mod tests{
    use super::{bind_poly_var_bot, bind_poly_var_bot_par};
    use ark_bls12_381::Fr;
    use ark_std::{test_rng, UniformRand};
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};

    #[test]
    fn test_bind_poly_var_bot() {
        let mut rng = test_rng();

        let nv = 10;
        let mut poly = DenseMultilinearExtension::<Fr>::rand(nv, &mut rng);
        let mut poly_2 = poly.clone();
        let point: Vec<_> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();
        bind_poly_var_bot(&mut poly, &point[0]);
        bind_poly_var_bot_par(&mut poly_2, &point[0], rayon::current_num_threads() / 2);
        assert_eq!(poly.evaluations, poly_2.evaluations);
    }
}

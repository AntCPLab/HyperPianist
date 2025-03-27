use ark_ff::PrimeField;
use core::ops::{Mul, MulAssign, Sub, SubAssign};
use num_traits::One;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

const MIN_PARALLEL_LEN: usize = 16; // The minimum size for which we should actually parallelize the compute.

/// Compute the evaluations of the columns of the matrix M that is derived from
/// `a`.
pub fn compute_v_vec<F: PrimeField>(a: &[F], L_vec: &[F], num: usize) -> Vec<F> {
    a.chunks(1 << num)
        .zip(L_vec.iter())
        .fold(unsafe_allocate_zero_vec::<F>(1 << num), |mut v, (row, l)| {
            v.iter_mut().zip(row).for_each(|(v, a)| *v += *l * a);
            v
        })
}

/// This method manipulates left and right such that
/// right[i] = left[i] * p and left[i] = left[i] * (1 - p)
fn compute_evaluation_vector_impl<F>(left: &mut [F], right: &mut [F], p: F)
where
    F: One + Sub<Output = F> + MulAssign + SubAssign + Mul<Output = F> + Send + Sync + Copy,
{
    let k = std::cmp::min(left.len(), right.len());
    let one_minus_p = F::one() - p;
    left.par_iter_mut()
        .with_min_len(MIN_PARALLEL_LEN)
        .zip(right.par_iter_mut())
        .for_each(|(li, ri)| {
            *ri = *li * p;
            *li -= *ri;
        });
    left[k..]
        .par_iter_mut()
        .with_min_len(MIN_PARALLEL_LEN)
        .for_each(|li| {
            *li *= one_minus_p;
        });
}

/// Given a point of evaluation, computes the vector that allows us
/// to evaluate a multilinear extension as an inner product.
pub fn compute_evaluation_vector<F>(v: &mut [F], point: &[F])
where
    F: One + Sub<Output = F> + MulAssign + SubAssign + Mul<Output = F> + Send + Sync + Copy,
{
    assert!(v.len() <= (1 << point.len()));
    if point.is_empty() || v.is_empty() {
        // v is guaranteed to be at most length 1 by the assert!.
        v.fill(F::one());
        return;
    }
    v[0] = F::one() - point[0];
    if v.len() > 1 {
        v[1] = point[0];
    }
    for (level, p) in point[1..].iter().enumerate() {
        let mid = 1 << (level + 1);
        let (left, right): (&mut [F], &mut [F]) = if mid >= v.len() {
            (v, &mut [])
        } else {
            v.split_at_mut(mid)
        };
        compute_evaluation_vector_impl(left, right, *p);
    }
}

macro_rules! par_join_5 {
    ($task1:expr, $task2:expr, $task3:expr, $task4:expr, $task5:expr) => {{
        let ((result1, result2), ((result3, result4), result5)) = rayon::join(
            || rayon::join($task1, $task2),
            || rayon::join(|| rayon::join($task3, $task4), $task5),
        );
        (result1, result2, result3, result4, result5)
    }};
}

pub fn unsafe_allocate_zero_vec<F: PrimeField + Sized>(size: usize) -> Vec<F> {
    // https://stackoverflow.com/questions/59314686/how-to-efficiently-create-a-large-vector-of-items-initialized-to-the-same-value

    // Check for safety of 0 allocation
    #[cfg(debug_assertions)]
    unsafe {
        let value = &F::zero();
        let ptr = value as *const F as *const u8;
        let bytes = std::slice::from_raw_parts(ptr, std::mem::size_of::<F>());
        assert!(bytes.iter().all(|&byte| byte == 0));
    }

    // Bulk allocate zeros, unsafely
    let result: Vec<F>;
    unsafe {
        let layout = std::alloc::Layout::array::<F>(size).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut F;

        if ptr.is_null() {
            panic!("Zero vec allocaiton failed");
        }

        result = Vec::from_raw_parts(ptr, size, size);
    }
    result
}

use ark_ff::{prelude::*, PrimeField};
use ark_std::{borrow::Borrow, iterable::Iterable, vec::Vec};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub mod stream_pippenger;
pub use stream_pippenger::*;

use super::ScalarMul;

pub trait VariableBaseMSM: ScalarMul {
    /// Computes an inner product between the [`PrimeField`] elements in `scalars`
    /// and the corresponding group elements in `bases`.
    ///
    /// If the elements have different length, it will chop the slices to the
    /// shortest length between `scalars.len()` and `bases.len()`.
    ///
    /// Reference: [`VariableBaseMSM::msm`]
    fn msm_unchecked(bases: &[Self::MulBase], scalars: &[Self::ScalarField]) -> Self {
        let mut bigints = cfg_into_iter!(scalars)
            .map(|s| s.into_bigint())
            .collect::<Vec<_>>();
        if Self::NEGATION_IS_CHEAP {
            let num_bits = Self::ScalarField::MODULUS_BIT_SIZE as usize;
            let size = ark_std::cmp::min(bases.len(), bigints.len());
            let c = compute_c(size, num_bits);
            process_digits(&mut bigints, c, num_bits);

            msm_bigint_wnaf_body(bases, &mut bigints, c)
        } else {
            msm_bigint(bases, &bigints)
        }
    }

    #[cfg(not(feature = "parallel"))]
    fn msm_unchecked_par(
        bases: &[Self::MulBase],
        scalars: &[Self::ScalarField],
        _num_tasks: usize,
    ) -> Self {
        Self::msm_unchecked(bases, scalars)
    }

    #[cfg(feature = "parallel")]
    fn msm_unchecked_par(
        bases: &[Self::MulBase],
        scalars: &[Self::ScalarField],
        num_tasks: usize,
    ) -> Self {
        let num_bits = Self::ScalarField::MODULUS_BIT_SIZE as usize;
        let size = ark_std::cmp::min(bases.len(), scalars.len());
        let c = compute_c(size, num_bits);
        let digits_count = (num_bits + c - 1) / c;
        if digits_count >= num_tasks {
            return Self::msm_unchecked(bases, scalars);
        }

        let mut bigints = cfg_into_iter!(scalars)
            .map(|s| s.into_bigint())
            .collect::<Vec<_>>();
        process_digits(&mut bigints, c, num_bits);

        let num_chunks = (num_tasks + digits_count - 1) / digits_count;
        let chunk_size = (size + num_chunks - 1) / num_chunks;

        let (sender, receiver) = std::sync::mpsc::sync_channel(num_chunks);
        let mut sum = Self::zero();

        // The original code uses rayon. Unfortunately, experiments have shown that
        // rayon does quite sub-optimally for this particular instance, and directly
        // spawning threads was faster.
        rayon::scope(|s| {
            let mut iter_base = bases.chunks(chunk_size);
            let mut iter_bigint = bigints.chunks_mut(chunk_size);
            let sender = &sender;
            for i in 0..num_chunks {
                let base = iter_base.next().unwrap();
                let bigints = iter_bigint.next().unwrap();
                if i == num_chunks - 1 {
                    sender.send(msm_bigint_wnaf_body::<Self>(base, bigints, c)).unwrap();
                } else {
                    s.spawn(move |_| {
                        sender.send(msm_bigint_wnaf_body::<Self>(base, bigints, c)).unwrap();
                    });
                }
            }
        });
        for i in 0..num_chunks {
            sum += receiver.recv().unwrap();
        }

        sum
    }

    fn msm_unchecked_par_auto(bases: &[Self::MulBase], scalars: &[Self::ScalarField]) -> Self {
        #[cfg(feature = "parallel")]
        let num_tasks = std::thread::available_parallelism().unwrap().get();
        #[cfg(not(feature = "parallel"))]
        let num_tasks = 1;

        Self::msm_unchecked_par(bases, scalars, num_tasks)
    }

    /// Performs multi-scalar multiplication, without checking that `bases.len() == scalars.len()`.
    ///
    /// # Warning
    ///
    /// This method checks that `bases` and `scalars` have the same length.
    /// If they are unequal, it returns an error containing
    /// the shortest length over which the MSM can be performed.
    fn msm(bases: &[Self::MulBase], scalars: &[Self::ScalarField]) -> Result<Self, usize> {
        (bases.len() == scalars.len())
            .then(|| Self::msm_unchecked(bases, scalars))
            .ok_or(bases.len().min(scalars.len()))
    }

    /// Optimized implementation of multi-scalar multiplication.
    fn msm_bigint(
        bases: &[Self::MulBase],
        bigints: &[<Self::ScalarField as PrimeField>::BigInt],
    ) -> Self {
        let mut bigints = bigints.to_vec();
        if Self::NEGATION_IS_CHEAP {
            let num_bits = Self::ScalarField::MODULUS_BIT_SIZE as usize;
            let size = ark_std::cmp::min(bases.len(), bigints.len());
            let c = compute_c(size, num_bits);
            process_digits(&mut bigints, c, num_bits);

            msm_bigint_wnaf_body(bases, &mut bigints, c)
        } else {
            msm_bigint(bases, &bigints)
        }
    }

    /// Streaming multi-scalar multiplication algorithm with hard-coded chunk
    /// size.
    fn msm_chunks<I: ?Sized, J>(bases_stream: &J, scalars_stream: &I) -> Self
    where
        I: Iterable,
        I::Item: Borrow<Self::ScalarField>,
        J: Iterable,
        J::Item: Borrow<Self::MulBase>,
    {
        assert!(scalars_stream.len() <= bases_stream.len());

        // remove offset
        let bases_init = bases_stream.iter();
        let mut scalars = scalars_stream.iter();

        // align the streams
        // TODO: change `skip` to `advance_by` once rust-lang/rust#7774 is fixed.
        // See <https://github.com/rust-lang/rust/issues/77404>
        let mut bases = bases_init.skip(bases_stream.len() - scalars_stream.len());
        let step: usize = 1 << 20;
        let mut result = Self::zero();
        for _ in 0..(scalars_stream.len() + step - 1) / step {
            let bases_step = (&mut bases)
                .take(step)
                .map(|b| *b.borrow())
                .collect::<Vec<_>>();
            let scalars_step = (&mut scalars)
                .take(step)
                .map(|s| s.borrow().into_bigint())
                .collect::<Vec<_>>();
            result += Self::msm_bigint(bases_step.as_slice(), scalars_step.as_slice());
        }
        result
    }
}

fn compute_c(size: usize, _num_bits: usize) -> usize {
    if size < 32 {
        3
    } else {
        let c = (ark_std::cmp::max(ark_std::log2(size) - ark_std::log2(ark_std::log2(size) as usize), 4) - 1 ..= ark_std::log2(size))
            .map(|c| (c, (size + (1 << (c as usize))) / (c as usize)))
            .min_by_key(|(_, cost)| *cost)
            .unwrap()
            .0 as usize;
        // gnark did not implement c = 17 - 19. 16 is a great point as it's well aligned.
        if c > 16 {
            16
        } else {
            c
        }
    }
}

// Compute msm using windowed non-adjacent form
fn msm_bigint_wnaf_body<V: VariableBaseMSM>(
    bases: &[V::MulBase],
    bigints: &mut [<V::ScalarField as PrimeField>::BigInt],
    c: usize,
) -> V {
    let size = ark_std::cmp::min(bases.len(), bigints.len());
    let scalars = &mut bigints[..size];
    let bases = &bases[..size];

    let num_bits = V::ScalarField::MODULUS_BIT_SIZE as usize;
    let digits_count = (num_bits + c - 1) / c;

    let mut window_sums = vec![V::zero(); digits_count];

    let process_digit = |i: usize, out: &mut V| {
        let mut buckets = if i == digits_count - 1 {
            // No borrow for the last digit
            let final_size = num_bits - (digits_count - 1) * c;
            vec![V::zero(); 1 << final_size]
        } else {
            vec![V::zero(); 1 << (c - 1)]
        };
        let bit_offset = i * c;
        let u64_idx = bit_offset / 64;
        let bit_idx = bit_offset % 64;

        let is_multi_word = bit_idx > 64 - c && i != digits_count - 1;
        let window_mask = (1 << c) - 1;
        let sign_mask = 1 << (c - 1);

        for (scalar, base) in scalars.iter().zip(bases) {
            let scalar = scalar.as_ref();

            if i == digits_count - 1 {
                let coef = scalar[u64_idx] >> bit_idx;
                if coef != 0 {
                    buckets[(coef - 1) as usize] += base;
                }
                continue;
            }

            let bit_buf = if is_multi_word {
                // Combine the current u64's bits with the bits from the next u64
                (scalar[u64_idx] >> bit_idx) | (scalar[1 + u64_idx] << (64 - bit_idx))
            } else {
                // This window's bits are contained in a single u64,
                // or it's the last u64 anyway.
                scalar[u64_idx] >> bit_idx
            };
            let coef = bit_buf & window_mask;

            if coef == 0 {
                continue;
            }

            if coef & sign_mask == 0 {
                buckets[(coef - 1) as usize] += base;
            } else {
                buckets[(coef & (!sign_mask)) as usize] -= base;
            }
        }

        let mut running_sum = V::zero();
        *out = V::zero();
        buckets.into_iter().rev().for_each(|b| {
            running_sum += &b;
            *out += &running_sum;
        });
    };

    // The original code uses rayon. Unfortunately, experiments have shown that
    // rayon does quite sub-optimally for this particular instance, and directly
    // spawning threads was faster.
    #[cfg(feature = "parallel")]
    rayon::scope(|s| {
        let len = window_sums.len();
        for (i, out) in window_sums.iter_mut().enumerate() {
            if i == len - 1 {
                process_digit(i, out);
            } else {
                s.spawn(move |_| {
                    process_digit(i, out);
                });
            }
        }
    });

    #[cfg(not(feature = "parallel"))]
    for (i, out) in window_sums.iter_mut().enumerate() {
        process_digit(i, out);
    }

    // We store the sum for the highest window.
    let mut total = *window_sums.last().unwrap();
    for i in (0..(window_sums.len() - 1)).rev() {
        for _ in 0..c {
            total.double_in_place();
        }
        total += &window_sums[i];
    }

    total
}

/// Optimized implementation of multi-scalar multiplication.
fn msm_bigint<V: VariableBaseMSM>(
    bases: &[V::MulBase],
    bigints: &[<V::ScalarField as PrimeField>::BigInt],
) -> V {
    let size = ark_std::cmp::min(bases.len(), bigints.len());
    let scalars = &bigints[..size];
    let bases = &bases[..size];
    let scalars_and_bases_iter = scalars.iter().zip(bases).filter(|(s, _)| !s.is_zero());

    let num_bits = V::ScalarField::MODULUS_BIT_SIZE as usize;
    let c = compute_c(size, num_bits);
    let one = V::ScalarField::one().into_bigint();

    let zero = V::zero();
    let window_starts: Vec<_> = (0..num_bits).step_by(c).collect();

    // Each window is of size `c`.
    // We divide up the bits 0..num_bits into windows of size `c`, and
    // in parallel process each such window.
    let window_sums: Vec<_> = ark_std::cfg_into_iter!(window_starts)
        .map(|w_start| {
            let mut res = zero;
            // We don't need the "zero" bucket, so we only have 2^c - 1 buckets.
            let mut buckets = vec![zero; (1 << c) - 1];
            // This clone is cheap, because the iterator contains just a
            // pointer and an index into the original vectors.
            scalars_and_bases_iter.clone().for_each(|(&scalar, base)| {
                if scalar == one {
                    // We only process unit scalars once in the first window.
                    if w_start == 0 {
                        res += base;
                    }
                } else {
                    let mut scalar = scalar;

                    // We right-shift by w_start, thus getting rid of the
                    // lower bits.
                    scalar.divn(w_start as u32);

                    // We mod the remaining bits by 2^{window size}, thus taking `c` bits.
                    let scalar = scalar.as_ref()[0] % (1 << c);

                    // If the scalar is non-zero, we update the corresponding
                    // bucket.
                    // (Recall that `buckets` doesn't have a zero bucket.)
                    if scalar != 0 {
                        buckets[(scalar - 1) as usize] += base;
                    }
                }
            });

            // Compute sum_{i in 0..num_buckets} (sum_{j in i..num_buckets} bucket[j])
            // This is computed below for b buckets, using 2b curve additions.
            //
            // We could first normalize `buckets` and then use mixed-addition
            // here, but that's slower for the kinds of groups we care about
            // (Short Weierstrass curves and Twisted Edwards curves).
            // In the case of Short Weierstrass curves,
            // mixed addition saves ~4 field multiplications per addition.
            // However normalization (with the inversion batched) takes ~6
            // field multiplications per element,
            // hence batch normalization is a slowdown.

            // `running_sum` = sum_{j in i..num_buckets} bucket[j],
            // where we iterate backward from i = num_buckets to 0.
            let mut running_sum = V::zero();
            buckets.into_iter().rev().for_each(|b| {
                running_sum += &b;
                res += &running_sum;
            });
            res
        })
        .collect();

    // We store the sum for the lowest window.
    let mut total = *window_sums.last().unwrap();
    for i in (0..(window_sums.len() - 1)).rev() {
        for _ in 0..c {
            total.double_in_place();
        }
        total += &window_sums[i];
    }

    total
}

fn process_digits<BigInt: BigInteger>(a: &mut [BigInt], w: usize, num_bits: usize) {
    let radix: u64 = 1 << w;
    let sign_mask: u64 = 1 << (w - 1);
    let window_mask: u64 = radix - 1;
    let digits_count = (num_bits + w - 1) / w;

    ark_std::cfg_iter_mut!(a).for_each(|scalar| {
        let scalar = scalar.as_mut();
        let mut carry = 0u64;
        for i in 0..digits_count {
            // Construct a buffer of bits of the scalar, starting at `bit_offset`.
            let bit_offset = i * w;
            let u64_idx = bit_offset / 64;
            let bit_idx = bit_offset % 64;

            // Read the bits from the scalar
            let is_multi_word = bit_idx > 64 - w && u64_idx != scalar.len() - 1;

            let bit_buf = if is_multi_word {
                // Combine the current u64's bits with the bits from the next u64
                (scalar[u64_idx] >> bit_idx) | (scalar[1 + u64_idx] << (64 - bit_idx))
            } else {
                scalar[u64_idx] >> bit_idx
            };

            // Read the actual coefficient value from the window
            let coef = carry + (bit_buf & window_mask); // coef = [0, 2^r)

            // Recenter coefficients from [0,2^w) to [-2^w/2, 2^w/2)
            carry = (coef + radix / 2) >> w;

            let val = if i == digits_count - 1 {
                // Cannot borrow anything for the last digit
                coef as i64
            } else {
                coef as i64 - (carry << w) as i64
            };

            let val = if val >= 0 {
                val as u64
            } else {
                (-val - 1) as u64 | sign_mask
            };

            let read_mask = window_mask << bit_idx;
            scalar[u64_idx] = (scalar[u64_idx] & (!read_mask)) | (val << bit_idx);
            if is_multi_word {
                let len = w - (64 - bit_idx);
                // Write to the bottom of the next word
                scalar[1 + u64_idx] = ((scalar[1 + u64_idx] >> len) << len)
                    | (val >> (64 - bit_idx));
            }
        }
    });
}

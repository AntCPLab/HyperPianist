use ark_ff::PrimeField;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use std::ops::{Add, Sub, Mul};
use crate::OptimizedMul;

#[derive(CanonicalSerialize, CanonicalDeserialize, Copy, Clone, Debug, PartialEq)]
pub struct Fraction<F: PrimeField> {
    pub p: F,
    pub q: F,
}

impl<F: PrimeField> Fraction<F> {
    pub fn zero() -> Self {
        Self {
            p: F::zero(),
            q: F::one(),
        }
    }

    pub fn rational_add(a: Fraction<F>, b: Fraction<F>) -> Fraction<F> {
        Fraction {
            p: a.p.mul_01_optimized(b.q) + a.q.mul_01_optimized(b.p),
            q: a.q.mul_1_optimized(b.q),
        }
    }
}

impl<F: PrimeField> Add<Fraction<F>> for Fraction<F> {
    type Output = Fraction<F>;

    fn add(self, rhs: Fraction<F>) -> Self::Output {
        Fraction {
            p: self.p + rhs.p,
            q: self.q + rhs.q,
        }
    }
}

impl<F: PrimeField> Sub<Fraction<F>> for Fraction<F> {
    type Output = Fraction<F>;

    fn sub(self, rhs: Fraction<F>) -> Self::Output {
        Fraction {
            p: self.p - rhs.p,
            q: self.q - rhs.q,
        }
    }
}

impl<F: PrimeField> Mul<Fraction<F>> for Fraction<F> {
    type Output = Fraction<F>;

    fn mul(self, rhs: Fraction<F>) -> Self::Output {
        Fraction {
            p: self.p.mul_01_optimized(rhs.p),
            q: self.q.mul_1_optimized(rhs.q),
        }
    }
}

impl<F: PrimeField> Mul<F> for Fraction<F> {
    type Output = Fraction<F>;

    fn mul(self, rhs: F) -> Self::Output {
        Fraction {
            p: self.p.mul_01_optimized(rhs),
            q: self.q.mul_1_optimized(rhs),
        }
    }
}

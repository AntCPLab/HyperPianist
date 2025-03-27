use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use ark_std::{
    borrow::Borrow,
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    hash::{Hash, Hasher},
    io::{Read, Write},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    rand::{
        distributions::{Distribution, Standard},
        Rng,
    },
    vec::Vec,
    One, Zero,
};

use ark_ff::{fields::Field, PrimeField, ToConstraintField, UniformRand};

use zeroize::Zeroize;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{Affine, Projective, SWCurveConfig};
use crate::{
    scalar_mul::{variable_base::VariableBaseMSM, ScalarMul},
    AffineRepr, CurveGroup, Group,
};

/// Jacobian coordinates for a point on an elliptic curve in short Weierstrass
/// form, over the base field `P::BaseField`. This struct implements arithmetic
/// via the Jacobian formulae
#[derive(Derivative)]
#[derivative(Copy(bound = "P: SWCurveConfig"), Clone(bound = "P: SWCurveConfig"))]
#[must_use]
pub struct ExtendedJacobian<P: SWCurveConfig> {
    pub x: P::BaseField,
    pub y: P::BaseField,
    pub zz: P::BaseField,
    pub zzz: P::BaseField,
}

impl<P: SWCurveConfig> Display for ExtendedJacobian<P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", Affine::from(*self))
    }
}

impl<P: SWCurveConfig> Debug for ExtendedJacobian<P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self.is_zero() {
            true => write!(f, "infinity"),
            false => write!(f, "({}, {}, {}, {})", self.x, self.y, self.zz, self.zzz),
        }
    }
}

impl<P: SWCurveConfig> Eq for ExtendedJacobian<P> {}
impl<P: SWCurveConfig> PartialEq for ExtendedJacobian<P> {
    fn eq(&self, other: &Self) -> bool {
        if self.is_zero() {
            return other.is_zero();
        }

        if other.is_zero() {
            return false;
        }

        // The points (X, Y, Z) and (X', Y', Z')
        // are equal when (X * Z^2) = (X' * Z'^2)
        // and (Y * Z^3) = (Y' * Z'^3).

        if self.x * &other.zz != other.x * &self.zz {
            false
        } else {
            self.y * &other.zzz == other.y * &self.zzz
        }
    }
}

impl<P: SWCurveConfig> PartialEq<Affine<P>> for ExtendedJacobian<P> {
    fn eq(&self, other: &Affine<P>) -> bool {
        let other_jac : ExtendedJacobian<P> = (*other).into();
        *self == other_jac
    }
}

impl<P: SWCurveConfig> Hash for ExtendedJacobian<P> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let affine : Affine<P> = (*self).into();
        affine.hash(state)
    }
}

impl<P: SWCurveConfig> Distribution<ExtendedJacobian<P>> for Standard {
    /// Generates a uniformly random instance of the curve.
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> ExtendedJacobian<P> {
        loop {
            let x = P::BaseField::rand(rng);
            let greatest = rng.gen();

            if let Some(p) = Affine::get_point_from_x_unchecked(x, greatest) {
                return P::mul_affine_extended_jac(&p, P::COFACTOR)
            }
        }
    }
}

impl<P: SWCurveConfig> Default for ExtendedJacobian<P> {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

impl<P: SWCurveConfig> ExtendedJacobian<P> {
    /// Constructs a new group element without checking whether the coordinates
    /// specify a point in the subgroup.
    pub const fn new_unchecked(x: P::BaseField, y: P::BaseField, zz: P::BaseField, zzz: P::BaseField) -> Self {
        Self { x, y, zz, zzz }
    }

    /// Constructs a new group element in a way while enforcing that points are in
    /// the prime-order subgroup.
    pub fn new(x: P::BaseField, y: P::BaseField, zz: P::BaseField, zzz: P::BaseField) -> Self {
        let p : Affine<P> = Self::new_unchecked(x, y, zz, zzz).into();

        let mut z1 = zz;
        z1.square_in_place();
        z1 *= &zz;

        let mut z2 = zzz;
        z2.square_in_place();

        assert!(z1 == z2);

        assert!(p.is_on_curve());
        assert!(p.is_in_correct_subgroup_assuming_on_curve());
        p.into()
    }
}

impl<P: SWCurveConfig> Zeroize for ExtendedJacobian<P> {
    fn zeroize(&mut self) {
        self.x.zeroize();
        self.y.zeroize();
        self.zz.zeroize();
        self.zzz.zeroize();
    }
}

impl<P: SWCurveConfig> Zero for ExtendedJacobian<P> {
    /// Returns the point at infinity, which always has Z = 0.
    #[inline]
    fn zero() -> Self {
        Self::new_unchecked(
            P::BaseField::one(),
            P::BaseField::one(),
            P::BaseField::zero(),
            P::BaseField::zero(),
        )
    }

    /// Checks whether `self.z.is_zero()`.
    #[inline]
    fn is_zero(&self) -> bool {
        self.zz == P::BaseField::ZERO
    }
}

impl<P: SWCurveConfig> Group for ExtendedJacobian<P> {
    type ScalarField = P::ScalarField;

    #[inline]
    fn generator() -> Self {
        Affine::generator().into()
    }

    /// Sets `self = 2 * self`. Note that Jacobian formulae are incomplete, and
    /// so doubling cannot be computed as `self + self`. Instead, this
    /// implementation uses the following specialized doubling formulae:
    /// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-dbl-2008-s-1
    fn double_in_place(&mut self) -> &mut Self {
        if self.is_zero() {
            return self;
        }

        let mut u = self.y;
        u.double_in_place();

        let mut v = u;
        v.square_in_place();
    
        let mut w = u;
        w *= &v;
        
        let mut s = self.x;
        s *= &v;

        let mut xx = self.x;
        xx.square_in_place();

        let mut m = xx;
        m.double_in_place();
        m += &xx;
        if P::COEFF_A != P::BaseField::ZERO {
            m += &P::mul_by_a(self.zz.square());
        }

        self.x = m;
        self.x.square_in_place();
        self.x -= &s.double();

        self.zz *= &v;
        self.zzz *= &w;

        s -= &self.x;
        w.neg_in_place();
        self.y = P::BaseField::sum_of_products(&[m, self.y], &[s, w]);

        self
    }

    #[inline]
    fn mul_bigint(&self, other: impl AsRef<[u64]>) -> Self {
        P::mul_extended_jac(self, other.as_ref())
    }
}

impl<P: SWCurveConfig> ExtendedJacobian<P> {
    /// Normalizes a slice of ExtendedJacobian elements so that
    /// conversion to affine is cheap.
    ///
    /// In more detail, this method converts a curve point in Jacobian
    /// coordinates (x, y, z) into an equivalent representation (x/z^2,
    /// y/z^3, 1).
    ///
    /// For `N = v.len()`, this costs 1 inversion + 6N field multiplications + N
    /// field squarings.
    ///
    /// (Where batch inversion comprises 3N field multiplications + 1 inversion
    /// of these operations)
    #[inline]
    pub fn normalize_batch(v: &[Self]) -> Vec<Affine<P>> {
        let mut zz_s = v.iter().map(|g| g.zz).collect::<Vec<_>>();
        let mut zzz_s = v.iter().map(|g| g.zzz).collect::<Vec<_>>();
        ark_ff::batch_inversion(&mut zz_s);
        ark_ff::batch_inversion(&mut zzz_s);

        // Perform affine transformations
        ark_std::cfg_iter!(v)
            .zip(zz_s)
            .zip(zzz_s)
            .map(|((g, zz), zzz)| match g.is_zero() {
                true => Affine::identity(),
                false => {
                    let x = g.x * zz;
                    let y = g.y * zzz;
                    Affine::new_unchecked(x, y)
                },
            })
            .collect()
    }
}

impl<P: SWCurveConfig> Neg for ExtendedJacobian<P> {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self {
        self.y = -self.y;
        self
    }
}

impl<P: SWCurveConfig, T: Borrow<Affine<P>>> AddAssign<T> for ExtendedJacobian<P> {
    /// Using http://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-madd-2008-s
    fn add_assign(&mut self, other: T) {
        let other = other.borrow();
        if let Some((&other_x, &other_y)) = other.xy() {
            if self.is_zero() {
                self.x = other_x;
                self.y = other_y;
                self.zz = P::BaseField::one();
                self.zzz = P::BaseField::one();
                return;
            }

            let mut u2 = other_x;
            u2 *= &self.zz;
            let mut s2 = other_y;
            s2 *= &self.zzz;

            if self.x == u2 && self.y == s2 {
                // The two points are equal, so we double.
                self.double_in_place();
            } else {
                // If we're adding -a and a together, self.z becomes zero as H becomes zero.

                // P = U2-X1
                u2 -= &self.x;

                // PP = P^2
                let mut pp = u2;
                pp.square_in_place();

                let mut q = self.x;
                q *= &pp;

                self.zz *= &pp;

                // PPP
                pp *= &u2;

                // R = S2 - Y1
                s2 -= &self.y;

                // X3 = R2-PPP-2*Q
                self.x = s2;
                self.x.square_in_place();
                self.x -= &pp;
                self.x -= &q.double();

                
                self.zzz *= &pp;

                // Y3 = R*(Q-X3)-Y1*PPP
                q -= &self.x;
                self.y.neg_in_place();
                self.y = P::BaseField::sum_of_products(&[s2, pp], &[q, self.y]);
            }
        }
    }
}

impl<P: SWCurveConfig, T: Borrow<Affine<P>>> Add<T> for ExtendedJacobian<P> {
    type Output = Self;
    fn add(mut self, other: T) -> Self {
        let other = other.borrow();
        self += other;
        self
    }
}

impl<P: SWCurveConfig, T: Borrow<Affine<P>>> SubAssign<T> for ExtendedJacobian<P> {
    fn sub_assign(&mut self, other: T) {
        *self += -(*other.borrow());
    }
}

impl<P: SWCurveConfig, T: Borrow<Affine<P>>> Sub<T> for ExtendedJacobian<P> {
    type Output = Self;
    fn sub(mut self, other: T) -> Self {
        self -= other.borrow();
        self
    }
}

ark_ff::impl_additive_ops_from_ref!(ExtendedJacobian, SWCurveConfig);

impl<'a, P: SWCurveConfig> Add<&'a Self> for ExtendedJacobian<P> {
    type Output = Self;

    #[inline]
    fn add(mut self, other: &'a Self) -> Self {
        self += other;
        self
    }
}

impl<'a, P: SWCurveConfig> AddAssign<&'a Self> for ExtendedJacobian<P> {
    fn add_assign(&mut self, other: &'a Self) {
        if self.is_zero() {
            *self = *other;
            return;
        }

        if other.is_zero() {
            return;
        }

        // https://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-add-2008-s
        // Works for all curves.

        // U1 = X1*ZZ2
        let mut u1 = self.x;
        u1 *= &other.zz;

        // U2 = X2*ZZ1
        let mut u2 = other.x;
        u2 *= &self.zz;

        // S1 = Y1*ZZZ2
        let mut s1= self.y;
        s1 *= &other.zzz;

        // S2 = Y2*ZZZ1
        let mut s2 = other.y;
        s2 *= &self.zzz;

        if u1 == u2 && s1 == s2 {
            // The two points are equal, so we double.
            self.double_in_place();
        } else {
            u2 -= &u1;
            s2 -= &s1;

            let mut pp = u2;
            pp.square_in_place();

            let mut q = u1;
            q *= &pp;

            self.zz *= &other.zz;
            self.zz *= &pp;

            // PPP
            pp *= &u2;
            
            self.x = s2;
            self.x.square_in_place();
            self.x -= &pp;
            self.x -= &q.double();

            self.zzz *= &other.zzz;
            self.zzz *= &pp;

            q -= &self.x;
            s1.neg_in_place();
            self.y = P::BaseField::sum_of_products(&[s2, s1], &[q, pp]);
        }
    }
}

impl<'a, P: SWCurveConfig> Sub<&'a Self> for ExtendedJacobian<P> {
    type Output = Self;

    #[inline]
    fn sub(mut self, other: &'a Self) -> Self {
        self -= other;
        self
    }
}

impl<'a, P: SWCurveConfig> SubAssign<&'a Self> for ExtendedJacobian<P> {
    fn sub_assign(&mut self, other: &'a Self) {
        *self += &(-(*other));
    }
}

impl<P: SWCurveConfig, T: Borrow<P::ScalarField>> MulAssign<T> for ExtendedJacobian<P> {
    fn mul_assign(&mut self, other: T) {
        *self = self.mul_bigint(other.borrow().into_bigint())
    }
}

impl<P: SWCurveConfig, T: Borrow<P::ScalarField>> Mul<T> for ExtendedJacobian<P> {
    type Output = Self;

    #[inline]
    fn mul(mut self, other: T) -> Self {
        self *= other;
        self
    }
}

// The affine point X, Y is represented in the Jacobian
// coordinates with Z = 1.
impl<P: SWCurveConfig> From<Affine<P>> for ExtendedJacobian<P> {
    #[inline]
    fn from(p: Affine<P>) -> ExtendedJacobian<P> {
        p.xy().map_or(ExtendedJacobian::zero(), |(&x, &y)| Self {
            x,
            y,
            zz: P::BaseField::one(),
            zzz: P::BaseField::one(),
        })
    }
}

impl<P: SWCurveConfig> From<Projective<P>> for ExtendedJacobian<P> {
    #[inline]
    fn from(p: Projective<P>) -> ExtendedJacobian<P> {
        let mut zz = p.z;
        zz.square_in_place();

        let mut zzz = zz;
        zzz *= &p.z;

        Self {
            x: p.x,
            y: p.y,
            zz,
            zzz
        }
    }
}

impl<P: SWCurveConfig> CanonicalSerialize for ExtendedJacobian<P> {
    #[inline]
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let aff = Affine::<P>::from(*self);
        P::serialize_with_mode(&aff, writer, compress)
    }

    #[inline]
    fn serialized_size(&self, compress: Compress) -> usize {
        P::serialized_size(compress)
    }
}

impl<P: SWCurveConfig> Valid for ExtendedJacobian<P> {
    fn check(&self) -> Result<(), SerializationError> {
        let mut zz = self.zz;
        zz.square_in_place();
        zz *= &self.zz;

        let mut zzz = self.zzz;
        zzz.square_in_place();

        if zz != zzz {
            return Err(SerializationError::InvalidData);
        }

        let affine : Affine<P> = (*self).into();
        affine.check()
    }
}

impl<P: SWCurveConfig> CanonicalDeserialize for ExtendedJacobian<P> {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let aff = P::deserialize_with_mode(reader, compress, validate)?;
        Ok(aff.into())
    }
}

impl<M: SWCurveConfig, ConstraintF: Field> ToConstraintField<ConstraintF> for ExtendedJacobian<M>
where
    M::BaseField: ToConstraintField<ConstraintF>,
{
    #[inline]
    fn to_field_elements(&self) -> Option<Vec<ConstraintF>> {
        Affine::from(*self).to_field_elements()
    }
}

impl<P: SWCurveConfig> ScalarMul for ExtendedJacobian<P> {
    type MulBase = Affine<P>;
    const NEGATION_IS_CHEAP: bool = true;

    fn batch_convert_to_mul_base(bases: &[Self]) -> Vec<Self::MulBase> {
        Self::normalize_batch(bases)
    }
}

impl<P: SWCurveConfig> VariableBaseMSM for ExtendedJacobian<P> {}

impl<P: SWCurveConfig, T: Borrow<Affine<P>>> core::iter::Sum<T> for ExtendedJacobian<P> {
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.fold(ExtendedJacobian::zero(), |sum, x| sum + x.borrow())
    }
}

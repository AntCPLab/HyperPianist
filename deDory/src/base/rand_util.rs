use ark_ec::pairing::Pairing;
#[cfg(test)]
use ark_std::rand::{rngs::StdRng, SeedableRng};
use ark_std::{rand::Rng, UniformRand};

use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Create a random number generator for testing.
pub fn test_rng() -> impl Rng {
    ark_std::test_rng()
}

/// Create a random number generator for testing with a specific seed.
#[cfg(test)]
pub fn test_seed_rng(seed: [u8; 32]) -> impl Rng {
    StdRng::from_seed(seed)
}

/// Creates two vectors of random G1 and G2 elements with length 2^num.
pub fn rand_G_vecs<E>(
    num: usize,
    rng: &mut impl ark_std::rand::Rng,
) -> (Vec<E::G1Affine>, Vec<E::G2Affine>)
where
    E: Pairing,
{
    core::iter::repeat_with(|| (E::G1Affine::rand(rng), E::G2Affine::rand(rng)))
        .take(num)
        .unzip()
}

pub fn rand_G_vecs_par<E>(
    num: usize,
) -> (Vec<E::G1Affine>, Vec<E::G2Affine>)
where
    E: Pairing,
{
    (0..num)
        .into_par_iter()
        .map_init(
            || rand::thread_rng(),
            |rng, _| (E::G1Affine::rand(rng), E::G2Affine::rand(rng)),
        )
        .unzip()
}

/// Creates two vectors of random F elements with length 2^num.
pub fn rand_F_vecs<E>(
    num: usize,
    rng: &mut impl ark_std::rand::Rng,
) -> (Vec<E::ScalarField>, Vec<E::ScalarField>)
where
    E: Pairing,
{
    core::iter::repeat_with(|| (E::ScalarField::rand(rng), E::ScalarField::rand(rng)))
        .take(1 << num)
        .unzip()
}

/// Creates two vectors of random F elements with length 2^num.
pub fn rand_F_tensors<E>(
    num: usize,
    rng: &mut impl ark_std::rand::Rng,
) -> (Vec<E::ScalarField>, Vec<E::ScalarField>)
where
    E: Pairing,
{
    core::iter::repeat_with(|| (E::ScalarField::rand(rng), E::ScalarField::rand(rng)))
        .take(num)
        .unzip()
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_bls12_381::Bls12_381 as E;

    #[test]
    fn we_can_create_rand_G_vecs() {
        let mut rng = test_rng();
        for num in 0..5 {
            let (Gamma_1, Gamma_2) = rand_G_vecs::<E>(1 << num, &mut rng);
            assert_eq!(Gamma_1.len(), 1 << num);
            assert_eq!(Gamma_2.len(), 1 << num);
        }
    }

    #[test]
    fn we_can_create_different_rand_G_vecs_consecutively_from_the_same_rng() {
        let mut rng = test_rng();
        for num in 0..5 {
            let (Gamma_1, Gamma_2) = rand_G_vecs::<E>(1 << num, &mut rng);
            let (Gamma_1_2, Gamma_2_2) = rand_G_vecs::<E>(1 << num, &mut rng);
            assert_ne!(Gamma_1, Gamma_1_2);
            assert_ne!(Gamma_2, Gamma_2_2);
        }
    }

    #[test]
    fn we_can_create_the_same_rand_G_vecs_from_the_same_seed() {
        let mut rng = test_seed_rng([1; 32]);
        let mut rng_2 = test_seed_rng([1; 32]);
        for num in 0..5 {
            let (Gamma_1, Gamma_2) = rand_G_vecs::<E>(1 << num, &mut rng);
            let (Gamma_1_2, Gamma_2_2) = rand_G_vecs::<E>(1 << num, &mut rng_2);
            assert_eq!(Gamma_1, Gamma_1_2);
            assert_eq!(Gamma_2, Gamma_2_2);
        }
    }

    #[test]
    fn we_can_create_different_rand_G_vecs_from_different_seeds() {
        let mut rng = test_seed_rng([1; 32]);
        let mut rng_2 = test_seed_rng([2; 32]);
        for num in 0..5 {
            let (Gamma_1, Gamma_2) = rand_G_vecs::<E>(1 << num, &mut rng);
            let (Gamma_1_2, Gamma_2_2) = rand_G_vecs::<E>(1 << num, &mut rng_2);
            assert_ne!(Gamma_1, Gamma_1_2);
            assert_ne!(Gamma_2, Gamma_2_2);
        }
    }

    #[test]
    fn we_can_create_rand_F_vecs() {
        let mut rng = test_rng();
        for num in 0..5 {
            let (s1, s2) = rand_F_vecs::<E>(num, &mut rng);
            assert_eq!(s1.len(), 1 << num);
            assert_eq!(s2.len(), 1 << num);
            assert_ne!(s1, s2);
        }
    }

    #[test]
    fn we_can_create_different_rand_F_vecs_consecutively_from_the_same_rng() {
        let mut rng = test_rng();
        for num in 0..5 {
            let (s1, s2) = rand_F_vecs::<E>(num, &mut rng);
            let (s1_2, s2_2) = rand_F_vecs::<E>(num, &mut rng);
            assert_ne!(s1, s1_2);
            assert_ne!(s2, s2_2);
        }
    }

    #[test]
    fn we_can_create_the_same_rand_F_vecs_from_the_same_seed() {
        let mut rng = test_seed_rng([1; 32]);
        let mut rng_2 = test_seed_rng([1; 32]);
        for num in 0..5 {
            let (s1, s2) = rand_F_vecs::<E>(num, &mut rng);
            let (s1_2, s2_2) = rand_F_vecs::<E>(num, &mut rng_2);
            assert_eq!(s1, s1_2);
            assert_eq!(s2, s2_2);
        }
    }

    #[test]
    fn we_can_create_different_rand_F_vecs_from_different_seeds() {
        let mut rng = test_seed_rng([1; 32]);
        let mut rng_2 = test_seed_rng([2; 32]);
        for num in 0..5 {
            let (s1, s2) = rand_F_vecs::<E>(num, &mut rng);
            let (s1_2, s2_2) = rand_F_vecs::<E>(num, &mut rng_2);
            assert_ne!(s1, s1_2);
            assert_ne!(s2, s2_2);
        }
    }
}

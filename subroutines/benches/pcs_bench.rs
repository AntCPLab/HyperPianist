// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

use arithmetic::evaluate_opt;
use ark_bn254::Bn254;
use ark_ec::{pairing::Pairing, CurveGroup, Group, VariableBaseMSM};
use ark_ff::UniformRand;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::{sync::Arc, test_rng, One, Zero};
use rayon::iter::{repeat, IntoParallelIterator, ParallelIterator};
use std::time::Instant;
use subroutines::{
    pcs::prelude::{Dory, PCSError, PolynomialCommitmentScheme},
    MultilinearKzgPCS,
};

use ark_bn254::{G1Affine, G1ExtendedJacobian};

fn main() {
    bench_msm::<Bn254>();
    // bench_ext_jac();
    // bench_g::<Bn254>();
    // bench_fp::<Bn254>();
    // bench_pcs::<Bn254, Dory<Bn254>>("Dory", 12).unwrap();
    // bench_pcs::<Bn254, MultilinearKzgPCS<Bn254>>("KZG", 24).unwrap();
}

fn bench_msm<E: Pairing>() {
    let num = 1 << 25;
    let test_points = (0..num)
        .into_par_iter()
        .map_init(|| rand::thread_rng(), |rng, _| E::G1Affine::rand(rng))
        .collect::<Vec<_>>();
    let test_scalars = (0..num)
        .into_par_iter()
        .map_init(|| rand::thread_rng(), |rng, _| E::ScalarField::rand(rng))
        .collect::<Vec<_>>();

    for nv in 20..=25 {
        let repetition = if nv < 10 {
            10
        } else if nv < 20 {
            5
        } else {
            2
        };

        let num = 1 << nv;

        let start = Instant::now();
        for _ in 0..repetition {
            let _value: E::G1Affine =
                E::G1MSM::msm_unchecked_par_auto(&test_points[..num], &test_scalars[..num]).into();
        }

        println!(
            "msm for {} variables: {} ns",
            nv,
            start.elapsed().as_nanos() / repetition as u128
        );
    }
}

fn bench_fp<E: Pairing>() {
    let mut val = E::ScalarField::one();
    let test_scalar = E::ScalarField::rand(&mut rand::thread_rng());
    let start = Instant::now();
    let repetition = 100000000;
    for _ in 0..repetition {
        val *= test_scalar;
    }
    println!(
        "mul: {} ns",
        (start.elapsed().as_nanos() as f64) / repetition as f64
    );
    println!("{:?}", val);

    let start = Instant::now();
    let repetition = 1000000000;
    for _ in 0..repetition {
        val += test_scalar;
    }
    println!(
        "add: {} ns",
        (start.elapsed().as_nanos() as f64) / repetition as f64
    );
    println!("{:?}", val);

    let start = Instant::now();
    let repetition = 1000000000;
    for _ in 0..repetition {
        val -= test_scalar;
    }
    println!(
        "sub: {} ns",
        (start.elapsed().as_nanos() as f64) / repetition as f64
    );
    println!("{:?}", val);
}

fn bench_g<E: Pairing>() {
    let mut rng = &mut rand::thread_rng();
    let test_g = E::G1Affine::rand(&mut rng);
    let mut val = E::G1::zero();
    let start = Instant::now();
    let repetition = 10000000;
    for _ in 0..repetition {
        val += test_g;
    }
    println!(
        "mixed_add: {} ns",
        (start.elapsed().as_nanos() as f64) / repetition as f64
    );
    println!("{:?}", val);

    let mut val2 = E::G1::zero();
    let start = Instant::now();
    let repetition = 10000000;
    for _ in 0..repetition {
        val2 += val;
    }
    println!(
        "add: {} ns",
        (start.elapsed().as_nanos() as f64) / repetition as f64
    );
    println!("{:?}", val2);

    let start = Instant::now();
    let repetition = 10000000;
    for _ in 0..repetition {
        val2.double_in_place();
    }
    println!(
        "double: {} ns",
        (start.elapsed().as_nanos() as f64) / repetition as f64
    );
    println!("{:?}", val2);
}

fn bench_ext_jac() {
    let mut rng = &mut rand::thread_rng();
    let test_g = G1Affine::rand(&mut rng);
    let mut val = G1ExtendedJacobian::zero();
    let start = Instant::now();
    let repetition = 10000000;
    for _ in 0..repetition {
        val += test_g;
    }
    println!(
        "mixed_add: {} ns",
        (start.elapsed().as_nanos() as f64) / repetition as f64
    );
    println!("{:?}", val);

    let mut val2 = G1ExtendedJacobian::zero();
    let start = Instant::now();
    let repetition = 10000000;
    for _ in 0..repetition {
        val2 += val;
    }
    println!(
        "add: {} ns",
        (start.elapsed().as_nanos() as f64) / repetition as f64
    );
    println!("{:?}", val2);

    let start = Instant::now();
    let repetition = 10000000;
    for _ in 0..repetition {
        val2.double_in_place();
    }
    println!(
        "double: {} ns",
        (start.elapsed().as_nanos() as f64) / repetition as f64
    );
    println!("{:?}", val2);
}

fn bench_pcs<
    E: Pairing,
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        Point = Vec<E::ScalarField>,
    >,
>(
    name: &str,
    supported_nv: usize,
) -> Result<(), PCSError> {
    let mut rng = test_rng();

    // normal polynomials
    let uni_params = PCS::gen_srs_for_testing(&mut rng, supported_nv)?;

    for nv in [20, 22] {
        let repetition = if nv < 10 {
            10
        } else if nv < 20 {
            5
        } else {
            2
        };

        let poly = Arc::new(DenseMultilinearExtension::rand(nv, &mut rng));
        let (ck, vk) = PCS::trim(&uni_params, None, Some(supported_nv))?;

        let point: Vec<_> = (0..nv).map(|_| E::ScalarField::rand(&mut rng)).collect();

        // commit
        let (com, advice) = {
            let start = Instant::now();
            for _ in 0..repetition {
                let _commit = PCS::commit(&ck, &poly)?;
            }

            println!(
                "{} commit for {} variables: {} ns",
                name,
                nv,
                start.elapsed().as_nanos() / repetition as u128
            );

            PCS::commit(&ck, &poly)?
        };

        // open
        let (proof, value) = {
            let start = Instant::now();
            for _ in 0..repetition {
                let _open = PCS::open(&ck, &poly, &advice, &point)?;
            }

            println!(
                "{} open for {} variables: {} ns",
                name,
                nv,
                start.elapsed().as_nanos() / repetition as u128
            );
            let proof = PCS::open(&ck, &poly, &advice, &point)?;
            let value = evaluate_opt(&poly, &point);
            (proof, value)
        };

        // verify
        {
            let start = Instant::now();
            for _ in 0..repetition {
                assert!(PCS::verify(&vk, &com, &point, &value, &proof)?);
            }
            println!(
                "{} verify for {} variables: {} ns",
                name,
                nv,
                start.elapsed().as_nanos() / repetition as u128
            );
        }

        println!("====================================");
    }

    Ok(())
}

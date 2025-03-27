// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

#[macro_use]
extern crate criterion;

use arithmetic::{bind_poly_var_bot, fix_variables, bind_poly_var_bot_par};
use ark_bls12_381::Fr;
use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::test_rng;
use criterion::{black_box, BatchSize, BenchmarkId, Criterion};

fn evaluation_op_bench<F: PrimeField>(c: &mut Criterion) {
    let mut rng = test_rng();
    let mut group = c.benchmark_group("Evaluate");
    for nv in [15, 18, 21] {
        group.bench_with_input(BenchmarkId::new("bind poly var bot", nv), &nv, |b, &nv| {
            let poly = DenseMultilinearExtension::<F>::rand(nv, &mut rng);
            let point: Vec<_> = (0..nv).map(|_| F::rand(&mut rng)).collect();
            b.iter_batched(
                || poly.clone(),
                |mut poly| black_box(bind_poly_var_bot(&mut poly, &point[0])),
                BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("bind poly var bot par", nv), &nv, |b, &nv| {
            let poly = DenseMultilinearExtension::<F>::rand(nv, &mut rng);
            let point: Vec<_> = (0..nv).map(|_| F::rand(&mut rng)).collect();
            b.iter_batched(
                || poly.clone(),
                |mut poly| black_box(bind_poly_var_bot_par(&mut poly, &point[0], rayon::current_num_threads() * 2)),
                BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("fix variables", nv), &nv, |b, &nv| {
            let poly = DenseMultilinearExtension::<F>::rand(nv, &mut rng);
            let point: Vec<_> = (0..nv).map(|_| F::rand(&mut rng)).collect();
            b.iter(|| black_box(fix_variables(&poly, &[point[0]])))
        });
    }
    group.finish();
}

fn bench_bls_381(c: &mut Criterion) {
    evaluation_op_bench::<Fr>(c);
}

criterion_group!(benches, bench_bls_381);
criterion_main!(benches);

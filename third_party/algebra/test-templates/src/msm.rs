use ark_ec::{
    scalar_mul::variable_base::{ChunkedPippenger, HashMapPippenger, VariableBaseMSM},
    ScalarMul,
};
use ark_ff::{PrimeField, UniformRand};
use ark_std::{vec::Vec, ops::Mul};

fn naive_var_base_msm<G: ScalarMul>(bases: &[G::MulBase], scalars: &[G::ScalarField]) -> G
    where G::MulBase : for<'a> Mul<&'a G::ScalarField, Output = G>
{
    let mut acc = G::zero();

    for (base, scalar) in bases.iter().zip(scalars.iter()) {
        acc += *base * scalar;
    }
    acc
}

pub fn test_var_base_msm_small<G: VariableBaseMSM, R>()
where
    R: VariableBaseMSM<MulBase = G::MulBase, ScalarField = G::ScalarField>,
    R::MulBase : for<'a> Mul<&'a R::ScalarField, Output = R>,
    R: Into<G>,
{
    const SAMPLES: usize = 1 << 4;

    let mut rng = ark_std::test_rng();

    let v = (0..SAMPLES)
        .map(|_| G::ScalarField::rand(&mut rng))
        .collect::<Vec<_>>();
    let g = (0..SAMPLES).map(|_| G::rand(&mut rng)).collect::<Vec<_>>();
    let g = G::batch_convert_to_mul_base(&g);

    let naive : G = naive_var_base_msm::<R>(g.as_slice(), v.as_slice()).into();
    let fast = G::msm(g.as_slice(), v.as_slice()).unwrap();

    assert_eq!(naive, fast);
}

pub fn test_var_base_msm<G: VariableBaseMSM, R>()
where
    R: VariableBaseMSM<MulBase = G::MulBase, ScalarField = G::ScalarField>,
    R::MulBase : for<'a> Mul<&'a R::ScalarField, Output = R>,
    R: Into<G>,
{
    const SAMPLES: usize = 1 << 10;

    let mut rng = ark_std::test_rng();

    let v = (0..SAMPLES)
        .map(|_| G::ScalarField::rand(&mut rng))
        .collect::<Vec<_>>();
    let g = (0..SAMPLES).map(|_| G::rand(&mut rng)).collect::<Vec<_>>();
    let g = G::batch_convert_to_mul_base(&g);

    let naive : G = naive_var_base_msm::<R>(g.as_slice(), v.as_slice()).into();
    let fast = G::msm(g.as_slice(), v.as_slice()).unwrap();

    assert_eq!(naive, fast);

    // Parallel is not enabled by default for tests. Here we test the splitting & combination.
    let fast_par = G::msm_unchecked_par(g.as_slice(), v.as_slice(), 64);
    assert_eq!(naive, fast_par);
}

pub fn test_chunked_pippenger<G: VariableBaseMSM>() {
    const SAMPLES: usize = 1 << 10;

    let mut rng = ark_std::test_rng();

    let v = (0..SAMPLES)
        .map(|_| G::ScalarField::rand(&mut rng).into_bigint())
        .collect::<Vec<_>>();
    let g = (0..SAMPLES).map(|_| G::rand(&mut rng)).collect::<Vec<_>>();
    let g = G::batch_convert_to_mul_base(&g);

    let arkworks = G::msm_bigint(g.as_slice(), v.as_slice());

    let mut p = ChunkedPippenger::<G>::new(1 << 20);
    for (s, g) in v.iter().zip(g) {
        p.add(g, s);
    }
    let mine = p.finalize();
    assert_eq!(arkworks, mine);
}

pub fn test_hashmap_pippenger<G: VariableBaseMSM>() {
    const SAMPLES: usize = 1 << 10;

    let mut rng = ark_std::test_rng();

    let mut v_scal = Vec::new();
    let v = (0..SAMPLES)
        .map(|_| {
            let x = G::ScalarField::rand(&mut rng);
            v_scal.push(x);
            x.into_bigint()
        })
        .collect::<Vec<_>>();
    let g = (0..SAMPLES).map(|_| G::rand(&mut rng)).collect::<Vec<_>>();
    let g = G::batch_convert_to_mul_base(&g);

    let arkworks = G::msm_bigint(g.as_slice(), v.as_slice());

    let mut p = HashMapPippenger::<G>::new(1 << 20);
    for (s, g) in v_scal.iter().zip(g) {
        p.add(g, s);
    }
    let mine = p.finalize();
    assert_eq!(arkworks, mine);
}

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

use arithmetic::{math::Math, VirtualPolynomial};
use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use std::{ops::FnOnce, path::PathBuf, sync::Arc};
use structopt::StructOpt;
use rand::{rngs::StdRng, SeedableRng};

#[derive(Debug, StructOpt)]
#[structopt(name = "example", about = "An example of StructOpt usage.")]
struct Opt {
    /// Id
    id: usize,

    /// Input file
    #[structopt(parse(from_os_str))]
    input: PathBuf,
}

pub(super) fn network_run<F>(func: F)
where
    F: FnOnce() -> (),
{
    let opt = Opt::from_args();
    Net::init_from_file(opt.input.to_str().unwrap(), opt.id);

    func();

    Net::deinit();
}

pub(super) fn d_evaluate<F: PrimeField>(
    poly: &VirtualPolynomial<F>,
    point: Option<&[F]>,
) -> Option<F> {
    if Net::am_master() {
        let num_party_vars = Net::n_parties().log_2() as usize;
        let point = point.unwrap();
        let nv = point.len() - num_party_vars;
        Net::recv_from_master_uniform(Some(point[..nv].to_vec()));

        let evals = poly.flattened_ml_extensions.iter()
            .map(|mle| mle.evaluate(&point[..nv]).unwrap())
            .collect::<Vec<_>>();

        let evals = Net::send_to_master(&evals).unwrap();
        let mle_evals = (0..evals[0].len())
            .map(|mle_index| 
                DenseMultilinearExtension::from_evaluations_vec(num_party_vars, evals.iter().map(
                    |party_evals| party_evals[mle_index]
                ).collect())
                .evaluate(&point[nv..]).unwrap()
            )
            .collect::<Vec<_>>();

        let result = poly.products.iter()
            .map(|(coeff, indices)| *coeff * indices.iter().map(
                |index| mle_evals[*index]
            ).product::<F>())
            .sum();
        Some(result)
    } else {
        let point = Net::recv_from_master_uniform::<Vec<F>>(None);
        let evals = poly.flattened_ml_extensions.iter()
            .map(|mle| mle.evaluate(&point).unwrap())
            .collect::<Vec<_>>();
        Net::send_to_master(&evals);
        None
    }
}

pub(super) fn d_evaluate_mle<F: PrimeField>(
    poly: &Arc<DenseMultilinearExtension<F>>,
    point: Option<&[F]>,
) -> Option<F> {
    d_evaluate(&VirtualPolynomial::new_from_mle(poly, F::one()), point)
}

pub(super) fn test_rng() -> StdRng {
    let mut seed = [0u8; 32];
    seed[0] = Net::party_id() as u8;
    rand::rngs::StdRng::from_seed(seed)
}

pub(super) fn test_rng_deterministic() -> StdRng {
    let seed = [69u8; 32];
    rand::rngs::StdRng::from_seed(seed)
}

use deDory::{SubProverSetup, PublicParameters, VerifierSetup, DeDoryCommitment};
use deDory::eval::{de_generate_eval_proof, verify_de_eval_proof, compute_evaluation_vector};
use ark_ec::pairing::Pairing;
use ark_std::UniformRand;
use ark_std::test_rng;
use ark_ff::Field;
use std::mem;
use std::time::Instant;
use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};
use transcript::IOPTranscript;
use std::path::Path;
use ark_bls12_381::Bls12_381;

use log::debug;
use std::path::PathBuf;
use structopt::StructOpt;
#[derive(Debug, StructOpt)]
#[structopt(name = "example", about = "An example of StructOpt usage.")]
struct Opt {
    /// Id
    id: usize,

    /// Input file
    #[structopt(parse(from_os_str))]
    input: PathBuf,
}

fn init() -> (usize, usize, usize, usize) {
    let opt = Opt::from_args();
    println!("{:?}", opt);
    Net::init_from_file(opt.input.to_str().unwrap(), opt.id);
    let M = Net::n_parties();
    let m:usize = ark_std::log2(M).try_into().unwrap();
    let sub_prover_id = Net::party_id();
    let n = 10;
    let max = 10;
    assert!((n + m) / 2 < max);

    println!("n: {:?}, m: {:?}", n, m);

    (n, m, max, sub_prover_id)
}

fn test_dedory_helper<E: Pairing>(
    sub_prover_setup: &SubProverSetup<E>,
    verifier_setup: &VerifierSetup<E>,
    n: usize,
    m: usize,
    sub_prover_id: usize,
) {
    // Generate random polynomial and evaluation points
    let time = Instant::now();
    let mut rng = ark_std::test_rng();
    let evals_num = 1usize << n;
    let f_evals = core::iter::repeat_with(|| E::ScalarField::rand(&mut rng))
        .take(evals_num)
        .collect::<Vec<_>>();
    let b_point = core::iter::repeat_with(|| E::ScalarField::rand(&mut rng))
        .take(n)
        .collect::<Vec<_>>();
    println!("Generating random poly evaluations and points time: {:?}", time.elapsed());
    
    // Generate sub-prover's witness matrix
    let time = Instant::now();
    let sub_mat_len = 1usize << ((n - m) / 2);
    // let top_offset = (n + m) / 2;

    let sub_witness_vec = &f_evals[(sub_prover_id << (n - m)).. ((sub_prover_id + 1) << (n - m))];
    println!("Generating sub-prover's witness matrix time: {:?}", time.elapsed());

    // Commit
    let time = Instant::now();
    let (full_f_comm, sub_T_vec_prime) = DeDoryCommitment::deCommit(sub_prover_id, &sub_witness_vec, m, n, &sub_prover_setup);
    println!("DeCommiting time: {:?}", time.elapsed());

    // DeProve
    let time = Instant::now();
    let mut sub_prover_transcript = IOPTranscript::new(b"Distributed Dory Evaluation Proof");
    let eval_proof = de_generate_eval_proof(
        sub_prover_id,
        &mut sub_prover_transcript,
        sub_witness_vec,
        &sub_T_vec_prime,
        &b_point,
        n,
        m,
        &sub_prover_setup,
    );
    println!("Proving time: {:?}", time.elapsed());

    // Verify
    if Net::am_master() {

        // Compute evaluation vector and product
        let time = Instant::now();
        let mut b = vec![E::ScalarField::ZERO; evals_num];
        compute_evaluation_vector(&mut b, &b_point);
        let product = f_evals.iter().zip(b.iter()).map(|(f, b)| *f * *b).sum();
        println!("Computing evaluation vector and product time: {:?}", time.elapsed());

        let mut eval_proof = eval_proof.unwrap();
        let time = Instant::now();
        let mut verifier_transcript = IOPTranscript::new(b"Distributed Dory Evaluation Proof");
        let r = verify_de_eval_proof(
            &mut verifier_transcript,
            &mut eval_proof,
            &full_f_comm.unwrap(),
            product,
            &b_point,
            n,
            m,
            &verifier_setup,
        );
        println!("Verification time: {:?}", time.elapsed());
        assert!(r.is_ok());
    }

    Net::deinit();
}

fn new_setup<E: Pairing>(
    max: usize
) -> (SubProverSetup<E>, VerifierSetup<E>) {
    // Setup
    let time = Instant::now();
    let mut rng = ark_std::test_rng();
    let public_parameters = if Net::am_master() {
        let public_parameters:PublicParameters<E> = PublicParameters::rand(max, &mut rng);
        println!("Generating public parameter time: {:?}", time.elapsed());

        Net::recv_from_master_uniform(Some(public_parameters))
    } else {
        Net::recv_from_master_uniform(None)
    };
    let time = Instant::now();

    let sub_prover_setup = SubProverSetup::new(&public_parameters);
    println!("Sub-Prover setup time: {:?}", time.elapsed());
    let time = Instant::now();
    let verifier_setup = VerifierSetup::new(&public_parameters);
    println!("Verifier setup time: {:?}", time.elapsed());

    (sub_prover_setup, verifier_setup)
}


fn new_setup_to_file<E: Pairing>(
    max: usize
) -> (SubProverSetup<E>, VerifierSetup<E>) {
    // Setup
    let time = Instant::now();
    let mut rng = ark_std::test_rng();
    let public_parameters = if Net::am_master() {
        let public_parameters:PublicParameters<E> = PublicParameters::rand(max, &mut rng);
        println!("Generating public parameter time: {:?}", time.elapsed());

        Net::recv_from_master_uniform(Some(public_parameters))
    } else {
        Net::recv_from_master_uniform(None)
    };
    let time = Instant::now();

    let sub_prover_setup_filepath = format!("../data/SubProver-max{}.paras", max);
    let verifier_setup_filepath = format!("../data/Verifier-max{}.paras", max);
    SubProverSetup::new_to_file(&public_parameters, &sub_prover_setup_filepath);
    println!("Sub-Prover setup, writing to and reading from file time: {:?}", time.elapsed());
    let time = Instant::now();
    VerifierSetup::new_to_file(&public_parameters, &verifier_setup_filepath);
    println!("Verifier setup, writing to and reading from filetime: {:?}", time.elapsed());

    let sub_prover_setup = SubProverSetup::read_from_file(&sub_prover_setup_filepath).unwrap();
    let verifier_setup = VerifierSetup::read_from_file(&verifier_setup_filepath).unwrap();
    
    (sub_prover_setup, verifier_setup)
}

fn setup_from_file<E: Pairing>(
    max: usize
) -> (SubProverSetup<E>, VerifierSetup<E>) {
    let sub_prover_setup_filepath = format!("../data/SubProver-max{}.paras", max);
    let verifier_setup_filepath = format!("../data/Verifier-max{}.paras", max);

    let time = Instant::now();
    let sub_prover_setup = SubProverSetup::read_from_file(&sub_prover_setup_filepath).unwrap();
    println!("Reading sub-prover's paras from file time: {:?}", time.elapsed());
    let time = Instant::now();
    let verifier_setup = VerifierSetup::read_from_file(&verifier_setup_filepath).unwrap();
    println!("Reading verifier's paras from file time: {:?}", time.elapsed());

    (sub_prover_setup, verifier_setup)
}


fn test_dedory_with_new_setup<E: Pairing>() {
    let (n, m, max, sub_prover_id) = init();
    let (sub_prover_setup, verifier_setup) = new_setup::<E>(max);
    test_dedory_helper::<E>(&sub_prover_setup, &verifier_setup, n, m, sub_prover_id);
}

fn test_dedory_with_new_setup_to_file<E: Pairing>() {
    let (n, m, max, sub_prover_id) = init();
    let (sub_prover_setup, verifier_setup) = new_setup_to_file::<E>(max);
    test_dedory_helper::<E>(&sub_prover_setup, &verifier_setup, n, m, sub_prover_id);
}

fn test_dedory_with_setup_from_file<E: Pairing>() {
    let (n, m, max, sub_prover_id) = init();
    let (sub_prover_setup, verifier_setup) = setup_from_file::<E>(max);
    test_dedory_helper::<E>(&sub_prover_setup, &verifier_setup, n, m, sub_prover_id);
}


fn main() {
    test_dedory_with_new_setup::<Bls12_381>();
    // test_dedory_with_new_setup_to_file::<Bls12_381>();
    // test_dedory_with_setup_from_file::<Bls12_381>();
}

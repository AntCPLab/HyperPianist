use std::{fs::File, io::Write, marker::PhantomData, time::Instant};

use arithmetic::math::Math;
use ark_bn254::{Bn254, Fr};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::UniformRand;
use deDory::{PublicParameters, SubProverSetup, VerifierSetup};
use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet, Stats};
use hyperpianist::{
    prelude::{CustomizedGates, HyperPlonkErrors, MockCircuit},
    HyperPlonkSNARK,
};
use std::{error::Error, path::PathBuf, sync::Arc};
use structopt::StructOpt;
use subroutines::{
    pcs::{
        prelude::{DeDory, DeDorySRS},
        PolynomialCommitmentScheme,
    },
    poly_iop::PolyIOP,
    BatchProof, DeMkzg, DeMkzgSRS, MultilinearProverParam, MultilinearVerifierParam,
};

mod common;
use common::{d_evaluate_mle, test_rng, test_rng_deterministic};

const SUPPORTED_SIZE: usize = 16;
const MIN_NUM_VARS: usize = 22;
const MAX_NUM_VARS: usize = 26;
const MIN_CUSTOM_DEGREE: usize = 1;
const MAX_CUSTOM_DEGREE: usize = 32;
const HIGH_DEGREE_TEST_NV: usize = 15;

#[derive(Debug, StructOpt)]
#[structopt(name = "example", about = "An example of StructOpt usage.")]
struct Opt {
    /// Id
    id: usize,

    /// Input file
    #[structopt(parse(from_os_str))]
    input: PathBuf,

    #[structopt(long)]
    dory: bool,

    #[structopt(long)]
    jellyfish: bool,

    num_vars: usize,
}

fn main() {
    let thread = rayon::current_num_threads();
    let opt = Opt::from_args();
    Net::init_from_file(opt.input.to_str().unwrap(), opt.id);

    let dedory_pcs_srs = {
        match read_dedory_srs() {
            Ok(p) => p,
            Err(_e) => {
                let mut srs_rng = test_rng_deterministic();
                let srs = DeDory::<Bn254>::gen_srs_for_testing(&mut srs_rng, SUPPORTED_SIZE).unwrap();
                let pp = match srs {
                    DeDorySRS::Unprocessed(pp) => pp,
                    _ => panic!("Unexpected processed"),
                };
                write_dedory_srs(&pp);

                let (prover, verifier) =
                    DeDory::trim(&DeDorySRS::Unprocessed(pp), None, None).unwrap();
                DeDorySRS::Processed((prover, verifier))
            },
        }
    };

    let deMkzg_pcs_srs = {
        match read_deMkzg_srs() {
            Ok(p) => p,
            Err(_e) => {
                let mut srs_rng = test_rng_deterministic();
                let srs = DeMkzg::<Bn254>::gen_srs_for_testing(&mut srs_rng, MAX_NUM_VARS).unwrap();
                let (prover, verifier) = DeMkzg::trim(&srs, None, Some(MAX_NUM_VARS)).unwrap();
                write_deMkzg_srs(&prover, &verifier);
                DeMkzgSRS::Processed((prover, verifier))
            },
        }
    };

    if opt.dory {
        if opt.jellyfish {
            Helper::<DeDory<Bn254>>::bench_jellyfish_plonk(&dedory_pcs_srs, thread, opt.num_vars).unwrap();
        } else {
            Helper::<DeDory<Bn254>>::bench_vanilla_plonk(&dedory_pcs_srs, thread, opt.num_vars).unwrap();
        }
    } else {
        if opt.jellyfish {
            Helper::<DeMkzg<Bn254>>::bench_jellyfish_plonk(&deMkzg_pcs_srs, thread, opt.num_vars).unwrap();
        } else {
            Helper::<DeMkzg<Bn254>>::bench_vanilla_plonk(&deMkzg_pcs_srs, thread, opt.num_vars).unwrap();
        }
    }
    // bench_jellyfish_plonk(&pcs_srs, thread).unwrap();
    // println!();
    // bench_dedory(&pcs_srs).unwrap();
    // println!("DeDory ====================================");
    // Helper::<DeDory<Bn254>>::bench_pcs(&dedory_pcs_srs).unwrap();
    // Helper::<DeDory<Bn254>>::bench_vanilla_plonk(&dedory_pcs_srs,
    // thread).unwrap();

    // println!();
    // for degree in MIN_CUSTOM_DEGREE..=MAX_CUSTOM_DEGREE {
    //     bench_high_degree_plonk(&pcs_srs, degree, thread).unwrap();
    //     println!();
    // }

    Net::deinit();
}

fn read_dedory_srs() -> Result<DeDorySRS<Bn254>, Box<dyn Error>> {
    let sub_prover_setup_filepath =
        format!("SubProver{}-max{}.paras", Net::party_id(), SUPPORTED_SIZE);
    let verifier_setup_filepath =
        format!("Verifier{}-max{}.paras", Net::party_id(), SUPPORTED_SIZE);
    let prover_setup = SubProverSetup::read_from_file(&sub_prover_setup_filepath)?;
    let verifier_setup = VerifierSetup::read_from_file(&verifier_setup_filepath)?;
    Ok(DeDorySRS::Processed((prover_setup, verifier_setup)))
}

fn write_dedory_srs(pp: &PublicParameters<Bn254>) -> Result<(), Box<dyn Error>> {
    let sub_prover_setup_filepath =
        format!("SubProver{}-max{}.paras", Net::party_id(), SUPPORTED_SIZE);
    let verifier_setup_filepath =
        format!("Verifier{}-max{}.paras", Net::party_id(), SUPPORTED_SIZE);
    SubProverSetup::new_to_file(pp, &sub_prover_setup_filepath)?;
    VerifierSetup::new_to_file(pp, &verifier_setup_filepath)?;
    Ok(())
}

fn read_deMkzg_srs() -> Result<DeMkzgSRS<Bn254>, Box<dyn Error>> {
    let sub_prover_setup_filepath = format!(
        "deMkzg-SubProver{}-max{}.paras",
        Net::party_id(),
        MAX_NUM_VARS
    );
    let verifier_setup_filepath = format!(
        "deMkzg-Verifier{}-max{}.paras",
        Net::party_id(),
        MAX_NUM_VARS
    );
    let prover_setup = {
        let file = std::fs::File::open(sub_prover_setup_filepath)?;
        MultilinearProverParam::deserialize_uncompressed_unchecked(std::io::BufReader::new(file))?
    };

    let verifier_setup = {
        let file = std::fs::File::open(verifier_setup_filepath)?;
        MultilinearVerifierParam::deserialize_uncompressed_unchecked(std::io::BufReader::new(file))?
    };
    Ok(DeMkzgSRS::Processed((prover_setup, verifier_setup)))
}

fn write_deMkzg_srs(
    prover: &MultilinearProverParam<Bn254>,
    verifier: &MultilinearVerifierParam<Bn254>,
) -> Result<(), Box<dyn Error>> {
    let sub_prover_setup_filepath = format!(
        "deMkzg-SubProver{}-max{}.paras",
        Net::party_id(),
        MAX_NUM_VARS
    );
    let verifier_setup_filepath = format!(
        "deMkzg-Verifier{}-max{}.paras",
        Net::party_id(),
        MAX_NUM_VARS
    );

    let file = std::fs::File::create(sub_prover_setup_filepath)?;
    prover.serialize_uncompressed(std::io::BufWriter::new(file))?;

    let file = std::fs::File::create(verifier_setup_filepath)?;
    verifier.serialize_uncompressed(std::io::BufWriter::new(file))?;
    Ok(())
}

fn print_stats(
    before: &Stats,
    after: &Stats
) {
    let to_master = after.to_master - before.to_master;
    let from_master = after.from_master - before.from_master;
    let bytes_sent = after.bytes_sent - before.bytes_sent;
    let bytes_recv = after.bytes_recv - before.bytes_recv;
    println!("to_master: {to_master}, from_master: {from_master}, bytes_sent: {bytes_sent}, bytes_recv: {bytes_recv}");
}

// This is just here to save repeating type constraints for each helper
struct Helper<PCS>(PhantomData<PCS>);

impl<PCS> Helper<PCS>
where
    PCS: PolynomialCommitmentScheme<
        Bn254,
        Polynomial = Arc<DenseMultilinearExtension<Fr>>,
        Point = Vec<Fr>,
        Evaluation = Fr,
        BatchProof = BatchProof<Bn254, PCS>,
    >,
{
    fn bench_pcs(pcs_srs: &PCS::SRS) -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let (ck, vk) = PCS::trim(pcs_srs, None, Some(MAX_NUM_VARS))?;
        for nv in MIN_NUM_VARS..=MAX_NUM_VARS {
            let nv = nv - Net::n_parties().log_2();
            let repetition = 5;

            let poly = Arc::new(DenseMultilinearExtension::rand(nv, &mut rng));

            let point = if Net::am_master() {
                let point = (0..(nv + Net::n_parties().log_2()))
                    .map(|_| Fr::rand(&mut rng))
                    .collect::<Vec<_>>();
                Net::recv_from_master_uniform(Some(point))
            } else {
                Net::recv_from_master_uniform(None)
            };

            // commit
            let (com, advice) = {
                let start = Instant::now();
                for _ in 0..repetition {
                    let _commit = PCS::d_commit(&ck, &poly)?;
                }

                println!(
                    "commit for {} variables: {} ns",
                    nv,
                    start.elapsed().as_nanos() / repetition as u128
                );

                PCS::d_commit(&ck, &poly)?
            };

            // open
            let (proof, value) = {
                let start = Instant::now();
                for _ in 0..repetition {
                    let _open = PCS::open(&ck, &poly, &advice, &point)?;
                }

                println!(
                    "open for {} variables: {} ns",
                    nv,
                    start.elapsed().as_nanos() / repetition as u128
                );
                let proof = PCS::open(&ck, &poly, &advice, &point)?;
                let value = d_evaluate_mle(&poly, Some(&point));
                (proof, value)
            };

            // verify
            if Net::am_master() {
                let com = com.unwrap();
                let value = value.unwrap();
                let start = Instant::now();
                for _ in 0..repetition {
                    assert!(PCS::verify(&vk, &com, &point, &value, &proof)?);
                }
                println!(
                    "verify for {} variables: {} ns",
                    nv,
                    start.elapsed().as_nanos() / repetition as u128
                );
            }

            println!("====================================");
        }

        Ok(())
    }

    fn bench_vanilla_plonk(
        pcs_srs: &PCS::SRS,
        thread: usize,
        nv: usize,
    ) -> Result<(), HyperPlonkErrors> {
        let vanilla_gate = CustomizedGates::vanilla_plonk_gate();
        Self::bench_mock_circuit_zkp_helper(nv, &vanilla_gate, pcs_srs)?;

        Ok(())
    }

    fn bench_jellyfish_plonk(
        pcs_srs: &PCS::SRS,
        thread: usize,
        nv: usize,
    ) -> Result<(), HyperPlonkErrors> {
        let jf_gate = CustomizedGates::jellyfish_turbo_plonk_gate();
        Self::bench_mock_circuit_zkp_helper(nv, &jf_gate, pcs_srs)?;

        Ok(())
    }

    fn bench_high_degree_plonk(
        pcs_srs: &PCS::SRS,
        degree: usize,
        thread: usize,
    ) -> Result<(), HyperPlonkErrors> {
        let vanilla_gate = CustomizedGates::mock_gate(2, degree);
        Self::bench_mock_circuit_zkp_helper(
            HIGH_DEGREE_TEST_NV,
            &vanilla_gate,
            pcs_srs,
        )?;

        Ok(())
    }

    fn bench_mock_circuit_zkp_helper(
        nv: usize,
        gate: &CustomizedGates,
        pcs_srs: &PCS::SRS,
    ) -> Result<(), HyperPlonkErrors> {
        let nv = nv - Net::n_parties().log_2();
        let repetition = if nv <= 20 {
            20
        } else if nv <= 22 {
            10
        } else {
            5
        };

        let mut rng = test_rng();
        //==========================================================
        let circuit = MockCircuit::<Fr>::d_new(1 << nv, gate, &mut rng);
        assert!(circuit.is_satisfied());
        let index = circuit.index;
        //==========================================================
        // generate pk and vks
        let start = Instant::now();
        

        let stats_a = Net::stats();
        let (pk, vk) = <PolyIOP<Fr> as HyperPlonkSNARK<Bn254, PCS>>::d_preprocess(&index, pcs_srs)?;
        println!(
            "key extraction for {} variables: {} us",
            nv,
            start.elapsed().as_micros() as u128
        );
        let stats_b = Net::stats();
        print_stats(&stats_a, &stats_b);

        // Re-synchronize
        Net::recv_from_master_uniform(if Net::am_master() {
            Some(1usize)
        } else {
            None
        });

        //==========================================================
        // generate a proof
        let stats_a = Net::stats();
        let proof = <PolyIOP<Fr> as HyperPlonkSNARK<Bn254, PCS>>::d_prove(
            &pk,
            &circuit.public_inputs,
            &circuit.witnesses,
            &(),
        )?;
        let stats_b = Net::stats();
        print_stats(&stats_a, &stats_b);

        let start = Instant::now();
        for _ in 0..repetition {
            let _proof = <PolyIOP<Fr> as HyperPlonkSNARK<Bn254, PCS>>::d_prove(
                &pk,
                &circuit.public_inputs,
                &circuit.witnesses,
                &(),
            )?;
        }
        println!(
            "proving for {} variables: {} us",
            nv,
            start.elapsed().as_micros() / repetition as u128
        );

        let mut bytes = Vec::with_capacity(CanonicalSerialize::compressed_size(&proof));
        CanonicalSerialize::serialize_compressed(&proof, &mut bytes).unwrap();
        println!("proof size for {} variables compressed: {} bytes", nv, bytes.len());

        let mut bytes = Vec::with_capacity(CanonicalSerialize::uncompressed_size(&proof));
        CanonicalSerialize::serialize_uncompressed(&proof, &mut bytes).unwrap();
        println!("proof size for {} variables uncompressed: {} bytes", nv, bytes.len());

        let all_pi = Net::send_to_master(&circuit.public_inputs);

        if Net::am_master() {
            let vk = vk.unwrap();
            let pi = all_pi.unwrap().concat();
            let proof = proof.unwrap();
            //==========================================================
            // verify a proof
            let start = Instant::now();
            for _ in 0..(repetition * 5) {
                let verify =
                    <PolyIOP<Fr> as HyperPlonkSNARK<Bn254, PCS>>::verify(&vk, &pi, &proof)?;
                assert!(verify);
            }
            println!(
                "verifying for {} variables: {} us",
                nv,
                start.elapsed().as_micros() / (repetition * 5) as u128
            );
        }
        Ok(())
    }
}

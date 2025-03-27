mod common;

use arithmetic::math::Math;
use ark_bn254::{Bn254, Fr};
use ark_ec::pairing::Pairing;
use ark_ff::{One, PrimeField, UniformRand, Zero};
use ark_poly::DenseMultilinearExtension;
use common::{d_evaluate_mle, test_rng};
use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};
use rand_core::RngCore;
use std::sync::Arc;
use subroutines::{
    CombinedCheck, DeMkzg, PolyIOP, PolyIOPErrors, PolynomialCommitmentScheme
};

fn generate_polys<R: RngCore>(
    num_witnesses: usize,
    num_selectors: usize,
    nv: usize,
    gate: &[(Option<usize>, Vec<usize>)],
    rng: &mut R,
) -> (
    Vec<Arc<DenseMultilinearExtension<Fr>>>,
    Vec<Arc<DenseMultilinearExtension<Fr>>>,
    Vec<Arc<DenseMultilinearExtension<Fr>>>,
) {
    let num_constraints = nv.pow2();
    let mut selectors: Vec<Vec<Fr>> = vec![vec![]; num_selectors];
    let mut witnesses: Vec<Vec<Fr>> = vec![vec![]; num_witnesses];

    for cs in 0..num_constraints {
        let mut cur_selectors: Vec<Fr> = (0..(num_selectors - 1)).map(|_| Fr::rand(rng)).collect();
        let cur_witness: Vec<Fr> = if cs < num_constraints / 4 {
            (0..num_witnesses).map(|_| Fr::rand(rng)).collect()
        } else {
            let row = cs % (num_constraints / 4);
            (0..num_witnesses).map(|i| witnesses[i][row]).collect()
        };
        let mut last_selector = Fr::zero();
        for (index, (q, wit)) in gate.iter().enumerate() {
            if index != num_selectors - 1 {
                let mut cur_monomial = Fr::one();
                cur_monomial = match q {
                    Some(p) => cur_monomial * cur_selectors[*p],
                    None => cur_monomial,
                };
                for wit_index in wit.iter() {
                    cur_monomial *= cur_witness[*wit_index];
                }
                last_selector += cur_monomial;
            } else {
                let mut cur_monomial = Fr::one();
                for wit_index in wit.iter() {
                    cur_monomial *= cur_witness[*wit_index];
                }
                last_selector /= -cur_monomial;
            }
        }
        cur_selectors.push(last_selector);
        for i in 0..num_selectors {
            selectors[i].push(cur_selectors[i]);
        }
        for i in 0..num_witnesses {
            witnesses[i].push(cur_witness[i]);
        }
    }

    let total_num_constraints = num_constraints * Net::n_parties();
    let shift = Net::party_id() * num_constraints;
    let permutation = (0..num_witnesses)
        .map(|witness_idx| {
            let portion_len = num_constraints / 4;
            (0..portion_len)
                .map(|i| {
                    Fr::from_u64(
                        (witness_idx * total_num_constraints + shift + i + portion_len) as u64,
                    )
                    .unwrap()
                })
                .chain((0..portion_len).map(|i| {
                    Fr::from_u64(
                        (witness_idx * total_num_constraints + shift + i + 3 * portion_len) as u64,
                    )
                    .unwrap()
                }))
                .chain((0..portion_len).map(|i| {
                    Fr::from_u64((witness_idx * total_num_constraints + shift + i) as u64).unwrap()
                }))
                .chain((0..portion_len).map(|i| {
                    Fr::from_u64(
                        (witness_idx * total_num_constraints + shift + i + 2 * portion_len) as u64,
                    )
                    .unwrap()
                }))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    (
        witnesses
            .into_iter()
            .map(|vec| Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, vec)))
            .collect(),
        permutation
            .into_iter()
            .map(|vec| Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, vec)))
            .collect(),
        selectors
            .into_iter()
            .map(|vec| Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, vec)))
            .collect(),
    )
}

fn test_combined_check_helper<
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E, Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>>,
>(
    witnesses: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
    perms: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
    selectors: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
    gate: &[(Option<usize>, Vec<usize>)],
    pcs_param: &PCS::ProverParam,
) -> Result<(), PolyIOPErrors> {
    // prover
    let mut transcript = <PolyIOP<E::ScalarField> as CombinedCheck<E, PCS>>::init_transcript();
    transcript.append_message(b"testing", b"initializing transcript for testing")?;

    let to_prove = <PolyIOP<E::ScalarField> as CombinedCheck<E, PCS>>::d_prove_prepare(
        pcs_param,
        &witnesses,
        perms,
        &mut transcript,
    )?;

    let (proof, _, h_poly) = <PolyIOP<E::ScalarField> as CombinedCheck<E, PCS>>::d_prove(
        to_prove,
        witnesses,
        perms,
        selectors,
        gate,
        &mut transcript,
    )?;

    if !Net::am_master() {
        for poly in witnesses {
            d_evaluate_mle(poly, None);
        }
        for poly in perms {
            d_evaluate_mle(poly, None);
        }
        for poly in selectors {
            d_evaluate_mle(poly, None);
        }
        d_evaluate_mle(&h_poly, None);
        return Ok(());
    }

    let (proof, _) = proof.unwrap();

    // verifier
    let mut transcript = <PolyIOP<E::ScalarField> as CombinedCheck<E, PCS>>::init_transcript();
    transcript.append_message(b"testing", b"initializing transcript for testing")?;
    let subclaim =
        <PolyIOP<E::ScalarField> as CombinedCheck<E, PCS>>::verify(&proof, &mut transcript)?;

    let witness_openings = witnesses
        .iter()
        .map(|f| d_evaluate_mle(f, Some(&subclaim.point)).unwrap())
        .collect::<Vec<_>>();
    let perm_openings = perms
        .iter()
        .map(|f| d_evaluate_mle(f, Some(&subclaim.point)).unwrap())
        .collect::<Vec<_>>();
    let selector_openings = selectors
        .iter()
        .map(|f| d_evaluate_mle(f, Some(&subclaim.point)).unwrap())
        .collect::<Vec<_>>();
    let h_opening = d_evaluate_mle(&h_poly, Some(&subclaim.point)).unwrap();

    <PolyIOP<E::ScalarField> as CombinedCheck<E, PCS>>::check_openings(
        &subclaim,
        &witness_openings,
        &perm_openings,
        &selector_openings,
        &h_opening,
        gate,
    )
}

fn test_combined_check(nv: usize) -> Result<(), PolyIOPErrors> {
    let mut rng = test_rng();

    let num_party_vars = Net::n_parties().log_2();
    let srs = DeMkzg::<Bn254>::gen_srs_for_testing(&mut rng, nv + num_party_vars)?;
    let (pcs_param, _) = DeMkzg::<Bn254>::trim(&srs, None, Some(nv + num_party_vars))?;

    let gate = vec![
        (Some(0), vec![0]),
        (Some(1), vec![1]),
        (Some(2), vec![2]),
        (Some(3), vec![3]),
        (Some(4), vec![0, 1]),
        (Some(5), vec![2, 3]),
        (Some(6), vec![0, 0, 0, 0, 0]),
        (Some(7), vec![1, 1, 1, 1, 1]),
        (Some(8), vec![2, 2, 2, 2, 2]),
        (Some(9), vec![3, 3, 3, 3, 3]),
        (Some(10), vec![0, 1, 2, 3]),
        (Some(11), vec![4]),
        (Some(12), vec![]),
    ];
    let (witnesses, perms, selectors) = generate_polys(5, 13, nv, &gate, &mut rng);
    test_combined_check_helper::<Bn254, DeMkzg<Bn254>>(
        &witnesses, &perms, &selectors, &gate, &pcs_param,
    )
}

fn main() {
    common::network_run(|| {
        test_combined_check(5).unwrap();
    });
}

mod common;

use arithmetic::{
    identity_permutation_mle, identity_permutation_mles, math::Math, random_permutation_u64,
};
use ark_bls12_381::{Bls12_381, Fr};
use ark_ec::pairing::Pairing;
use ark_ff::{PrimeField, UniformRand};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use common::{d_evaluate_mle, test_rng};
use itertools::{izip, zip_eq, MultiUnzip};
use rand_core::RngCore;
use std::{mem::take, sync::Arc};
use subroutines::{
    DeDory, DeDorySRS, PermutationCheck, PolyIOP, PolyIOPErrors, PolynomialCommitmentScheme,
};
use transcript::IOPTranscript;

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

#[cfg(not(feature = "rational_sumcheck_piop"))]
use subroutines::{BatchedDenseGrandProduct, BatchedGrandProduct};

#[cfg(not(feature = "rational_sumcheck_piop"))]
fn test_layered_circuit() {
    const LAYER_SIZE: usize = 1 << 8;
    const BATCH_SIZE: usize = 4;
    let mut rng = test_rng();
    let leaves: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
        std::iter::repeat_with(|| Fr::rand(&mut rng))
            .take(LAYER_SIZE)
            .collect()
    })
    .take(BATCH_SIZE)
    .collect();

    let expected_claims = leaves
        .iter()
        .map(|leave| leave.iter().product::<Fr>())
        .collect::<Vec<_>>();
    let all_expected_claims = Net::send_to_master(&expected_claims);

    let polys = leaves
        .iter()
        .map(|leave| {
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                LAYER_SIZE.log_2(),
                leave.clone(),
            ))
        })
        .collect::<Vec<_>>();

    let (mut batched_circuit, mut final_circuit) =
        <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<Fr>>::d_construct(leaves);
    let mut transcript: IOPTranscript<Fr> = IOPTranscript::new(b"test_transcript");

    // I love the rust type system

    let proof_ret =
        <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<Fr>>::d_prove_grand_product(
            &mut batched_circuit,
            final_circuit.as_mut(),
            &mut transcript,
        );

    if Net::am_master() {
        let (proof, r_prover) = proof_ret.unwrap();
        let claims = <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<Fr>>::claims(
            &final_circuit.unwrap(),
        );
        let all_expected_claims = all_expected_claims.unwrap();
        let expected_claims = (0..all_expected_claims[0].len())
            .map(|leave_id| {
                all_expected_claims
                    .iter()
                    .map(|leave| leave[leave_id])
                    .product::<Fr>()
            })
            .collect::<Vec<_>>();
        assert_eq!(claims, expected_claims);

        let mut transcript: IOPTranscript<Fr> = IOPTranscript::new(b"test_transcript");
        let (final_claims, r_verifier) =
            BatchedDenseGrandProduct::verify_grand_product(&proof, &claims, &mut transcript);
        assert_eq!(r_prover, r_verifier);
        for (claim, poly) in zip_eq(final_claims.iter(), polys.iter()) {
            assert_eq!(d_evaluate_mle(poly, Some(&r_verifier)).unwrap(), *claim);
        }
    } else {
        for poly in &polys {
            d_evaluate_mle(poly, None);
        }
    }
}

fn test_permutation_check_helper<
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E, Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>>,
>(
    fxs: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
    gxs: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
    perms: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
    pcs_param: &PCS::ProverParam,
) -> Result<(), PolyIOPErrors> {
    // Distribute witnesses
    let num_party_vars = Net::n_parties().log_2();
    let (fxs_polys, gxs_polys, perms_polys) = if Net::am_master() {
        let messages = (0..Net::n_parties())
            .map(|i| {
                let (fxs, gxs, perms): (Vec<_>, Vec<_>, Vec<_>) = izip!(fxs, gxs, perms)
                    .map(|(fx, gx, perm)| {
                        debug_assert_eq!(fx.num_vars, gx.num_vars);
                        debug_assert_eq!(gx.num_vars, perm.num_vars);
                        let num_evals = fx.evaluations.len() / Net::n_parties();

                        (
                            DenseMultilinearExtension::from_evaluations_vec(
                                fx.num_vars - num_party_vars,
                                fx.evaluations[i * num_evals..(i + 1) * num_evals].to_vec(),
                            ),
                            DenseMultilinearExtension::from_evaluations_vec(
                                gx.num_vars - num_party_vars,
                                gx.evaluations[i * num_evals..(i + 1) * num_evals].to_vec(),
                            ),
                            DenseMultilinearExtension::from_evaluations_vec(
                                perm.num_vars - num_party_vars,
                                perm.evaluations[i * num_evals..(i + 1) * num_evals].to_vec(),
                            ),
                        )
                    })
                    .multiunzip();
                (fxs, gxs, perms)
            })
            .collect::<Vec<_>>();
        Net::recv_from_master(Some(messages))
    } else {
        Net::recv_from_master(None)
    };

    let wrap_in_arc =
        |mut x: Vec<DenseMultilinearExtension<E::ScalarField>>| -> Vec<Arc<DenseMultilinearExtension<E::ScalarField>>> {
            x.iter_mut().map(|poly| Arc::new(take(poly))).collect()
        };
    let (fxs_polys, gxs_polys, perms_polys) = (
        wrap_in_arc(fxs_polys),
        wrap_in_arc(gxs_polys),
        wrap_in_arc(perms_polys),
    );
    let (fxs, gxs, perms) = (&fxs_polys, &gxs_polys, &perms_polys);

    // prover
    let mut transcript = <PolyIOP<E::ScalarField> as PermutationCheck<E, PCS>>::init_transcript();
    transcript.append_message(b"testing", b"initializing transcript for testing")?;
    #[cfg(feature = "rational_sumcheck_piop")]
    let (to_prove, claims) =
        <PolyIOP<E::ScalarField> as PermutationCheck<E, PCS>>::d_prove_prepare(
            pcs_param,
            fxs,
            gxs,
            perms,
            &mut transcript,
        )?;

    #[cfg(feature = "rational_sumcheck_piop")]
    let proof_ret = <PolyIOP<E::ScalarField> as PermutationCheck<E, PCS>>::d_prove(
        pcs_param,
        to_prove,
        claims,
        &mut transcript,
    )?;

    #[cfg(not(feature = "rational_sumcheck_piop"))]
    let proof_ret = <PolyIOP<E::ScalarField> as PermutationCheck<E, PCS>>::d_prove(
        fxs,
        gxs,
        perms,
        &mut transcript,
    )?;

    #[cfg(feature = "rational_sumcheck_piop")]
    let (proof_ret, _, h_polys) = proof_ret;

    if Net::am_master() {
        let (proof, _) = proof_ret.unwrap();

        // verifier
        let mut transcript =
            <PolyIOP<E::ScalarField> as PermutationCheck<E, PCS>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;
        let perm_check_sub_claim =
            <PolyIOP<E::ScalarField> as PermutationCheck<E, PCS>>::verify(&proof, &mut transcript)?;

        Net::recv_from_master_uniform(Some(perm_check_sub_claim.clone()));

        let mut f_openings = vec![];
        let mut g_openings = vec![];
        #[cfg(feature = "rational_sumcheck_piop")]
        let mut h_openings = vec![];
        let mut perm_openings = vec![];
        let mut offset = 0;

        #[cfg(feature = "rational_sumcheck_piop")]
        {
            for (i, (subclaim, len)) in perm_check_sub_claim.subclaims.iter().enumerate() {
                let mut f_evals = fxs[offset..offset + len]
                    .iter()
                    .map(|f| d_evaluate_mle(f, Some(&subclaim.sumcheck_point)).unwrap())
                    .collect::<Vec<_>>();
                let mut g_evals = gxs[offset..offset + len]
                    .iter()
                    .map(|g| d_evaluate_mle(g, Some(&subclaim.sumcheck_point)).unwrap())
                    .collect::<Vec<_>>();
                let h_eval = d_evaluate_mle(&h_polys[i], Some(&subclaim.sumcheck_point)).unwrap();
                let mut perm_evals = perms[offset..offset + len]
                    .iter()
                    .map(|perm| {
                        d_evaluate_mle(perm, Some(&subclaim.sumcheck_point)).unwrap()
                    })
                    .collect::<Vec<_>>();

                f_openings.append(&mut f_evals);
                g_openings.append(&mut g_evals);
                h_openings.push(h_eval);
                perm_openings.append(&mut perm_evals);
                offset += len;
            }

            return <PolyIOP<E::ScalarField> as PermutationCheck<E, PCS>>::check_openings(
                &perm_check_sub_claim,
                &f_openings,
                &g_openings,
                &h_openings,
                &perm_openings,
            );
        }

        #[cfg(not(feature = "rational_sumcheck_piop"))]
        {
            for subclaim in perm_check_sub_claim.subclaims.iter() {
                let mut f_evals = fxs[offset..offset + subclaim.len]
                    .iter()
                    .map(|f| d_evaluate_mle(f, Some(&subclaim.point)).unwrap())
                    .collect::<Vec<_>>();
                let mut g_evals = gxs[offset..offset + subclaim.len]
                    .iter()
                    .map(|g| d_evaluate_mle(g, Some(&subclaim.point)).unwrap())
                    .collect::<Vec<_>>();
                let mut perm_evals = perms[offset..offset + subclaim.len]
                    .iter()
                    .map(|perm| d_evaluate_mle(perm, Some(&subclaim.point)).unwrap())
                    .collect::<Vec<_>>();

                f_openings.append(&mut f_evals);
                g_openings.append(&mut g_evals);
                perm_openings.append(&mut perm_evals);
                offset += subclaim.len;
            }

            <PolyIOP<E::ScalarField> as PermutationCheck<E, PCS>>::check_openings(
                &perm_check_sub_claim,
                &f_openings,
                &g_openings,
                &perm_openings,
            )
        }
    } else {
        let perm_check_sub_claim: <PolyIOP<E::ScalarField> as PermutationCheck<E, PCS>>::PermutationCheckSubClaim =
            Net::recv_from_master_uniform(None);

        let mut offset = 0;
        #[cfg(feature = "rational_sumcheck_piop")]
        for (i, (_, len)) in perm_check_sub_claim.subclaims.iter().enumerate() {
            for f in &fxs[offset..offset + len] {
                d_evaluate_mle(f, None);
            }
            for g in &gxs[offset..offset + len] {
                d_evaluate_mle(g, None);
            }
            d_evaluate_mle(&h_polys[i], None);
            for perm in &perms[offset..offset + len] {
                d_evaluate_mle(perm, None);
            }
            offset += len;
        }

        #[cfg(not(feature = "rational_sumcheck_piop"))]
        for subclaim in perm_check_sub_claim.subclaims.iter() {
            for f in &fxs[offset..offset + subclaim.len] {
                d_evaluate_mle(f, None);
            }
            for g in &gxs[offset..offset + subclaim.len] {
                d_evaluate_mle(g, None);
            }
            for perm in &perms[offset..offset + subclaim.len] {
                d_evaluate_mle(perm, None);
            }
            offset += subclaim.len;
        }

        Ok(())
    }
}

fn generate_polys<R: RngCore>(
    nv: &[usize],
    rng: &mut R,
) -> Vec<Arc<DenseMultilinearExtension<Fr>>> {
    nv.iter()
        .map(|x| Arc::new(DenseMultilinearExtension::rand(*x, rng)))
        .collect()
}

fn test_permutation_check(
    nv: Vec<usize>,
    id_perms: Vec<Arc<DenseMultilinearExtension<Fr>>>,
) -> Result<(), PolyIOPErrors> {
    let mut rng = test_rng();

    let pcs_srs = if Net::am_master() {
        let pcs_srs = DeDory::<Bls12_381>::gen_srs_for_testing(&mut rng, 8)?;
        let pp = match &pcs_srs {
            DeDorySRS::Unprocessed(pp) => pp,
            _ => panic!("Unexpected processed"),
        };
        Net::recv_from_master_uniform(Some(pp.clone()));
        pcs_srs
    } else {
        DeDorySRS::Unprocessed(Net::recv_from_master_uniform(None))
    };

    let (prover_param, _) = DeDory::<Bls12_381>::trim(&pcs_srs, None, Some(8))?;

    if !Net::am_master() {
        // We'll receive our share from master
        for _ in 0..2 {
            test_permutation_check_helper::<Bls12_381, DeDory<Bls12_381>>(
                &[],
                &[],
                &[],
                &prover_param,
            )?;
        }
        return Ok(());
    }

    {
        // good path: (w1, w2) is a permutation of (w1, w2) itself under the identify
        // map
        let ws = generate_polys(&nv, &mut rng);
        // perms is the identity map
        test_permutation_check_helper::<Bls12_381, DeDory<Bls12_381>>(
            &ws,
            &ws,
            &id_perms,
            &prover_param,
        )?;
    }

    {
        let fs = generate_polys(&nv, &mut rng);

        let size0 = nv[0].pow2();

        let perm = random_permutation_u64(nv[0].pow2() + nv[1].pow2(), &mut rng);
        let perms = vec![
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                nv[0],
                perm[..size0]
                    .iter()
                    .map(|x| Fr::from_u64(*x).unwrap())
                    .collect(),
            )),
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                nv[1],
                perm[size0..]
                    .iter()
                    .map(|x| Fr::from_u64(*x).unwrap())
                    .collect(),
            )),
        ];

        let get_f = |index: usize| {
            if index < size0 {
                fs[0].evaluations[index]
            } else {
                fs[1].evaluations[index - size0]
            }
        };

        let g_evals = (
            (0..size0)
                .map(|x| get_f(perm[x] as usize))
                .collect::<Vec<_>>(),
            (size0..size0 + nv[1].pow2())
                .map(|x| get_f(perm[x] as usize))
                .collect::<Vec<_>>(),
        );
        let gs = vec![
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                nv[0], g_evals.0,
            )),
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                nv[1], g_evals.1,
            )),
        ];

        test_permutation_check_helper::<Bls12_381, DeDory<Bls12_381>>(
            &fs,
            &gs,
            &perms,
            &prover_param,
        )?;
    }

    Ok(())
}

fn test_trivial_polynomial() -> Result<(), PolyIOPErrors> {
    let id_perms = identity_permutation_mles(3, 2);
    test_permutation_check(vec![3, 3], id_perms)
}

fn test_normal_polynomial() -> Result<(), PolyIOPErrors> {
    let id_perms = identity_permutation_mles(7, 2);
    test_permutation_check(vec![7, 7], id_perms)
}

fn test_different_lengths() -> Result<(), PolyIOPErrors> {
    let id_perms = vec![
        identity_permutation_mle(0, 7),
        identity_permutation_mle(128, 6),
    ];
    test_permutation_check(vec![7, 6], id_perms)
}

fn main() {
    common::network_run(|| {
        #[cfg(not(feature = "rational_sumcheck_piop"))]
        test_layered_circuit();
        // test_trivial_polynomial().unwrap();
        test_normal_polynomial().unwrap();
        test_different_lengths().unwrap();
    });
}

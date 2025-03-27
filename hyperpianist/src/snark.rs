// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

use crate::{
    errors::HyperPlonkErrors,
    lookup::HyperPlonkLookupPlugin,
    prelude::HyperPlonkParams,
    structs::{HyperPlonkIndex, HyperPlonkProof, HyperPlonkProvingKey, HyperPlonkVerifyingKey},
    utils::{
        build_f, eval_f, prover_sanity_check, PcsDynamicAccumulator, PcsDynamicOpenings,
        PcsDynamicVerifier,
    },
    witness::WitnessColumn,
    HyperPlonkSNARK,
};
use arithmetic::{evaluate_opt, math::Math, VPAuxInfo};
use ark_ec::pairing::Pairing;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::{end_timer, log2, start_timer, Zero};
use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};
use itertools::izip;
use lazy_static::lazy_static;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
use rayon::{iter::IntoParallelRefIterator, ThreadPoolBuilder};
use std::{iter::zip, marker::PhantomData, mem::take, sync::Arc};
use subroutines::{
    pcs::prelude::PolynomialCommitmentScheme,
    poly_iop::{
        prelude::{PermutationCheck, ZeroCheck},
        PolyIOP,
    },
    BatchProof,
};
use transcript::IOPTranscript;

// lazy_static! {
//     static ref ZERO_CHECK_POOL: rayon::ThreadPool = {
//         ThreadPoolBuilder::new()
//             .num_threads(rayon::current_num_threads() / 2)
//             .build()
//             .unwrap()
//     };
//     static ref PERM_CHECK_POOL: rayon::ThreadPool = {
//         ThreadPoolBuilder::new()
//             .num_threads(rayon::current_num_threads() / 2)
//             .build()
//             .unwrap()
//     };
// }

impl<E, PCS, Lookup> HyperPlonkSNARK<E, PCS, Lookup> for PolyIOP<E::ScalarField>
where
    E: Pairing,
    // Ideally we want to access polynomial as PCS::Polynomial, instead of instantiating it here.
    // But since PCS::Polynomial can be both univariate or multivariate in our implementation
    // we cannot bound PCS::Polynomial with a property trait bound.
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        Point = Vec<E::ScalarField>,
        Evaluation = E::ScalarField,
        BatchProof = BatchProof<E, PCS>,
    >,
    Lookup: HyperPlonkLookupPlugin<E, PCS, Transcript = IOPTranscript<E::ScalarField>>,
{
    type Index = HyperPlonkIndex<E::ScalarField>;
    type ProvingKey = HyperPlonkProvingKey<E, PCS, Lookup>;
    type VerifyingKey = HyperPlonkVerifyingKey<E, PCS, Lookup>;
    type Proof = HyperPlonkProof<E, Self, Self, PCS, Lookup>;

    fn preprocess(
        index: &Self::Index,
        pcs_srs: &PCS::SRS,
    ) -> Result<(Self::ProvingKey, Self::VerifyingKey), HyperPlonkErrors> {
        let supported_ml_degree = index.max_num_variables::<E, PCS, Lookup>();

        // extract PCS prover and verifier keys from SRS
        let (pcs_prover_param, pcs_verifier_param) =
            PCS::trim(pcs_srs, None, Some(supported_ml_degree))?;

        // build permutation oracles
        let mut permutation_oracles = vec![];
        let mut perm_comms = vec![];
        let mut perm_advices = vec![];

        let num_witness_columns = vec![
            vec![index.params.gate_func.num_witness_columns()],
            Lookup::num_witness_columns(),
        ]
        .concat();
        let num_constraints = vec![
            &[index.params.num_constraints][..],
            &index.params.num_lookup_constraints,
        ]
        .concat();

        if num_witness_columns.len() != num_constraints.len() {
            return Err(HyperPlonkErrors::InvalidParameters(format!(
                "num_witness_columns.len() = {}, num_constraints.len() = {}",
                num_witness_columns.len(),
                num_constraints.len()
            )));
        }

        let mut current_index = 0;
        for (&witnesses, &constraints) in zip(num_witness_columns.iter(), num_constraints.iter()) {
            if constraints == 0 {
                // witnesses = 0
                continue;
            }
            let num_vars = log2(constraints) as usize;
            let length = num_vars.pow2();
            for _ in 0..witnesses {
                let perm_oracle = Arc::new(DenseMultilinearExtension::from_evaluations_slice(
                    num_vars,
                    &index.permutation[current_index..current_index + length],
                ));
                let (perm_comm, perm_advice) = PCS::commit(&pcs_prover_param, &perm_oracle)?;
                permutation_oracles.push(perm_oracle);
                perm_comms.push(perm_comm);
                perm_advices.push(perm_advice);
                current_index += length;
            }
        }

        // build selector oracles and commit to it
        let selector_oracles: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>> = index
            .selectors
            .iter()
            .map(|s| Arc::new(DenseMultilinearExtension::from(s)))
            .collect();

        let (selector_commitments, selector_advices): (Vec<_>, Vec<_>) = selector_oracles
            .par_iter()
            .map(|poly| PCS::commit(&pcs_prover_param, poly).unwrap())
            .unzip();

        let lookup_preprocessing = Lookup::preprocess();

        Ok((
            Self::ProvingKey {
                params: index.params.clone(),
                permutation_oracles,
                selector_oracles,
                selector_commitments: selector_commitments.clone(),
                permutation_commitments: perm_comms.clone(),
                selector_advices,
                permutation_advices: perm_advices,
                pcs_param: pcs_prover_param,
                lookup_preprocessing: lookup_preprocessing.clone(),
            },
            Self::VerifyingKey {
                params: index.params.clone(),
                pcs_param: pcs_verifier_param,
                selector_commitments,
                perm_commitments: perm_comms,
                lookup_preprocessing,
                num_party_vars: 0,
            },
        ))
    }

    fn d_preprocess(
        index: &Self::Index,
        pcs_srs: &PCS::SRS,
    ) -> Result<(Self::ProvingKey, Option<Self::VerifyingKey>), HyperPlonkErrors> {
        let supported_ml_degree =
            index.max_num_variables::<E, PCS, Lookup>() + Net::n_parties().log_2();

        // extract PCS prover and verifier keys from SRS
        let (pcs_prover_param, pcs_verifier_param) =
            PCS::trim(pcs_srs, None, Some(supported_ml_degree))?;

        // build permutation oracles
        let mut permutation_oracles = vec![];
        let mut perm_comms = vec![];
        let mut perm_advices = vec![];

        let num_witness_columns = vec![
            vec![index.params.gate_func.num_witness_columns()],
            Lookup::num_witness_columns(),
        ]
        .concat();
        let num_constraints = vec![
            &[index.params.num_constraints][..],
            &index.params.num_lookup_constraints,
        ]
        .concat();

        if num_witness_columns.len() != num_constraints.len() {
            return Err(HyperPlonkErrors::InvalidParameters(format!(
                "num_witness_columns.len() = {}, num_constraints.len() = {}",
                num_witness_columns.len(),
                num_constraints.len()
            )));
        }

        let mut current_index = 0;
        for (&witnesses, &constraints) in zip(num_witness_columns.iter(), num_constraints.iter()) {
            if constraints == 0 {
                // witnesses = 0
                continue;
            }
            let num_vars = log2(constraints) as usize;
            let length = num_vars.pow2();
            for _ in 0..witnesses {
                let perm_oracle = Arc::new(DenseMultilinearExtension::from_evaluations_slice(
                    num_vars,
                    &index.permutation[current_index..current_index + length],
                ));
                let (perm_comm, perm_advice) = PCS::d_commit(&pcs_prover_param, &perm_oracle)?;
                permutation_oracles.push(perm_oracle);
                if Net::am_master() {
                    perm_comms.push(perm_comm.unwrap());
                }
                perm_advices.push(perm_advice);
                current_index += length;
            }
        }

        // build selector oracles and commit to it
        let selector_oracles: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>> = index
            .selectors
            .iter()
            .map(|s| Arc::new(DenseMultilinearExtension::from(s)))
            .collect();

        let (selector_commitments, selector_advices): (Vec<_>, Vec<_>) = selector_oracles
            .iter()
            .map(|poly| PCS::d_commit(&pcs_prover_param, poly).unwrap())
            .unzip();

        let lookup_preprocessing = Lookup::preprocess();

        if Net::am_master() {
            let selector_commitments = selector_commitments
                .iter()
                .map(|x| x.clone().unwrap())
                .collect::<Vec<_>>();
            let vk_params = HyperPlonkParams {
                num_constraints: index.params.num_constraints * Net::n_parties(),
                num_lookup_constraints: index
                    .params
                    .num_lookup_constraints
                    .iter()
                    .map(|x| x * Net::n_parties())
                    .collect(),
                num_pub_input: index.params.num_pub_input * Net::n_parties(),
                gate_func: index.params.gate_func.clone(),
            };
            Ok((
                Self::ProvingKey {
                    params: index.params.clone(),
                    permutation_oracles,
                    selector_oracles,
                    selector_commitments: selector_commitments.clone(),
                    permutation_commitments: perm_comms.clone(),
                    selector_advices,
                    permutation_advices: perm_advices,
                    pcs_param: pcs_prover_param,
                    lookup_preprocessing: lookup_preprocessing.clone(),
                },
                Some(Self::VerifyingKey {
                    params: vk_params,
                    pcs_param: pcs_verifier_param,
                    selector_commitments,
                    perm_commitments: perm_comms,
                    lookup_preprocessing,
                    num_party_vars: Net::n_parties().log_2(),
                }),
            ))
        } else {
            Ok((
                Self::ProvingKey {
                    params: index.params.clone(),
                    permutation_oracles,
                    selector_oracles,
                    selector_commitments: vec![],
                    permutation_commitments: vec![],
                    selector_advices,
                    permutation_advices: perm_advices,
                    pcs_param: pcs_prover_param,
                    lookup_preprocessing: lookup_preprocessing.clone(),
                },
                None,
            ))
        }
    }

    /// Generate HyperPlonk SNARK proof.
    ///
    /// Inputs:
    /// - `pk`: circuit proving key
    /// - `pub_input`: online public input of length 2^\ell
    /// - `witness`: witness assignment of length 2^n
    /// Outputs:
    /// - The HyperPlonk SNARK proof.
    ///
    /// Steps:
    ///
    /// 1. Commit Witness polynomials `w_i(x)` and append commitment to
    /// transcript
    ///
    /// 2. Run ZeroCheck on
    ///
    ///     `f(q_0(x),...q_l(x), w_0(x),...w_d(x))`  
    ///
    /// where `f` is the constraint polynomial i.e.,
    /// ```ignore
    ///     f(q_l, q_r, q_m, q_o, w_a, w_b, w_c)
    ///     = q_l w_a(x) + q_r w_b(x) + q_m w_a(x)w_b(x) - q_o w_c(x)
    /// ```
    /// in vanilla plonk, and obtain a ZeroCheckSubClaim
    ///
    /// 3. Run permutation check on `\{w_i(x)\}` and `permutation_oracle`, and
    /// obtain a PermCheckSubClaim.
    ///
    /// 4. Generate evaluations and corresponding proofs
    /// - 4.1. (deferred) batch opening prod(x) at
    ///   - [0, perm_check_point]
    ///   - [1, perm_check_point]
    ///   - [perm_check_point, 0]
    ///   - [perm_check_point, 1]
    ///   - [1,...1, 0]
    ///
    /// - 4.2. permutation check evaluations and proofs
    ///   - 4.2.1. (deferred) wi_poly(perm_check_point)
    ///
    /// - 4.3. zero check evaluations and proofs
    ///   - 4.3.1. (deferred) wi_poly(zero_check_point)
    ///   - 4.3.2. (deferred) selector_poly(zero_check_point)
    ///
    /// - 4.4. public input consistency checks
    ///   - pi_poly(r_pi) where r_pi is sampled from transcript
    ///
    /// - 5. deferred batch opening
    fn prove(
        pk: &Self::ProvingKey,
        pub_input: &[E::ScalarField],
        witnesses: &[WitnessColumn<E::ScalarField>],
        ops: &Lookup::Ops,
    ) -> Result<Self::Proof, HyperPlonkErrors> {
        Net::set_channel_id(0);

        let start = start_timer!(|| "hyperplonk proving");
        let mut transcript = IOPTranscript::<E::ScalarField>::new(b"hyperplonk");

        prover_sanity_check(&pk.params, pub_input, witnesses)?;

        // witness assignment of length 2^n
        let num_vars = pk.params.num_variables();

        // online public input of length 2^\ell
        let ell = log2(pk.params.num_pub_input) as usize;

        // We use accumulators to store the polynomials and their eval points.
        // They are batch opened at a later stage.
        let mut pcs_acc = PcsDynamicAccumulator::<E, PCS>::new();

        // =======================================================================
        // 1. Commit Witness polynomials `w_i(x)` and append commitment to
        // transcript
        // =======================================================================
        let step = start_timer!(|| "commit witnesses");

        let mut witness_polys: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>> = witnesses
            .iter()
            .map(|w| Arc::new(DenseMultilinearExtension::from(w)))
            .collect();
        witness_polys.append(&mut Lookup::construct_witnesses(ops));

        let (witness_commits, witness_advices): (Vec<_>, Vec<_>) = witness_polys
            .par_iter()
            .map(|x| PCS::commit(&pk.pcs_param, x).unwrap())
            .unzip();
        for w_com in witness_commits.iter() {
            transcript.append_serializable_element(b"w", w_com)?;
        }

        end_timer!(step);

        let num_gate_witnesses = pk.params.gate_func.num_witness_columns();

        let zero_check_proof = {
            let mut transcript = transcript.clone();

            // =======================================================================
            // 2 Run ZeroCheck on
            //
            //     `f(q_0(x),...q_l(x), w_0(x),...w_d(x))`
            //
            // where `f` is the constraint polynomial i.e.,
            //
            //     f(q_l, q_r, q_m, q_o, w_a, w_b, w_c)
            //     = q_l w_a(x) + q_r w_b(x) + q_m w_a(x)w_b(x) - q_o w_c(x)
            //
            // in vanilla plonk, and obtain a ZeroCheckSubClaim
            // =======================================================================
            let step = start_timer!(|| "ZeroCheck on f");

            let fx = build_f(
                &pk.params.gate_func,
                pk.params.num_variables(),
                &pk.selector_oracles,
                &witness_polys[..num_gate_witnesses],
            )?;

            let zero_check_proof = <Self as ZeroCheck<E::ScalarField>>::prove(fx, &mut transcript)?;
            end_timer!(step);

            zero_check_proof
        };

        let (lookup_proof, lookup_opening_points) = {
            let mut transcript = transcript.clone();

            let step = start_timer!(|| "LookupCheck");

            // =======================================================================
            // 3. Construct and perform lookup checks
            // =======================================================================
            let (lookup_proof, lookup_opening_points) = Lookup::prove(
                &pk.lookup_preprocessing,
                &pk.pcs_param,
                ops,
                &mut transcript,
            );
            end_timer!(step);
            (lookup_proof, lookup_opening_points)
        };

        let perm_check_ret = {
            let mut transcript = transcript.clone();

            // =======================================================================
            // 3. Run permutation check on `\{w_i(x)\}` and `permutation_oracle`, and
            // obtain a PermCheckSubClaim.
            // =======================================================================
            let step = start_timer!(|| "Permutation check on w_i(x)");

            let perm_check_ret = <Self as PermutationCheck<E, PCS>>::prove(
                #[cfg(feature = "rational_sumcheck_piop")]
                &pk.pcs_param,
                &witness_polys,
                &witness_polys,
                &pk.permutation_oracles,
                &mut transcript,
            )?;

            end_timer!(step);

            perm_check_ret
        };

        #[cfg(feature = "rational_sumcheck_piop")]
        let (perm_check_proof, perm_h_advices, perm_check_points, perm_h_polys) = perm_check_ret;

        #[cfg(not(feature = "rational_sumcheck_piop"))]
        let (perm_check_proof, perm_check_points) = perm_check_ret;

        // =======================================================================
        // 4. Generate evaluations and corresponding proofs
        // - permcheck
        //  1. (deferred) batch opening perms(x) at
        //   - [perm_check_point]
        //  2. (deferred) batch opening witness_i(x) at
        //   - [perm_check_point]
        //
        // - zero check evaluations and proofs
        //   - 4.3.1. (deferred) wi_poly(zero_check_point)
        //   - 4.3.2. (deferred) selector_poly(zero_check_point)
        //
        // - 4.4. (deferred) public input consistency checks
        //   - pi_poly(r_pi) where r_pi is sampled from transcript
        // =======================================================================
        let step = start_timer!(|| "openings setup");

        // perms(x)'s points
        let mut last_nv = pk.permutation_oracles[0].num_vars();
        let mut index = 0;
        for (perm, advice) in zip(&pk.permutation_oracles, &pk.permutation_advices) {
            if perm.num_vars() != last_nv {
                last_nv = perm.num_vars();
                index += 1;
            }
            pcs_acc.insert_poly_and_points(perm, advice, &perm_check_points[index]);
        }

        // witnesses' points
        // TODO: refactor so it remains correct even if the order changed
        let mut last_nv = witness_polys[0].num_vars();
        let mut index = 0;
        for (wpoly, wadvice) in zip(&witness_polys, &witness_advices) {
            if wpoly.num_vars() != last_nv {
                last_nv = wpoly.num_vars();
                index += 1;
            }
            pcs_acc.insert_poly_and_points(wpoly, wadvice, &perm_check_points[index]);
        }

        #[cfg(feature = "rational_sumcheck_piop")]
        for (index, (poly, advice)) in zip(&perm_h_polys, &perm_h_advices).enumerate() {
            pcs_acc.insert_poly_and_points(poly, advice, &perm_check_points[index]);
        }

        for (wpoly, wadvice) in zip(&witness_polys[..num_gate_witnesses], &witness_advices) {
            pcs_acc.insert_poly_and_points(wpoly, wadvice, &zero_check_proof.point);
        }
        for (wpoly, wadvice, point) in izip!(
            witness_polys[num_gate_witnesses..].iter(),
            witness_advices[num_gate_witnesses..].iter(),
            lookup_opening_points.witness_openings.iter()
        ) {
            pcs_acc.insert_poly_and_points(wpoly, wadvice, point);
        }

        //   - 4.3.2. (deferred) selector_poly(zero_check_point)
        for (poly, advice) in zip(&pk.selector_oracles, &pk.selector_advices) {
            pcs_acc.insert_poly_and_points(poly, advice, &zero_check_proof.point);
        }

        // - 4.4. public input consistency checks
        //   - pi_poly(r_pi) where r_pi is sampled from transcript
        let r_pi = transcript.get_and_append_challenge_vectors(b"r_pi", ell)?;
        // padded with zeros
        let r_pi_padded = [r_pi, vec![E::ScalarField::zero(); num_vars - ell]].concat();
        // Evaluate witness_poly[0] at r_pi||0s which is equal to public_input evaluated
        // at r_pi. Assumes that public_input is a power of 2
        pcs_acc.insert_poly_and_points(&witness_polys[0], &witness_advices[0], &r_pi_padded);

        // - 5. lookup check points
        for (poly, advice_idx, point) in lookup_opening_points.regular_openings.iter() {
            pcs_acc.insert_poly_and_points(
                poly,
                &lookup_opening_points.regular_advices[*advice_idx],
                point,
            );
        }
        end_timer!(step);

        let step = start_timer!(|| "evaluate all");
        pcs_acc.evaluate_all();
        end_timer!(step);

        // =======================================================================
        // 5. deferred batch opening
        // =======================================================================
        let step = start_timer!(|| "deferred batch openings prod(x)");
        drop(witness_polys);
        drop(lookup_opening_points);
        let batch_openings = pcs_acc.multi_open(&pk.pcs_param, &mut transcript)?;
        end_timer!(step);

        end_timer!(start);

        Ok(HyperPlonkProof {
            // PCS commit for witnesses
            witness_commits,
            // batch_openings,
            batch_openings,
            // =======================================================================
            // IOP proofs
            // =======================================================================
            // the custom gate zerocheck proof
            zero_check_proof,
            // the permutation check proof for copy constraints
            perm_check_proof,
            lookup_proof,
        })
    }

    fn d_prove(
        pk: &Self::ProvingKey,
        pub_input: &[E::ScalarField],
        witnesses: &[WitnessColumn<E::ScalarField>],
        ops: &Lookup::Ops,
    ) -> Result<Option<Self::Proof>, HyperPlonkErrors> {
        let start = start_timer!(|| "hyperplonk proving");
        let mut transcript = IOPTranscript::<E::ScalarField>::new(b"hyperplonk");

        prover_sanity_check(&pk.params, pub_input, witnesses)?;

        // witness assignment of length 2^n
        let num_vars = pk.params.num_variables();
        let num_party_vars = Net::n_parties().log_2();

        // online public input of length 2^\ell
        let ell = log2(pk.params.num_pub_input) as usize;

        // We use accumulators to store the polynomials and their eval points.
        // They are batch opened at a later stage.
        let mut pcs_acc = PcsDynamicAccumulator::<E, PCS>::new();

        // =======================================================================
        // 1. Commit Witness polynomials `w_i(x)` and append commitment to
        // transcript
        // =======================================================================
        let step = start_timer!(|| "commit witnesses");

        let mut witness_polys: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>> = witnesses
            .iter()
            .map(|w| Arc::new(DenseMultilinearExtension::from(w)))
            .collect();
        witness_polys.append(&mut Lookup::construct_witnesses(ops));

        let (mut witness_commits, witness_advices): (Vec<_>, Vec<_>) =
            PCS::batch_d_commit(&pk.pcs_param, &witness_polys).unwrap();
        if Net::am_master() {
            for w_com in witness_commits.iter() {
                transcript.append_serializable_element(b"w", w_com.as_ref().unwrap())?;
            }
        }

        end_timer!(step);

        let mut perm_transcript = transcript.clone();

        #[cfg(feature = "rational_sumcheck_piop")]
        let (perm_to_prove, perm_claims) = <Self as PermutationCheck<E, PCS>>::d_prove_prepare(
            &pk.pcs_param,
            &witness_polys,
            &witness_polys,
            &pk.permutation_oracles,
            &mut perm_transcript,
        )?;

        let num_gate_witnesses = pk.params.gate_func.num_witness_columns();

        let ((zero_check_proof, (lookup_proof, lookup_opening_points)), perm_check_ret) =
            rayon::join(
                || {
                    rayon::join(
                        || {
                            // ZERO_CHECK_POOL.install(|| {
                                Net::set_channel_id(0);
                                let mut transcript = transcript.clone();

                                // =======================================================================
                                // 2 Run ZeroCheck on
                                //
                                //     `f(q_0(x),...q_l(x), w_0(x),...w_d(x))`
                                //
                                // where `f` is the constraint polynomial i.e.,
                                //
                                //     f(q_l, q_r, q_m, q_o, w_a, w_b, w_c)
                                //     = q_l w_a(x) + q_r w_b(x) + q_m w_a(x)w_b(x) - q_o w_c(x)
                                //
                                // in vanilla plonk, and obtain a ZeroCheckSubClaim
                                // =======================================================================
                                let step = start_timer!(|| "ZeroCheck on f");

                                let fx = build_f(
                                    &pk.params.gate_func,
                                    pk.params.num_variables(),
                                    &pk.selector_oracles,
                                    &witness_polys[..num_gate_witnesses],
                                )
                                .unwrap();

                                let zero_check_proof =
                                    <Self as ZeroCheck<E::ScalarField>>::d_prove(
                                        fx,
                                        &mut transcript,
                                    )
                                    .unwrap();
                                end_timer!(step);

                                zero_check_proof
                            // })
                        },
                        || {
                            Net::set_channel_id(1);
                            let mut transcript = transcript.clone();

                            let step = start_timer!(|| "LookupCheck");

                            // =======================================================================
                            // 3. Construct and perform lookup checks
                            // =======================================================================
                            let (lookup_proof, lookup_opening_points) = Lookup::d_prove(
                                &pk.lookup_preprocessing,
                                &pk.pcs_param,
                                ops,
                                &mut transcript,
                            );
                            end_timer!(step);

                            (lookup_proof, lookup_opening_points)
                        },
                    )
                },
                || {
                    // PERM_CHECK_POOL.install(|| {
                        Net::set_channel_id(2);

                        // =======================================================================
                        // 3. Run permutation check on `\{w_i(x)\}` and `permutation_oracle`, and
                        // obtain a PermCheckSubClaim.
                        // =======================================================================
                        let step = start_timer!(|| "Permutation check on w_i(x)");

                        #[cfg(feature = "rational_sumcheck_piop")]
                        let perm_check_ret = <Self as PermutationCheck<E, PCS>>::d_prove(
                            &pk.pcs_param,
                            perm_to_prove,
                            perm_claims,
                            &mut perm_transcript,
                        )
                        .unwrap();

                        #[cfg(not(feature = "rational_sumcheck_piop"))]
                        let perm_check_ret = <Self as PermutationCheck<E, PCS>>::d_prove(
                            &witness_polys,
                            &witness_polys,
                            &pk.permutation_oracles,
                            &mut perm_transcript,
                        )
                        .unwrap();

                        end_timer!(step);

                        perm_check_ret
                    // })
                },
            );

        #[cfg(feature = "rational_sumcheck_piop")]
        let (perm_check_ret, perm_h_advices, perm_h_polys) = perm_check_ret;

        Net::set_channel_id(0);
        let (zero_check_point, perm_check_points, r_pi) = if Net::am_master() {
            let zero_check_point = &zero_check_proof.as_ref().unwrap().point;
            let perm_check_points = &perm_check_ret.as_ref().unwrap().1;

            let r_pi =
                transcript.get_and_append_challenge_vectors(b"r_pi", ell + num_party_vars)?;
            Net::recv_from_master_uniform(Some((
                zero_check_point.clone(),
                perm_check_points.clone(),
                r_pi.clone(),
            )))
        } else {
            Net::recv_from_master_uniform(None)
        };
        // =======================================================================
        // 4. Generate evaluations and corresponding proofs
        // - permcheck
        //  1. (deferred) batch opening perms(x) at
        //   - [perm_check_point]
        //  2. (deferred) batch opening witness_i(x) at
        //   - [perm_check_point]
        //
        // - zero check evaluations and proofs
        //   - 4.3.1. (deferred) wi_poly(zero_check_point)
        //   - 4.3.2. (deferred) selector_poly(zero_check_point)
        //
        // - 4.4. (deferred) public input consistency checks
        //   - pi_poly(r_pi) where r_pi is sampled from transcript
        // =======================================================================
        let step = start_timer!(|| "openings setup");

        // perms(x)'s points
        let mut last_nv = pk.permutation_oracles[0].num_vars();
        let mut index = 0;
        for (perm, advice) in zip(&pk.permutation_oracles, &pk.permutation_advices) {
            if perm.num_vars() != last_nv {
                last_nv = perm.num_vars();
                index += 1;
            }
            pcs_acc.insert_poly_and_points(perm, advice, &perm_check_points[index]);
        }

        // witnesses' points
        // TODO: refactor so it remains correct even if the order changed
        let mut last_nv = witness_polys[0].num_vars();
        let mut index = 0;
        for (wpoly, wadvice) in zip(&witness_polys, &witness_advices) {
            if wpoly.num_vars() != last_nv {
                last_nv = wpoly.num_vars();
                index += 1;
            }
            pcs_acc.insert_poly_and_points(wpoly, wadvice, &perm_check_points[index]);
        }

        #[cfg(feature = "rational_sumcheck_piop")]
        for (index, (poly, advice)) in zip(&perm_h_polys, &perm_h_advices).enumerate() {
            pcs_acc.insert_poly_and_points(poly, advice, &perm_check_points[index]);
        }

        for (wpoly, wadvice) in zip(&witness_polys[..num_gate_witnesses], &witness_advices) {
            pcs_acc.insert_poly_and_points(wpoly, wadvice, &zero_check_point);
        }
        for (wpoly, wadvice, point) in izip!(
            witness_polys[num_gate_witnesses..].iter(),
            witness_advices[num_gate_witnesses..].iter(),
            lookup_opening_points.witness_openings.iter()
        ) {
            pcs_acc.insert_poly_and_points(wpoly, wadvice, point);
        }

        //   - 4.3.2. (deferred) selector_poly(zero_check_point)
        for (poly, advice) in zip(&pk.selector_oracles, &pk.selector_advices) {
            pcs_acc.insert_poly_and_points(poly, advice, &zero_check_point);
        }

        // - 4.4. public input consistency checks
        //   - pi_poly(r_pi) where r_pi is sampled from transcript
        // padded with zeros
        let r_pi_padded = [
            &r_pi[..ell],
            &vec![E::ScalarField::zero(); num_vars - ell],
            &r_pi[ell..],
        ]
        .concat();
        // Evaluate witness_poly[0] at r_pi||0s which is equal to public_input evaluated
        // at r_pi. Assumes that public_input is a power of 2
        pcs_acc.insert_poly_and_points(&witness_polys[0], &witness_advices[0], &r_pi_padded);

        // - 5. lookup check points
        for (poly, advice_idx, point) in lookup_opening_points.regular_openings.iter() {
            pcs_acc.insert_poly_and_points(
                poly,
                &lookup_opening_points.regular_advices[*advice_idx],
                point,
            );
        }
        end_timer!(step);

        let step = start_timer!(|| "evaluate all");
        pcs_acc.evaluate_all();
        end_timer!(step);

        // =======================================================================
        // 5. deferred batch opening
        // =======================================================================
        let step = start_timer!(|| "deferred batch openings prod(x)");
        drop(witness_polys);
        drop(lookup_opening_points);
        let batch_openings = pcs_acc.d_multi_open(&pk.pcs_param, &mut transcript)?;
        end_timer!(step);

        end_timer!(start);

        if Net::am_master() {
            Ok(Some(HyperPlonkProof {
                // PCS commit for witnesses
                witness_commits: witness_commits
                    .iter_mut()
                    .map(|comm| take(comm).unwrap())
                    .collect(),
                // batch_openings,
                batch_openings: batch_openings.unwrap(),
                // =======================================================================
                // IOP proofs
                // =======================================================================
                // the custom gate zerocheck proof
                zero_check_proof: zero_check_proof.unwrap(),
                // the permutation check proof for copy constraints
                perm_check_proof: perm_check_ret.unwrap().0,
                lookup_proof: lookup_proof.unwrap(),
            }))
        } else {
            Ok(None)
        }
    }

    /// Verify the HyperPlonk proof.
    ///
    /// Inputs:
    /// - `vk`: verification key
    /// - `pub_input`: online public input
    /// - `proof`: HyperPlonk SNARK proof
    /// Outputs:
    /// - Return a boolean on whether the verification is successful
    ///
    /// 1. Verify zero_check_proof on
    ///
    ///     `f(q_0(x),...q_l(x), w_0(x),...w_d(x))`
    ///
    /// where `f` is the constraint polynomial i.e.,
    /// ```ignore
    ///     f(q_l, q_r, q_m, q_o, w_a, w_b, w_c)
    ///     = q_l w_a(x) + q_r w_b(x) + q_m w_a(x)w_b(x) - q_o w_c(x)
    /// ```
    /// in vanilla plonk, and obtain a ZeroCheckSubClaim
    ///
    /// 2. Verify perm_check_proof on `\{w_i(x)\}` and `permutation_oracles`
    ///
    /// 3. check subclaim validity
    ///
    /// 4. Verify the opening against the commitment:
    /// - check permutation check evaluations
    /// - check zero check evaluations
    /// - public input consistency checks
    fn verify(
        vk: &Self::VerifyingKey,
        pub_input: &[E::ScalarField],
        proof: &Self::Proof,
    ) -> Result<bool, HyperPlonkErrors> {
        let start = start_timer!(|| "hyperplonk verification");

        let mut transcript = IOPTranscript::<E::ScalarField>::new(b"hyperplonk");

        let num_selectors = vk.params.num_selector_columns();
        let num_witnesses = vk.params.num_witness_columns::<E, PCS, Lookup>();
        let num_vars = vk.params.num_variables();

        //  online public input of length 2^\ell
        let ell = log2(vk.params.num_pub_input) as usize;

        // =======================================================================
        // 0. sanity checks
        // =======================================================================
        // public input length
        if pub_input.len() != vk.params.num_pub_input {
            return Err(HyperPlonkErrors::InvalidProver(format!(
                "Public input length is not correct: got {}, expect {}",
                pub_input.len(),
                1 << ell
            )));
        }

        // Extract evaluations from openings
        let mut openings = PcsDynamicOpenings::new(&proof.batch_openings);
        let perm_evals = &openings.next_openings(num_witnesses);
        let witness_perm_evals = &openings.next_openings(num_witnesses);

        #[cfg(feature = "rational_sumcheck_piop")]
        let perm_h_evals = &openings.next_openings(proof.perm_check_proof.proofs.len());

        let witness_gate_evals = &openings.next_openings(num_witnesses);
        let selector_evals = &openings.next_openings(num_selectors);
        let pi_eval = openings.next_openings(1)[0];
        let lookup_evals = &openings.next_openings(usize::MAX);

        let num_gate_witnesses = vk.params.gate_func.num_witness_columns();

        // push witness to transcript
        for w_com in proof.witness_commits.iter() {
            transcript.append_serializable_element(b"w", w_com)?;
        }

        // =======================================================================
        // 1. Verify zero_check_proof on `f(q_0(x),...q_l(x), w_0(x),...w_d(x))`
        //
        // where `f` is the constraint polynomial i.e.,
        //
        //     f(q_l, q_r, q_m, q_o, w_a, w_b, w_c)
        //     = q_l w_a(x) + q_r w_b(x) + q_m w_a(x)w_b(x) - q_o w_c(x)
        //
        // =======================================================================
        let (zero_check_point, (mut lookup_opening_points, perm_check_sub_claim)) = rayon::join(
            || {
                let mut transcript = transcript.clone();

                let step = start_timer!(|| "verify zero check");
                // Zero check and perm check have different AuxInfo
                let zero_check_aux_info = VPAuxInfo::<E::ScalarField> {
                    max_degree: vk.params.gate_func.degree(),
                    num_variables: num_vars,
                    phantom: PhantomData::default(),
                };

                let zero_check_sub_claim = <Self as ZeroCheck<E::ScalarField>>::verify(
                    &proof.zero_check_proof,
                    &zero_check_aux_info,
                    &mut transcript,
                )
                .unwrap();

                let zero_check_point = zero_check_sub_claim.point;

                // check zero check subclaim
                let f_eval = eval_f(
                    &vk.params.gate_func,
                    selector_evals,
                    &witness_gate_evals[..num_gate_witnesses],
                )
                .unwrap();
                assert_eq!(f_eval, zero_check_sub_claim.expected_evaluation);

                end_timer!(step);
                zero_check_point
            },
            || {
                rayon::join(
                    || {
                        let mut transcript = transcript.clone();
                        let step = start_timer!(|| "verify lookup check");

                        // Verify lookup
                        let lookup_opening_points = Lookup::verify(
                            &proof.lookup_proof,
                            &witness_gate_evals[num_gate_witnesses..],
                            lookup_evals,
                            &mut transcript,
                        )
                        .unwrap();

                        end_timer!(step);

                        lookup_opening_points
                    },
                    || {
                        let mut transcript = transcript.clone();
                        // =======================================================================
                        // 2. Verify perm_check_proof on `\{w_i(x)\}` and `permutation_oracle`
                        // =======================================================================
                        let step = start_timer!(|| "verify permutation check");

                        let perm_check_sub_claim = <Self as PermutationCheck<E, PCS>>::verify(
                            &proof.perm_check_proof,
                            &mut transcript,
                        )
                        .unwrap();

                        <Self as PermutationCheck<E, PCS>>::check_openings(
                            &perm_check_sub_claim,
                            &witness_perm_evals,
                            &witness_perm_evals,
                            #[cfg(feature = "rational_sumcheck_piop")]
                            &perm_h_evals,
                            &perm_evals,
                        )
                        .unwrap();

                        end_timer!(step);

                        perm_check_sub_claim
                    },
                )
            },
        );

        // =======================================================================
        // 3. Verify the opening against the commitment
        // =======================================================================
        let step = start_timer!(|| "assemble commitments");

        // generate evaluation points and commitments
        let mut pcs_acc = PcsDynamicVerifier::<E, PCS>::new();

        #[cfg(feature = "rational_sumcheck_piop")]
        {
            // perms' points
            let mut index = 0;
            for (subclaim, len) in perm_check_sub_claim.subclaims.iter() {
                for i in 0..*len {
                    pcs_acc.insert_comm_and_points(
                        vk.perm_commitments[index + i].clone(),
                        subclaim.sumcheck_point.clone(),
                    );
                }
                index += len;
            }

            // witnesses' points
            // TODO: merge points
            let mut index = 0;
            for (subclaim, len) in perm_check_sub_claim.subclaims.iter() {
                for i in 0..*len {
                    pcs_acc.insert_comm_and_points(
                        proof.witness_commits[index + i].clone(),
                        subclaim.sumcheck_point.clone(),
                    );
                }
                index += len;
            }

            for (h_comm, (subclaim, _)) in zip(
                &proof.perm_check_proof.h_comms,
                &perm_check_sub_claim.subclaims,
            ) {
                pcs_acc.insert_comm_and_points(h_comm.clone(), subclaim.sumcheck_point.clone());
            }
        }

        #[cfg(not(feature = "rational_sumcheck_piop"))]
        {
            // perms' points
            let mut index = 0;
            for subclaim in perm_check_sub_claim.subclaims.iter() {
                for i in 0..subclaim.len {
                    pcs_acc.insert_comm_and_points(
                        vk.perm_commitments[index + i].clone(),
                        subclaim.point.clone(),
                    );
                }
                index += subclaim.len;
            }

            // witnesses' points
            // TODO: merge points
            let mut index = 0;
            for subclaim in perm_check_sub_claim.subclaims.iter() {
                for i in 0..subclaim.len {
                    pcs_acc.insert_comm_and_points(
                        proof.witness_commits[index + i].clone(),
                        subclaim.point.clone(),
                    );
                }
                index += subclaim.len;
            }
        }

        for wcom in proof.witness_commits[..num_gate_witnesses].iter() {
            pcs_acc.insert_comm_and_points(wcom.clone(), zero_check_point.clone());
        }
        for (wcom, point) in zip(
            proof.witness_commits[num_gate_witnesses..].iter(),
            lookup_opening_points.witness_openings.iter_mut(),
        ) {
            pcs_acc.insert_comm_and_points(wcom.clone(), take(point));
        }

        // selector_poly(zero_check_point)
        for com in vk.selector_commitments.iter() {
            pcs_acc.insert_comm_and_points(com.clone(), zero_check_point.clone());
        }

        // - 4.4. public input consistency checks
        //   - pi_poly(r_pi) where r_pi is sampled from transcript
        let r_pi = transcript.get_and_append_challenge_vectors(b"r_pi", ell)?;

        // check public evaluation
        let pi_step = start_timer!(|| "check public evaluation");
        let pi_poly = DenseMultilinearExtension::from_evaluations_slice(ell, pub_input);
        let expect_pi_eval = evaluate_opt(&pi_poly, &r_pi[..]);
        if expect_pi_eval != pi_eval {
            return Err(HyperPlonkErrors::InvalidProver(format!(
                "Public input eval mismatch: got {}, expect {}",
                pi_eval, expect_pi_eval,
            )));
        }
        let r_pi_padded = [
            &r_pi[..r_pi.len() - vk.num_party_vars],
            &vec![E::ScalarField::zero(); num_vars - ell],
            &r_pi[r_pi.len() - vk.num_party_vars..],
        ]
        .concat();

        pcs_acc.insert_comm_and_points(proof.witness_commits[0].clone(), r_pi_padded);
        end_timer!(pi_step);

        for (comm, point) in lookup_opening_points.regular_openings.iter_mut() {
            pcs_acc.insert_comm_and_points(comm.clone(), take(point));
        }

        end_timer!(step);
        let step = start_timer!(|| "PCS batch verify");
        // check proof
        let res =
            pcs_acc.batch_verify(&vk.pcs_param, &proof.batch_openings.proofs, &mut transcript)?;

        end_timer!(step);
        end_timer!(start);
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        custom_gate::CustomizedGates, jolt_lookup, selectors::SelectorColumn,
        structs::HyperPlonkParams, witness::WitnessColumn,
    };
    use arithmetic::{identity_permutation, random_permutation, random_permutation_raw};
    use ark_bls12_381::Bls12_381;
    use ark_std::{test_rng, One};
    use subroutines::{
        instruction::{and::ANDInstruction, or::ORInstruction, xor::XORInstruction},
        pcs::prelude::MultilinearKzgPCS,
    };

    #[test]
    fn test_hyperplonk_e2e() -> Result<(), HyperPlonkErrors> {
        // Example:
        //     q_L(X) * W_1(X)^5 - W_2(X) = 0
        // is represented as
        // vec![
        //     ( 1,    Some(id_qL),    vec![id_W1, id_W1, id_W1, id_W1, id_W1]),
        //     (-1,    None,           vec![id_W2])
        // ]
        //
        // 4 public input
        // 1 selector,
        // 2 witnesses,
        // 2 variables for MLE,
        // 4 wires,
        let gates = CustomizedGates {
            gates: vec![(1, Some(0), vec![0, 0, 0, 0, 0]), (-1, None, vec![1])],
        };
        test_hyperplonk_helper::<Bls12_381>(gates)
    }

    fn test_hyperplonk_helper<E: Pairing>(
        gate_func: CustomizedGates,
    ) -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let num_constraints = 4;
        let num_pub_input = 4;
        let num_witnesses = 2;

        let nv = log2(num_constraints) as usize;
        let pcs_srs = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;

        // generate index
        let params = HyperPlonkParams {
            num_constraints,
            num_lookup_constraints: vec![],
            num_pub_input,
            gate_func,
        };
        let permutation = identity_permutation(nv, num_witnesses);
        let q1 = SelectorColumn(vec![
            E::ScalarField::one(),
            E::ScalarField::one(),
            E::ScalarField::one(),
            E::ScalarField::one(),
        ]);
        let index = HyperPlonkIndex {
            params,
            permutation,
            selectors: vec![q1],
        };

        // generate pk and vks
        let (pk, vk) =
            <PolyIOP<E::ScalarField> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::preprocess(
                &index, &pcs_srs,
            )?;

        // w1 := [0, 1, 2, 3]
        let w1 = WitnessColumn(vec![
            E::ScalarField::zero(),
            E::ScalarField::one(),
            E::ScalarField::from(2u128),
            E::ScalarField::from(3u128),
        ]);
        // w2 := [0^5, 1^5, 2^5, 3^5]
        let w2 = WitnessColumn(vec![
            E::ScalarField::zero(),
            E::ScalarField::one(),
            E::ScalarField::from(32u128),
            E::ScalarField::from(243u128),
        ]);
        // public input = w1
        let pi = w1.clone();

        // generate a proof and verify
        let proof = <PolyIOP<E::ScalarField> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::prove(
            &pk,
            &pi.0,
            &[w1.clone(), w2.clone()],
            &(),
        )?;

        let _verify =
            <PolyIOP<E::ScalarField> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::verify(
                &vk, &pi.0, &proof,
            )?;

        // bad path 1: wrong permutation
        let rand_perm: Vec<E::ScalarField> = random_permutation(nv, num_witnesses, &mut rng);
        let mut bad_index = index;
        bad_index.permutation = rand_perm;
        // generate pk and vks
        let (_, bad_vk) =
            <PolyIOP<E::ScalarField> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::preprocess(
                &bad_index, &pcs_srs,
            )?;
        assert!(!<PolyIOP<E::ScalarField> as HyperPlonkSNARK<
            E,
            MultilinearKzgPCS<E>,
        >>::verify(&bad_vk, &pi.0, &proof,)?);

        // bad path 2: wrong witness
        let mut w1_bad = w1;
        w1_bad.0[0] = E::ScalarField::one();
        assert!(
            <PolyIOP<E::ScalarField> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::prove(
                &pk,
                &pi.0,
                &[w1_bad, w2],
                &(),
            )
            .is_err()
        );

        Ok(())
    }

    const C: usize = 2;
    const M: usize = 1 << 8;

    jolt_lookup! { LookupPlugin, C, M ;
        XORInstruction,
        ORInstruction,
        ANDInstruction
    }

    #[test]
    fn test_hyperplonk_lookup() -> Result<(), HyperPlonkErrors> {
        // Example:
        //     q_L(X) * W_1(X)^5 - W_2(X) = 0
        // is represented as
        // vec![
        //     ( 1,    Some(id_qL),    vec![id_W1, id_W1, id_W1, id_W1, id_W1]),
        //     (-1,    None,           vec![id_W2])
        // ]
        //
        // 4 public input
        // 1 selector,
        // 2 witnesses,
        // 2 variables for MLE,
        // 4 wires,
        let gates = CustomizedGates {
            gates: vec![(1, Some(0), vec![0, 0, 0, 0, 0]), (-1, None, vec![1])],
        };
        test_hyperplonk_lookup_helper::<Bls12_381>(gates)
    }

    fn test_hyperplonk_lookup_helper<E: Pairing>(
        gate_func: CustomizedGates,
    ) -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let pcs_srs = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 16)?;

        let num_constraints = 4;
        let num_pub_input = 4;
        let nv = log2(num_constraints) as usize;
        let num_witnesses = 2;

        // generate index
        let params = HyperPlonkParams {
            num_constraints,
            num_lookup_constraints: vec![5, 0, 3],
            num_pub_input,
            gate_func,
        };
        let perm_len = (1u64 << nv) * (num_witnesses as u64) + (1u64 << 3) * 3 + (1u64 << 2) * 3;
        let permutation = (0..perm_len).map(E::ScalarField::from).collect();
        let q1 = SelectorColumn(vec![
            E::ScalarField::one(),
            E::ScalarField::one(),
            E::ScalarField::one(),
            E::ScalarField::one(),
        ]);
        let index = HyperPlonkIndex {
            params,
            permutation,
            selectors: vec![q1],
        };

        // generate pk and vks
        let ops = (
            Some(vec![
                XORInstruction(0, 1),
                XORInstruction(101, 101),
                XORInstruction(202, 1),
                XORInstruction(220, 1),
                XORInstruction(220, 1),
            ]),
            None,
            Some(vec![
                ANDInstruction(113, 5),
                ANDInstruction(220, 7),
                ANDInstruction(221, 9),
            ]),
        );
        let (pk, vk) = <PolyIOP<E::ScalarField> as HyperPlonkSNARK<
            E,
            MultilinearKzgPCS<E>,
            LookupPlugin,
        >>::preprocess(&index, &pcs_srs)?;

        // w1 := [0, 1, 2, 3]
        let w1 = WitnessColumn(vec![
            E::ScalarField::zero(),
            E::ScalarField::one(),
            E::ScalarField::from(2u128),
            E::ScalarField::from(3u128),
        ]);
        // w2 := [0^5, 1^5, 2^5, 3^5]
        let w2 = WitnessColumn(vec![
            E::ScalarField::zero(),
            E::ScalarField::one(),
            E::ScalarField::from(32u128),
            E::ScalarField::from(243u128),
        ]);
        // public input = w1
        let pi = w1.clone();

        // generate a proof and verify
        let proof = <PolyIOP<E::ScalarField> as HyperPlonkSNARK<
            E,
            MultilinearKzgPCS<E>,
            LookupPlugin,
        >>::prove(&pk, &pi.0, &[w1.clone(), w2.clone()], &ops)?;

        let verify = <PolyIOP<E::ScalarField> as HyperPlonkSNARK<
            E,
            MultilinearKzgPCS<E>,
            LookupPlugin,
        >>::verify(&vk, &pi.0, &proof)?;
        assert!(verify);

        // bad path 1: wrong permutation
        let rand_perm: Vec<E::ScalarField> = random_permutation_raw(perm_len, &mut rng);
        let mut bad_index = index;
        bad_index.permutation = rand_perm;
        // generate pk and vks
        let (_, bad_vk) = <PolyIOP<E::ScalarField> as HyperPlonkSNARK<
            E,
            MultilinearKzgPCS<E>,
            LookupPlugin,
        >>::preprocess(&bad_index, &pcs_srs)?;
        assert!(!<PolyIOP<E::ScalarField> as HyperPlonkSNARK<
            E,
            MultilinearKzgPCS<E>,
            LookupPlugin,
        >>::verify(&bad_vk, &pi.0, &proof,)?);

        // bad path 2: wrong witness
        let mut w1_bad = w1;
        w1_bad.0[0] = E::ScalarField::one();
        assert!(<PolyIOP<E::ScalarField> as HyperPlonkSNARK<
            E,
            MultilinearKzgPCS<E>,
            LookupPlugin,
        >>::prove(&pk, &pi.0, &[w1_bad, w2], &ops,)
        .is_err());

        Ok(())
    }
}

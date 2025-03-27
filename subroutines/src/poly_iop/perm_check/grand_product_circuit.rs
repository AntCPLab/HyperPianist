use crate::poly_iop::{
    sum_check::{
        batched_cubic_sumcheck::BatchedCubicSumcheckInstance,
        generic_sumcheck::ZerocheckInstanceProof,
    },
    utils::drop_in_background_thread,
};
use arithmetic::{bind_poly_var_bot, eq_poly::EqPolynomial, math::Math, unsafe_allocate_zero_vec};
use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::*;
use ark_std::{end_timer, start_timer};
use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};
use itertools::Itertools;
use rayon::prelude::*;
use std::{iter::zip, mem::take};
use transcript::IOPTranscript;

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchedGrandProductLayerProof<F: PrimeField> {
    pub proof: ZerocheckInstanceProof<F>,
    pub left_claims: Vec<F>,
    pub right_claims: Vec<F>,
}

impl<F: PrimeField> BatchedGrandProductLayerProof<F> {
    pub fn verify(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        zerocheck_r: &[F],
        transcript: &mut IOPTranscript<F>,
    ) -> (F, Vec<F>) {
        self.proof
            .verify(claim, num_rounds, degree_bound, zerocheck_r, transcript)
            .unwrap()
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchedGrandProductProof<F: PrimeField> {
    pub layers: Vec<BatchedGrandProductLayerProof<F>>,
}

pub trait BatchedGrandProduct<F: PrimeField>: Sized {
    /// The bottom/input layer of the grand products
    type Leaves;

    /// Constructs the grand product circuit(s) from `leaves`
    fn construct(leaves: Self::Leaves) -> Self;

    fn d_construct(leaves: Self::Leaves) -> (Self, Option<Self>);

    /// The number of layers in the grand product.
    fn num_layers(&self) -> usize;
    /// The claimed outputs of the grand products.
    fn claims(&self) -> Vec<F>;
    /// Returns an iterator over the layers of this batched grand product
    /// circuit. Each layer is mutable so that its polynomials can be bound
    /// over the course of proving.
    fn layers(&'_ mut self) -> impl Iterator<Item = &'_ mut dyn BatchedGrandProductLayer<F>>;

    /// Computes a batched grand product proof, layer by layer.
    // #[tracing::instrument(skip_all, name =
    // "BatchedGrandProduct::prove_grand_product")]
    fn prove_grand_product(
        &mut self,
        transcript: &mut IOPTranscript<F>,
    ) -> (BatchedGrandProductProof<F>, Vec<F>) {
        let mut proof_layers = Vec::with_capacity(self.num_layers());
        let mut claims_to_verify = self.claims();
        let mut r_grand_product = Vec::new();

        for layer in self.layers() {
            proof_layers.push(layer.prove_layer(
                &mut claims_to_verify,
                &mut r_grand_product,
                transcript,
            ));
        }

        r_grand_product.reverse();

        (
            BatchedGrandProductProof {
                layers: proof_layers,
            },
            r_grand_product,
        )
    }

    fn d_prove_grand_product(
        &mut self,
        final_circuit: Option<&mut Self>,
        transcript: &mut IOPTranscript<F>,
    ) -> Option<(BatchedGrandProductProof<F>, Vec<F>)> {
        let timer = start_timer!(|| "d prove grand product");

        let mut proof_layers: Vec<BatchedGrandProductLayerProof<F>> =
            Vec::with_capacity(self.num_layers());
        let mut claims_to_verify = Vec::new();
        let mut r_grand_product = Vec::new();

        let step = start_timer!(|| "master prove");
        r_grand_product = if Net::am_master() {
            let final_circuit = final_circuit.unwrap();
            claims_to_verify = final_circuit.claims();
            for layer in final_circuit.layers() {
                proof_layers.push(layer.prove_layer(
                    &mut claims_to_verify,
                    &mut r_grand_product,
                    transcript,
                ));
            }
            Net::recv_from_master_uniform(Some(r_grand_product))
        } else {
            Net::recv_from_master_uniform(None)
        };
        end_timer!(step);

        let step = start_timer!(|| "prove layers");
        for layer in self.layers() {
            let layer_proof =
                layer.d_prove_layer(&mut claims_to_verify, &mut r_grand_product, transcript);
            if Net::am_master() {
                proof_layers.push(layer_proof.unwrap());
            }
        }

        r_grand_product.reverse();
        end_timer!(step);

        end_timer!(timer);

        if Net::am_master() {
            Some((
                BatchedGrandProductProof {
                    layers: proof_layers,
                },
                r_grand_product,
            ))
        } else {
            None
        }
    }

    /// Verifies that the `sumcheck_claim` output by sumcheck verification is
    /// consistent with the `left_claims` and `right_claims` of
    /// corresponding `BatchedGrandProductLayerProof`. This function may be
    /// overridden if the layer isn't just multiplication gates, e.g. in the
    /// case of `ToggledBatchedGrandProduct`.
    fn verify_sumcheck_claim(
        layer_proofs: &[BatchedGrandProductLayerProof<F>],
        layer_index: usize,
        coeffs: &[F],
        sumcheck_claim: F,
        grand_product_claims: &mut Vec<F>,
        r_grand_product: &mut Vec<F>,
        transcript: &mut IOPTranscript<F>,
    ) {
        let layer_proof = &layer_proofs[layer_index];
        let expected_sumcheck_claim: F = (0..grand_product_claims.len())
            .map(|i| coeffs[i] * layer_proof.left_claims[i] * layer_proof.right_claims[i])
            .sum();

        assert_eq!(expected_sumcheck_claim, sumcheck_claim);

        // produce a random challenge to condense two claims into a single claim
        let r_layer = transcript
            .get_and_append_challenge(b"challenge_r_layer")
            .unwrap();

        *grand_product_claims = layer_proof
            .left_claims
            .iter()
            .zip_eq(layer_proof.right_claims.iter())
            .map(|(&left_claim, &right_claim)| left_claim + r_layer * (right_claim - left_claim))
            .collect();

        r_grand_product.push(r_layer);
    }

    /// Function used for layer sumchecks in the generic batch verifier as well
    /// as the quark layered sumcheck hybrid
    fn verify_layers(
        proof_layers: &[BatchedGrandProductLayerProof<F>],
        claims: &Vec<F>,
        transcript: &mut IOPTranscript<F>,
        r_start: Vec<F>,
    ) -> (Vec<F>, Vec<F>) {
        let mut claims_to_verify = claims.to_owned();
        // We allow a non empty start in this function call because the quark hybrid
        // form provides prespecified random for most of the positions and then
        // we proceed with GKR on the remaining layers using the preset random values.
        // For default thaler '13 layered grand products this should be empty.
        let mut r_grand_product = r_start.clone();
        let fixed_at_start = r_start.len();

        for (layer_index, layer_proof) in proof_layers.iter().enumerate() {
            // produce a fresh set of coeffs
            let coeffs: Vec<F> = transcript
                .get_and_append_challenge_vectors(b"rand_coeffs_next_layer", claims_to_verify.len())
                .unwrap();
            // produce a joint claim
            let claim = claims_to_verify
                .iter()
                .zip_eq(coeffs.iter())
                .map(|(&claim, &coeff)| claim * coeff)
                .sum();

            let mut zerocheck_r = r_grand_product.clone();
            zerocheck_r.reverse();
            let (sumcheck_claim, r_sumcheck) = layer_proof.verify(
                claim,
                layer_index + fixed_at_start,
                2,
                &zerocheck_r,
                transcript,
            );

            assert_eq!(claims.len(), layer_proof.left_claims.len());
            assert_eq!(claims.len(), layer_proof.right_claims.len());

            for (left, right) in layer_proof
                .left_claims
                .iter()
                .zip_eq(layer_proof.right_claims.iter())
            {
                transcript
                    .append_field_element(b"sumcheck left claim", left)
                    .unwrap();
                transcript
                    .append_field_element(b"sumcheck right claim", right)
                    .unwrap();
            }

            assert_eq!(r_grand_product.len(), r_sumcheck.len());

            r_grand_product = r_sumcheck.into_iter().rev().collect();

            Self::verify_sumcheck_claim(
                proof_layers,
                layer_index,
                &coeffs,
                sumcheck_claim,
                &mut claims_to_verify,
                &mut r_grand_product,
                transcript,
            );
        }

        r_grand_product.reverse();
        (claims_to_verify, r_grand_product)
    }

    /// Verifies the given grand product proof.
    fn verify_grand_product(
        proof: &BatchedGrandProductProof<F>,
        claims: &Vec<F>,
        transcript: &mut IOPTranscript<F>,
    ) -> (Vec<F>, Vec<F>) {
        // Pass the inputs to the layer verification function, by default we have no
        // quarks and so we do not use the quark proof fields.
        let r_start = Vec::<F>::new();
        Self::verify_layers(&proof.layers, claims, transcript, r_start)
    }
}

pub trait BatchedGrandProductLayer<F: PrimeField>: BatchedCubicSumcheckInstance<F> {
    /// Proves a single layer of a batched grand product circuit
    fn prove_layer(
        &mut self,
        claims: &mut Vec<F>,
        r_grand_product: &mut Vec<F>,
        transcript: &mut IOPTranscript<F>,
    ) -> BatchedGrandProductLayerProof<F> {
        // produce a fresh set of coeffs
        let coeffs: Vec<F> = transcript
            .get_and_append_challenge_vectors(b"rand_coeffs_next_layer", claims.len())
            .unwrap();
        // produce a joint claim
        let claim = claims
            .iter()
            .zip_eq(coeffs.iter())
            .map(|(&claim, &coeff)| claim * coeff)
            .sum();

        let (sumcheck_proof, r_sumcheck, mut sumcheck_claims) =
            self.prove_sumcheck(&claim, &coeffs, &r_grand_product, transcript, &F::zero());

        let (left_claims, right_claims) =
            (take(&mut sumcheck_claims[0]), take(&mut sumcheck_claims[1]));
        for (left, right) in left_claims.iter().zip_eq(right_claims.iter()) {
            transcript
                .append_field_element(b"sumcheck left claim", left)
                .unwrap();
            transcript
                .append_field_element(b"sumcheck right claim", right)
                .unwrap();
        }

        r_sumcheck
            .into_par_iter()
            .rev()
            .collect_into_vec(r_grand_product);

        // produce a random challenge to condense two claims into a single claim
        let r_layer = transcript
            .get_and_append_challenge(b"challenge_r_layer")
            .unwrap();

        *claims = left_claims
            .iter()
            .zip_eq(right_claims.iter())
            .map(|(&left_claim, &right_claim)| left_claim + r_layer * (right_claim - left_claim))
            .collect::<Vec<F>>();

        r_grand_product.push(r_layer);

        BatchedGrandProductLayerProof {
            proof: sumcheck_proof,
            left_claims,
            right_claims,
        }
    }

    fn d_prove_layer(
        &mut self,
        claims: &mut Vec<F>,
        r_grand_product: &mut Vec<F>,
        transcript: &mut IOPTranscript<F>,
    ) -> Option<BatchedGrandProductLayerProof<F>> {
        let coeffs = if Net::am_master() {
            // produce a fresh set of coeffs
            let coeffs: Vec<F> = transcript
                .get_and_append_challenge_vectors(b"rand_coeffs_next_layer", claims.len())
                .unwrap();
            Net::recv_from_master_uniform(Some(coeffs))
        } else {
            Net::recv_from_master_uniform(None)
        };

        // produce a joint claim
        let claim = if Net::am_master() {
            claims
                .iter()
                .zip_eq(coeffs.iter())
                .map(|(&claim, &coeff)| claim * coeff)
                .sum()
        } else {
            // No need for non-masters to know the claim
            F::zero()
        };

        let num_party_vars = Net::n_parties().log_2();

        let proof =
            self.d_prove_sumcheck(&claim, &coeffs, &r_grand_product, transcript, &F::zero());

        if Net::am_master() {
            let (sumcheck_proof, r_sumcheck, sumcheck_claims) = proof.unwrap();
            let (left_claims_polys, right_claims_polys) = (0..sumcheck_claims[0][0].len())
                .map(|i| {
                    let (left_evals, right_evals) = sumcheck_claims
                        .iter()
                        .map(|claims| (claims[0][i], claims[1][i]))
                        .unzip();
                    (
                        DenseMultilinearExtension::from_evaluations_vec(num_party_vars, left_evals),
                        DenseMultilinearExtension::from_evaluations_vec(
                            num_party_vars,
                            right_evals,
                        ),
                    )
                })
                .unzip::<_, _, Vec<_>, Vec<_>>();

            let r_party = &r_sumcheck[r_sumcheck.len() - num_party_vars..];

            let left_claims = left_claims_polys
                .iter()
                .map(|poly| poly.evaluate(&r_party).unwrap())
                .collect::<Vec<_>>();
            let right_claims = right_claims_polys
                .iter()
                .map(|poly| poly.evaluate(&r_party).unwrap())
                .collect::<Vec<_>>();

            for (left, right) in left_claims.iter().zip_eq(right_claims.iter()) {
                transcript
                    .append_field_element(b"sumcheck left claim", left)
                    .unwrap();
                transcript
                    .append_field_element(b"sumcheck right claim", right)
                    .unwrap();
            }

            r_sumcheck
                .into_par_iter()
                .rev()
                .collect_into_vec(r_grand_product);

            // produce a random challenge to condense two claims into a single claim
            let r_layer = transcript
                .get_and_append_challenge(b"challenge_r_layer")
                .unwrap();

            r_grand_product.push(r_layer);

            Net::recv_from_master_uniform(Some(r_grand_product.clone()));

            *claims = left_claims
                .iter()
                .zip_eq(right_claims.iter())
                .map(|(&left_claim, &right_claim)| {
                    left_claim + r_layer * (right_claim - left_claim)
                })
                .collect::<Vec<F>>();

            Some(BatchedGrandProductLayerProof {
                proof: sumcheck_proof,
                left_claims,
                right_claims,
            })
        } else {
            *r_grand_product = Net::recv_from_master_uniform(None);
            None
        }
    }
}

/// Represents a single layer of a single grand product circuit.
/// A layer is assumed to be arranged in "interleaved" order, i.e. the natural
/// order in the visual representation of the circuit:
///      Λ        Λ        Λ        Λ
///     / \      / \      / \      / \
///   L0   R0  L1   R1  L2   R2  L3   R3   <- This is layer would be represented
/// as [L0, R0, L1, R1, L2, R2, L3, R3]                                         
/// (as opposed to e.g. [L0, L1, L2, L3, R0, R1, R2, R3])
pub type DenseGrandProductLayer<F> = Vec<F>;

/// Represents a batch of `DenseGrandProductLayer`, all of the same length
/// `layer_len`.
#[derive(Debug, Clone)]
pub struct BatchedDenseGrandProductLayer<F: PrimeField> {
    pub layers: Vec<DenseGrandProductLayer<F>>,
    pub layer_len: usize,
}

impl<F: PrimeField> BatchedDenseGrandProductLayer<F> {
    pub fn new(values: Vec<Vec<F>>) -> Self {
        let layer_len = values[0].len();
        Self {
            layers: values,
            layer_len,
        }
    }
}

impl<F: PrimeField> BatchedGrandProductLayer<F> for BatchedDenseGrandProductLayer<F> {}
impl<F: PrimeField> BatchedCubicSumcheckInstance<F> for BatchedDenseGrandProductLayer<F> {
    fn num_rounds(&self) -> usize {
        self.layer_len.log_2() - 1
    }

    /// Incrementally binds a variable of this batched layer's polynomials.
    /// Even though each layer is backed by a single Vec<F>, it represents two
    /// polynomials one for the left nodes in the circuit, one for the right
    /// nodes in the circuit. These two polynomials' coefficients are
    /// interleaved into one Vec<F>. To preserve this interleaved order, we
    /// bind values like this:   0'  1'     2'  3'
    ///   |\ |\      |\ |\
    ///   | \| \     | \| \
    ///   |  \  \    |  \  \
    ///   |  |\  \   |  |\  \
    ///   0  1 2  3  4  5 6  7
    /// Left nodes have even indices, right nodes have odd indices.
    // #[tracing::instrument(skip_all, name =
    // "BatchedDenseGrandProductLayer::bind")]
    fn bind(&mut self, r: &F) {
        debug_assert!(self.layer_len % 4 == 0);
        let timer = start_timer!(|| "bind");
        // TODO(moodlezoup): parallelize over chunks instead of over batch
        let concurrency = (rayon::current_num_threads() * 4 + self.layers.len() - 1) / self.layers.len();
        self.layers
            .par_iter_mut()
            .for_each(|layer: &mut DenseGrandProductLayer<F>| {
                let n = layer.len() / 4;
                let mut chunk_size = (layer.len() + concurrency - 1) / concurrency;
                chunk_size += (4 - chunk_size % 4) % 4;

                let num_chunks = (layer.len() + chunk_size - 1) / chunk_size;
                layer.par_chunks_mut(chunk_size).for_each(|chunk| {
                    for i in 0..chunk.len() / 4 {
                        chunk[2 * i] = chunk[4 * i] + *r * (chunk[4 * i + 2] - chunk[4 * i]);
                        chunk[2 * i + 1] =
                            chunk[4 * i + 1] + *r * (chunk[4 * i + 3] - chunk[4 * i + 1]);
                    }
                });
                for i in 1..num_chunks {
                    let src_start = i * chunk_size;
                    let dst_start = (i * chunk_size) / 2;
                    let size = (std::cmp::min((i + 1) * chunk_size, layer.len()) - src_start) / 2;
                    unsafe {
                        let data = layer.as_mut_ptr();
                        std::ptr::copy_nonoverlapping(
                            data.add(src_start),
                            data.add(dst_start),
                            size,
                        );
                    }
                }
                layer.truncate(2 * n);
            });
        self.layer_len /= 2;
        end_timer!(timer);
    }

    /// We want to compute the evaluations of the following univariate cubic
    /// polynomial at points {0, 1, 2, 3}:
    ///     Σ coeff[batch_index] * (Σ eq(r, x) * left(x) * right(x))
    /// where the inner summation is over all but the "least significant bit" of
    /// the multilinear polynomials `eq`, `left`, and `right`. We denote
    /// this "least significant" variable x_b.
    ///
    /// Computing these evaluations requires processing pairs of adjacent
    /// coefficients of `eq`, `left`, and `right`.
    /// Recall that the `left` and `right` polynomials are interleaved in each
    /// layer of `self.layers`, so we process each layer 4 values at a time:
    ///                  layer = [L, R, L, R, L, R, ...]
    ///                           |  |  |  |
    ///    left(0, 0, 0, ..., x_b=0) |  |  right(0, 0, 0, ..., x_b=1)
    ///     right(0, 0, 0, ..., x_b=0)  left(0, 0, 0, ..., x_b=1)
    // #[tracing::instrument(skip_all, name =
    // "BatchedDenseGrandProductLayer::compute_cubic")]
    fn compute_cubic(&self, coeffs: &[F], eq_table: &[F], _lambda: &F) -> (F, F) {
        let timer = start_timer!(|| "compute cubic");
        let ret = (0..eq_table.len())
            .into_par_iter()
            .map(|i| {
                let mut evals = (F::zero(), F::zero());

                self.layers
                    .iter()
                    .enumerate()
                    .for_each(|(batch_index, layer)| {
                        let effective_coeff = coeffs[batch_index] * eq_table[i];
                        let left = (
                            effective_coeff * layer[4 * i],
                            effective_coeff * layer[4 * i + 2],
                        );
                        let right = (layer[4 * i + 1], layer[4 * i + 3]);

                        let m_left = left.1 - left.0;
                        let m_right = right.1 - right.0;
                        let left_eval_2 = left.1 + m_left;
                        let right_eval_2 = right.1 + m_right;

                        evals.0 += left.0 * right.0;
                        evals.1 += left_eval_2 * right_eval_2;
                    });

                evals
            })
            .reduce(
                || (F::zero(), F::zero()),
                |sum, evals| (sum.0 + evals.0, sum.1 + evals.1),
            );
        end_timer!(timer);
        ret
    }

    fn final_claims(&self) -> Vec<Vec<F>> {
        assert_eq!(self.layer_len, 2);
        let (left_claims, right_claims) =
            self.layers.iter().map(|layer| (layer[0], layer[1])).unzip();
        vec![left_claims, right_claims]
    }

    fn compute_cubic_direct(
        &self,
        coeffs: &[F],
        evaluations: &[Vec<DenseMultilinearExtension<F>>],
        eq_table: &[F],
        _lambda: &F,
    ) -> (F, F) {
        (0..eq_table.len())
            .into_par_iter()
            .map(|i| {
                let mut evals = (F::zero(), F::zero());

                zip(&evaluations[0], &evaluations[1]).enumerate().for_each(
                    |(batch_index, (left, right))| {
                        let effective_coeff = coeffs[batch_index] * eq_table[i];
                        let left = (
                            effective_coeff * left.evaluations[2 * i],
                            effective_coeff * left.evaluations[2 * i + 1],
                        );
                        let right = (right.evaluations[2 * i], right.evaluations[2 * i + 1]);

                        let m_left = left.1 - left.0;
                        let m_right = right.1 - right.0;

                        let left_eval_2 = left.1 + m_left;
                        let right_eval_2 = right.1 + m_right;

                        evals.0 += left.0 * right.0;
                        evals.1 += left_eval_2 * right_eval_2;
                    },
                );
                evals
            })
            .reduce(
                || (F::zero(), F::zero()),
                |sum, evals| (sum.0 + evals.0, sum.1 + evals.1),
            )
    }
}

/// A batched grand product circuit.
/// Note that the circuit roots are not included in `self.layers`
///        o
///      /   \
///     o     o  <- layers[layers.len() - 1]
///    / \   / \
///   o   o o   o  <- layers[layers.len() - 2]
///       ...
pub struct BatchedDenseGrandProduct<F: PrimeField> {
    layers: Vec<BatchedDenseGrandProductLayer<F>>,
}

impl<F: PrimeField> BatchedGrandProduct<F> for BatchedDenseGrandProduct<F> {
    type Leaves = Vec<Vec<F>>;

    // #[tracing::instrument(skip_all, name =
    // "BatchedDenseGrandProduct::construct")]
    fn construct(leaves: Self::Leaves) -> Self {
        let num_layers = leaves[0].len().log_2();
        let mut layers: Vec<BatchedDenseGrandProductLayer<F>> = Vec::with_capacity(num_layers);
        layers.push(BatchedDenseGrandProductLayer::new(leaves));

        for i in 0..num_layers - 1 {
            let previous_layers = &layers[i];
            let len = previous_layers.layer_len / 2;
            let new_layers = previous_layers
                .layers
                .par_iter()
                .map(|previous_layer| {
                    (0..len)
                        .into_par_iter()
                        .map(|i| previous_layer[2 * i] * previous_layer[2 * i + 1])
                        .collect::<Vec<_>>()
                })
                .collect();
            layers.push(BatchedDenseGrandProductLayer::new(new_layers));
        }

        Self { layers }
    }

    fn d_construct(leaves: Self::Leaves) -> (Self, Option<Self>) {
        let circuit = Self::construct(leaves);

        let claims = Net::send_to_master(&circuit.claims());
        if Net::am_master() {
            // Construct the final layers
            let claims = claims.unwrap();
            let leaves = (0..claims[0].len())
                .map(|i| claims.iter().map(|claim| claim[i]).collect())
                .collect();

            let final_circuit = Self::construct(leaves);
            (circuit, Some(final_circuit))
        } else {
            (circuit, None)
        }
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn claims(&self) -> Vec<F> {
        let num_layers = <BatchedDenseGrandProduct<F> as BatchedGrandProduct<F>>::num_layers(self);
        let last_layers = &self.layers[num_layers - 1];
        assert_eq!(last_layers.layer_len, 2);
        last_layers
            .layers
            .iter()
            .map(|layer| layer[0] * layer[1])
            .collect()
    }

    fn layers(&'_ mut self) -> impl Iterator<Item = &'_ mut dyn BatchedGrandProductLayer<F>> {
        self.layers
            .iter_mut()
            .map(|layer| layer as &mut dyn BatchedGrandProductLayer<F>)
            .rev()
    }
}

#[cfg(test)]
mod grand_product_tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    #[test]
    fn dense_prove_verify() {
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

        let mut batched_circuit =
            <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<Fr>>::construct(leaves);
        let mut transcript: IOPTranscript<Fr> = IOPTranscript::new(b"test_transcript");

        // I love the rust type system
        let claims =
            <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<Fr>>::claims(&batched_circuit);
        let (proof, r_prover) =
            <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<Fr>>::prove_grand_product(
                &mut batched_circuit,
                &mut transcript,
            );

        let mut transcript: IOPTranscript<Fr> = IOPTranscript::new(b"test_transcript");
        let (_, r_verifier) =
            BatchedDenseGrandProduct::verify_grand_product(&proof, &claims, &mut transcript);
        assert_eq!(r_prover, r_verifier);
    }
}

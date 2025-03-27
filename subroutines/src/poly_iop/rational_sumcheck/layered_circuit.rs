use crate::poly_iop::{
    sum_check::{
        batched_cubic_sumcheck::BatchedCubicSumcheckInstance,
        generic_sumcheck::ZerocheckInstanceProof,
    },
    utils::drop_in_background_thread,
};
use arithmetic::{
    bind_poly_var_bot, eq_poly::EqPolynomial, math::Math, unsafe_allocate_zero_vec, Fraction,
    OptimizedMul,
};
use ark_ff::{PrimeField, Zero};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::*;
use ark_std::{end_timer, start_timer};
use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};
use itertools::{izip, Itertools};
use rayon::prelude::*;
use std::{iter::zip, mem::take};
use transcript::IOPTranscript;

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchedRationalSumLayerProof<F: PrimeField> {
    pub proof: ZerocheckInstanceProof<F>,
    pub left_p: Vec<F>,
    pub left_q: Vec<F>,
    pub right_p: Vec<F>,
    pub right_q: Vec<F>,
}

impl<F: PrimeField> BatchedRationalSumLayerProof<F> {
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
pub struct BatchedRationalSumProof<F: PrimeField> {
    pub layers: Vec<BatchedRationalSumLayerProof<F>>,
}

pub trait BatchedRationalSum<F: PrimeField>: Sized {
    /// The bottom/input layer of the grand products
    type Leaves;

    /// Constructs the grand product circuit(s) from `leaves`
    fn construct(leaves: Self::Leaves) -> Self;

    type CompanionCircuit: BatchedRationalSum<F>;

    fn d_construct(leaves: Self::Leaves) -> (Self, Option<Self::CompanionCircuit>);

    /// The number of layers in the grand product
    fn num_layers(&self) -> usize;
    /// The claimed outputs of the rational sums (p, q)
    fn claims(&self) -> Vec<Fraction<F>>;
    /// Returns an iterator over the layers of this batched grand product
    /// circuit. Each layer is mutable so that its polynomials can be bound
    /// over the course of proving.
    fn layers(&'_ mut self) -> impl Iterator<Item = &'_ mut dyn BatchedRationalSumLayer<F>>;

    /// Computes a batched grand product proof, layer by layer.
    // #[tracing::instrument(skip_all, name =
    // "BatchedRationalSum::prove_rational_sum")]
    fn prove_rational_sum(
        &mut self,
        transcript: &mut IOPTranscript<F>,
    ) -> (BatchedRationalSumProof<F>, Vec<F>) {
        let mut proof_layers = Vec::with_capacity(self.num_layers());
        let mut claims_to_verify = self.claims();
        let mut r_rational_sum = Vec::new();

        for layer in self.layers() {
            proof_layers.push(layer.prove_layer(
                &mut claims_to_verify,
                &mut r_rational_sum,
                transcript,
            ));
        }
        r_rational_sum.reverse();

        (
            BatchedRationalSumProof {
                layers: proof_layers,
            },
            r_rational_sum,
        )
    }

    fn d_prove_rational_sum(
        &mut self,
        companion_circuit: Option<&mut Self::CompanionCircuit>,
        transcript: &mut IOPTranscript<F>,
    ) -> Option<(BatchedRationalSumProof<F>, Vec<F>)> {
        let mut proof_layers = Vec::with_capacity(self.num_layers());
        let mut claims_to_verify = Vec::new();
        let mut r_rational_sum = Vec::new();

        r_rational_sum = if Net::am_master() {
            let companion_circuit = companion_circuit.unwrap();
            claims_to_verify = companion_circuit.claims();
            for layer in companion_circuit.layers() {
                proof_layers.push(layer.prove_layer(
                    &mut claims_to_verify,
                    &mut r_rational_sum,
                    transcript,
                ));
            }
            Net::recv_from_master_uniform(Some(r_rational_sum))
        } else {
            Net::recv_from_master_uniform(None)
        };

        for layer in self.layers() {
            let layer_proof =
                layer.d_prove_layer(&mut claims_to_verify, &mut r_rational_sum, transcript);
            if Net::am_master() {
                proof_layers.push(layer_proof.unwrap());
            }
        }
        r_rational_sum.reverse();

        if Net::am_master() {
            Some((
                BatchedRationalSumProof {
                    layers: proof_layers,
                },
                r_rational_sum,
            ))
        } else {
            None
        }
    }

    /// Verifies that the `sumcheck_claim` output by sumcheck verification is
    /// consistent with the `left_claims` and `right_claims` of
    /// corresponding `BatchedRationalSumLayerProof`. This function may be
    /// overridden if the layer isn't just multiplication gates, e.g. in the
    /// case of `ToggledBatchedRationalSum`.
    fn verify_sumcheck_claim(
        layer_proofs: &[BatchedRationalSumLayerProof<F>],
        layer_index: usize,
        coeffs: &[F],
        sumcheck_claim: F,
        claims: &mut Vec<Fraction<F>>,
        r_rational_sum: &mut Vec<F>,
        lambda_layer: F,
        transcript: &mut IOPTranscript<F>,
    ) {
        let layer_proof = &layer_proofs[layer_index];
        let expected_sumcheck_claim: F = (0..claims.len())
            .map(|i| {
                coeffs[i]
                    * (layer_proof.right_p[i] * layer_proof.left_q[i]
                        + (layer_proof.left_p[i] + lambda_layer * layer_proof.left_q[i])
                            * layer_proof.right_q[i])
            })
            .sum();

        assert_eq!(expected_sumcheck_claim, sumcheck_claim);

        // produce a random challenge to condense two claims into a single claim
        let r_layer = transcript
            .get_and_append_challenge(b"challenge_r_layer")
            .unwrap();

        *claims = izip!(
            &layer_proof.left_p,
            &layer_proof.left_q,
            &layer_proof.right_p,
            &layer_proof.right_q
        )
        .map(|(&left_p, &left_q, &right_p, &right_q)| Fraction {
            p: left_p + (right_p - left_p) * r_layer,
            q: left_q + (right_q - left_q) * r_layer,
        })
        .collect();

        r_rational_sum.push(r_layer);
    }

    /// Function used for layer sumchecks in the generic batch verifier as well
    /// as the quark layered sumcheck hybrid
    fn verify_layers(
        proof_layers: &[BatchedRationalSumLayerProof<F>],
        claims: &Vec<Fraction<F>>,
        transcript: &mut IOPTranscript<F>,
        r_start: Vec<F>,
    ) -> (Vec<Fraction<F>>, Vec<F>) {
        let mut claims_to_verify = claims.to_owned();
        // We allow a non empty start in this function call because the quark hybrid
        // form provides prespecified random for most of the positions and then
        // we proceed with GKR on the remaining layers using the preset random values.
        // For default thaler '13 layered grand products this should be empty.
        let mut r_rational_sum = r_start.clone();
        let fixed_at_start = r_start.len();

        for (layer_index, layer_proof) in proof_layers.iter().enumerate() {
            // produce a fresh set of coeffs
            let coeffs: Vec<F> = transcript
                .get_and_append_challenge_vectors(b"rand_coeffs_next_layer", claims_to_verify.len())
                .unwrap();

            // Random combination for p and q
            let lambda_layer = transcript
                .get_and_append_challenge(b"challenge_lambda_layer")
                .unwrap();

            // produce a joint claim
            let claim = claims_to_verify
                .iter()
                .zip_eq(coeffs.iter())
                .map(|(Fraction { p, q }, &coeff)| *p * coeff + *q * coeff * lambda_layer)
                .sum();

            let step = start_timer!(|| "verify sumcheck");

            let mut zerocheck_r = r_rational_sum.clone();
            zerocheck_r.reverse();
            let (sumcheck_claim, r_sumcheck) = layer_proof.verify(
                claim,
                layer_index + fixed_at_start,
                2,
                &zerocheck_r,
                transcript,
            );

            end_timer!(step);
            assert_eq!(claims.len(), layer_proof.left_p.len());
            assert_eq!(claims.len(), layer_proof.right_p.len());
            assert_eq!(claims.len(), layer_proof.left_q.len());
            assert_eq!(claims.len(), layer_proof.right_q.len());

            for (left_p, left_q, right_p, right_q) in izip!(
                &layer_proof.left_p,
                &layer_proof.left_q,
                &layer_proof.right_p,
                &layer_proof.right_q
            ) {
                transcript
                    .append_field_element(b"sumcheck left claim p", &left_p)
                    .unwrap();
                transcript
                    .append_field_element(b"sumcheck left claim q", &left_q)
                    .unwrap();
                transcript
                    .append_field_element(b"sumcheck right claim p", &right_p)
                    .unwrap();
                transcript
                    .append_field_element(b"sumcheck right claim q", &right_q)
                    .unwrap();
            }

            assert_eq!(r_rational_sum.len(), r_sumcheck.len());

            r_rational_sum = r_sumcheck.into_iter().rev().collect();

            let step = start_timer!(|| "verify sumcheck claim");

            Self::verify_sumcheck_claim(
                proof_layers,
                layer_index,
                &coeffs,
                sumcheck_claim,
                &mut claims_to_verify,
                &mut r_rational_sum,
                lambda_layer,
                transcript,
            );

            end_timer!(step);
        }

        r_rational_sum.reverse();
        (claims_to_verify, r_rational_sum)
    }

    /// Verifies the given grand product proof.
    // #[tracing::instrument(skip_all, name =
    // "BatchedRationalSum::verify_rational_sum")]
    fn verify_rational_sum(
        proof: &BatchedRationalSumProof<F>,
        claims: &Vec<Fraction<F>>,
        transcript: &mut IOPTranscript<F>,
    ) -> (Vec<Fraction<F>>, Vec<F>) {
        // Pass the inputs to the layer verification function, by default we have no
        // quarks and so we do not use the quark proof fields.
        let r_start = Vec::<F>::new();
        Self::verify_layers(&proof.layers, claims, transcript, r_start)
    }
}

pub trait BatchedRationalSumLayer<F: PrimeField>: BatchedCubicSumcheckInstance<F> {
    /// Proves a single layer of a batched grand product circuit
    fn prove_layer(
        &mut self,
        claims: &mut Vec<Fraction<F>>,
        r_rational_sum: &mut Vec<F>,
        transcript: &mut IOPTranscript<F>,
    ) -> BatchedRationalSumLayerProof<F> {
        // produce a fresh set of coeffs
        let coeffs: Vec<F> = transcript
            .get_and_append_challenge_vectors(b"rand_coeffs_next_layer", claims.len())
            .unwrap();
        // Random combination for p and q
        let lambda_layer: F = transcript
            .get_and_append_challenge(b"challenge_lambda_layer")
            .unwrap();
        // produce a joint claim
        let claim = claims
            .iter()
            .zip_eq(coeffs.iter())
            .map(|(Fraction { p, q }, &coeff)| *p * coeff + *q * coeff * lambda_layer)
            .sum();

        let (sumcheck_proof, r_sumcheck, mut sumcheck_claims) =
            self.prove_sumcheck(&claim, &coeffs, &r_rational_sum, transcript, &lambda_layer);

        let (left_p, left_q, right_p, right_q) = (
            take(&mut sumcheck_claims[0]),
            take(&mut sumcheck_claims[1]),
            take(&mut sumcheck_claims[2]),
            take(&mut sumcheck_claims[3]),
        );
        for (left_p, left_q, right_p, right_q) in izip!(&left_p, &left_q, &right_p, &right_q) {
            transcript
                .append_field_element(b"sumcheck left claim p", &left_p)
                .unwrap();
            transcript
                .append_field_element(b"sumcheck left claim q", &left_q)
                .unwrap();
            transcript
                .append_field_element(b"sumcheck right claim p", &right_p)
                .unwrap();
            transcript
                .append_field_element(b"sumcheck right claim q", &right_q)
                .unwrap();
        }

        r_sumcheck
            .into_par_iter()
            .rev()
            .collect_into_vec(r_rational_sum);

        // produce a random challenge to condense two claims into a single claim
        let r_layer = transcript
            .get_and_append_challenge(b"challenge_r_layer")
            .unwrap();

        *claims = izip!(&left_p, &left_q, &right_p, &right_q)
            .map(|(&left_p, &left_q, &right_p, &right_q)| Fraction {
                p: left_p + (right_p - left_p) * r_layer,
                q: left_q + (right_q - left_q) * r_layer,
            })
            .collect();

        r_rational_sum.push(r_layer);

        BatchedRationalSumLayerProof {
            proof: sumcheck_proof,
            left_p,
            left_q,
            right_p,
            right_q,
        }
    }

    fn d_prove_layer(
        &mut self,
        claims: &mut Vec<Fraction<F>>,
        r_rational_sum: &mut Vec<F>,
        transcript: &mut IOPTranscript<F>,
    ) -> Option<BatchedRationalSumLayerProof<F>> {
        let (coeffs, lambda_layer) = if Net::am_master() {
            // produce a fresh set of coeffs
            let coeffs: Vec<F> = transcript
                .get_and_append_challenge_vectors(b"rand_coeffs_next_layer", claims.len())
                .unwrap();
            // Random combination for p and q
            let lambda_layer: F = transcript
                .get_and_append_challenge(b"challenge_lambda_layer")
                .unwrap();
            Net::recv_from_master_uniform(Some((coeffs, lambda_layer)))
        } else {
            Net::recv_from_master_uniform(None)
        };

        // produce a joint claim
        let claim = if Net::am_master() {
            claims
                .iter()
                .zip_eq(coeffs.iter())
                .map(|(Fraction { p, q }, &coeff)| *p * coeff + *q * coeff * lambda_layer)
                .sum()
        } else {
            // No need for non-masters to know the claim
            F::zero()
        };

        let num_party_vars = Net::n_parties().log_2();

        let proof =
            self.d_prove_sumcheck(&claim, &coeffs, &r_rational_sum, transcript, &lambda_layer);

        if Net::am_master() {
            let (sumcheck_proof, r_sumcheck, sumcheck_claims) = proof.unwrap();
            let (left_p_polys, left_q_polys, right_p_polys, right_q_polys): (
                Vec<_>,
                Vec<_>,
                Vec<_>,
                Vec<_>,
            ) = (0..sumcheck_claims[0][0].len())
                .map(|i| {
                    let (left_p, left_q, right_p, right_q) = sumcheck_claims
                        .iter()
                        .map(|claims| (claims[0][i], claims[1][i], claims[2][i], claims[3][i]))
                        .multiunzip();

                    (
                        DenseMultilinearExtension::from_evaluations_vec(num_party_vars, left_p),
                        DenseMultilinearExtension::from_evaluations_vec(num_party_vars, left_q),
                        DenseMultilinearExtension::from_evaluations_vec(num_party_vars, right_p),
                        DenseMultilinearExtension::from_evaluations_vec(num_party_vars, right_q),
                    )
                })
                .multiunzip();

            let r_party = &r_sumcheck[r_sumcheck.len() - num_party_vars..];

            let left_p = left_p_polys
                .iter()
                .map(|poly| poly.evaluate(&r_party).unwrap())
                .collect::<Vec<_>>();
            let left_q = left_q_polys
                .iter()
                .map(|poly| poly.evaluate(&r_party).unwrap())
                .collect::<Vec<_>>();
            let right_p = right_p_polys
                .iter()
                .map(|poly| poly.evaluate(&r_party).unwrap())
                .collect::<Vec<_>>();
            let right_q = right_q_polys
                .iter()
                .map(|poly| poly.evaluate(&r_party).unwrap())
                .collect::<Vec<_>>();

            for (left_p, left_q, right_p, right_q) in izip!(&left_p, &left_q, &right_p, &right_q) {
                transcript
                    .append_field_element(b"sumcheck left claim p", &left_p)
                    .unwrap();
                transcript
                    .append_field_element(b"sumcheck left claim q", &left_q)
                    .unwrap();
                transcript
                    .append_field_element(b"sumcheck right claim p", &right_p)
                    .unwrap();
                transcript
                    .append_field_element(b"sumcheck right claim q", &right_q)
                    .unwrap();
            }

            r_sumcheck
                .into_par_iter()
                .rev()
                .collect_into_vec(r_rational_sum);

            // produce a random challenge to condense two claims into a single claim
            let r_layer = transcript
                .get_and_append_challenge(b"challenge_r_layer")
                .unwrap();

            *claims = izip!(&left_p, &left_q, &right_p, &right_q)
                .map(|(&left_p, &left_q, &right_p, &right_q)| Fraction {
                    p: left_p + (right_p - left_p) * r_layer,
                    q: left_q + (right_q - left_q) * r_layer,
                })
                .collect();

            r_rational_sum.push(r_layer);
            Net::recv_from_master_uniform(Some(r_rational_sum.clone()));

            Some(BatchedRationalSumLayerProof {
                proof: sumcheck_proof,
                left_p,
                left_q,
                right_p,
                right_q,
            })
        } else {
            *r_rational_sum = Net::recv_from_master_uniform(None);
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
pub type DenseRationalSumLayer<F> = Vec<F>;

/// Represents a batch of `DenseRationalSumLayer`, all of the same length
/// `layer_len`.
#[derive(Debug, Clone)]
pub struct BatchedDenseRationalSumLayer<F: PrimeField, const C: usize> {
    pub layers_p: Vec<Vec<F>>,
    pub layers_q: Vec<Vec<F>>,
    pub layer_len: usize,
}

impl<F: PrimeField, const C: usize> BatchedDenseRationalSumLayer<F, C> {
    pub fn new(layers_p: Vec<Vec<F>>, layers_q: Vec<Vec<F>>) -> Self {
        let layer_len = layers_p[0].len();
        Self {
            layers_p,
            layers_q,
            layer_len,
        }
    }
}

fn compute_cubic_direct<F: PrimeField>(
    coeffs: &[F],
    evaluations: &[Vec<DenseMultilinearExtension<F>>],
    eq_table: &[F],
    lambda: &F,
) -> (F, F) {
    (0..eq_table.len())
        .into_par_iter()
        .map(|i| {
            let mut evals = (F::zero(), F::zero());

            izip!(
                &evaluations[0],
                &evaluations[1],
                &evaluations[2],
                &evaluations[3]
            )
            .enumerate()
            .for_each(|(batch_index, (left_p, left_q, right_p, right_q))| {
                // We want to compute:
                //     evals.0 += coeff * left.0 * right.0
                //     evals.1 += coeff * (2 * left.1 - left.0) * (2 * right.1 - right.0)
                //     evals.2 += coeff * (3 * left.1 - 2 * left.0) * (3 * right.1 - 2 *
                // right.0) which naively requires 3 multiplications
                // by `coeff`. By multiplying by the coefficient
                // early, we only use 2 multiplications by `coeff`.
                let left = (
                    Fraction {
                        p: left_p.evaluations[2 * i],
                        q: left_q.evaluations[2 * i],
                    },
                    Fraction {
                        p: left_p.evaluations[2 * i + 1],
                        q: left_q.evaluations[2 * i + 1],
                    },
                );
                let right = (
                    Fraction {
                        p: right_p.evaluations[2 * i],
                        q: right_q.evaluations[2 * i],
                    },
                    Fraction {
                        p: right_p.evaluations[2 * i + 1],
                        q: right_q.evaluations[2 * i + 1],
                    },
                );

                let m_left = left.1 - left.0;
                let m_right = right.1 - right.0;

                let left_eval_2 = left.1 + m_left;
                let right_eval_2 = right.1 + m_right;

                let effective_coeff = coeffs[batch_index] * eq_table[i];
                evals.0 += effective_coeff
                    * (right.0.p * left.0.q + (left.0.p + *lambda * left.0.q) * right.0.q);
                evals.1 += effective_coeff
                    * (right_eval_2.p * left_eval_2.q
                        + (left_eval_2.p + *lambda * left_eval_2.q) * right_eval_2.q);
            });

            evals
        })
        .reduce(
            || (F::zero(), F::zero()),
            |sum, evals| (sum.0 + evals.0, sum.1 + evals.1),
        )
}

impl<F: PrimeField, const C: usize> BatchedRationalSumLayer<F>
    for BatchedDenseRationalSumLayer<F, C>
{
}
impl<F: PrimeField, const C: usize> BatchedCubicSumcheckInstance<F>
    for BatchedDenseRationalSumLayer<F, C>
{
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
    // #[tracing::instrument(skip_all, name = "BatchedDenseRationalSumLayer::bind")]
    fn bind(&mut self, r: &F) {
        debug_assert!(self.layer_len % 4 == 0);
        // TODO(moodlezoup): parallelize over chunks instead of over batch
        rayon::join(
            || {
                self.layers_p.par_iter_mut().for_each(|layer: &mut Vec<F>| {
                    let mut new_layer = unsafe_allocate_zero_vec::<F>(self.layer_len / 2);
                    new_layer
                        .par_chunks_exact_mut(2)
                        .zip_eq(layer.par_chunks_exact(4))
                        .for_each(|(new_chunk, chunk)| {
                            new_chunk[0] = chunk[0] + *r * (chunk[2] - chunk[0]);
                            new_chunk[1] = chunk[1] + *r * (chunk[3] - chunk[1]);
                        });
                    *layer = new_layer;
                })
            },
            || {
                self.layers_q.par_iter_mut().for_each(|layer: &mut Vec<F>| {
                    let mut new_layer = unsafe_allocate_zero_vec::<F>(self.layer_len / 2);
                    new_layer
                        .par_chunks_exact_mut(2)
                        .zip_eq(layer.par_chunks_exact(4))
                        .for_each(|(new_chunk, chunk)| {
                            new_chunk[0] = chunk[0] + *r * (chunk[2] - chunk[0]);
                            new_chunk[1] = chunk[1] + *r * (chunk[3] - chunk[1]);
                        });
                    *layer = new_layer;
                })
            },
        );
        self.layer_len /= 2;
    }

    /// We want to compute the evaluations of the following univariate cubic
    /// polynomial at points {0, 1, 2, 3}:
    ///     Σ coeff[batch_index] * (Σ eq(r, x) * (right(x).p * left(x).q +
    /// (left(x).p + lambda * left(x).q) * right(x).q)) where the inner
    /// summation is over all but the "least significant bit" of the multilinear
    /// polynomials `eq`, `left`, and `right`. We denote this "least
    /// significant" variable x_b.
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
    // "BatchedDenseRationalSumLayer::compute_cubic")]
    fn compute_cubic(&self, coeffs: &[F], eq_table: &[F], lambda: &F) -> (F, F) {
        (0..eq_table.len())
            .into_par_iter()
            .map(|i| {
                let mut evals = (F::zero(), F::zero());

                for subtable_index in 0..self.layers_p.len() / C {
                    let left_q = (
                        self.layers_q[subtable_index][4 * i],
                        self.layers_q[subtable_index][4 * i + 2],
                    );
                    let right_q = (
                        self.layers_q[subtable_index][4 * i + 1],
                        self.layers_q[subtable_index][4 * i + 3],
                    );

                    let m_left_q = left_q.1 - left_q.0;
                    let m_right_q = right_q.1 - right_q.0;
                    let left_q_eval_2 = left_q.1 + m_left_q;
                    let right_q_eval_2 = right_q.1 + m_right_q;

                    let base = (
                        *lambda * left_q.0 * right_q.0,
                        *lambda * left_q_eval_2 * right_q_eval_2,
                    );
                    for batch_index in subtable_index * C..(subtable_index + 1) * C {
                        let layer = &self.layers_p[batch_index];
                        let left = (layer[4 * i], layer[4 * i + 2]);
                        let right = (layer[4 * i + 1], layer[4 * i + 3]);

                        let m_left = left.1 - left.0;
                        let m_right = right.1 - right.0;
                        let left_eval_2 = left.1 + m_left;
                        let right_eval_2 = right.1 + m_right;

                        let effective_coeff = coeffs[batch_index] * eq_table[i];
                        evals.0 +=
                            effective_coeff * (base.0 + left.0 * right_q.0 + right.0 * left_q.0);
                        evals.1 += effective_coeff
                            * (base.1
                                + left_eval_2 * right_q_eval_2
                                + right_eval_2 * left_q_eval_2);
                    }
                }
                evals
            })
            .reduce(
                || (F::zero(), F::zero()),
                |sum, evals| (sum.0 + evals.0, sum.1 + evals.1),
            )
    }

    fn final_claims(&self) -> Vec<Vec<F>> {
        assert_eq!(self.layer_len, 2);
        let (left_p, right_p) = self
            .layers_p
            .iter()
            .map(|layer| (layer[0], layer[1]))
            .unzip();
        let (left_q, right_q) = (0..self.layers_p.len())
            .map(|i| (self.layers_q[i / C][0], self.layers_q[i / C][1]))
            .unzip();
        vec![left_p, left_q, right_p, right_q]
    }

    fn compute_cubic_direct(
        &self,
        coeffs: &[F],
        evaluations: &[Vec<DenseMultilinearExtension<F>>],
        eq_table: &[F],
        lambda: &F,
    ) -> (F, F) {
        compute_cubic_direct(coeffs, evaluations, eq_table, lambda)
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
pub struct BatchedDenseRationalSum<F: PrimeField, const C: usize> {
    layers: Vec<BatchedDenseRationalSumLayer<F, C>>,
}

impl<F: PrimeField, const C: usize> BatchedRationalSum<F> for BatchedDenseRationalSum<F, C> {
    type Leaves = (Vec<Vec<F>>, Vec<Vec<F>>);

    // #[tracing::instrument(skip_all, name = "BatchedDenseRationalSum::construct")]
    fn construct(leaves: Self::Leaves) -> Self {
        let (leaves_p, leaves_q) = leaves;
        let num_layers = leaves_p[0].len().log_2();
        let mut layers: Vec<BatchedDenseRationalSumLayer<F, C>> = Vec::with_capacity(num_layers);
        layers.push(BatchedDenseRationalSumLayer::new(leaves_p, leaves_q));

        for i in 0..num_layers - 1 {
            let previous_layers = &layers[i];
            let len = previous_layers.layer_len / 2;
            // TODO(moodlezoup): parallelize over chunks instead of over batch

            let (new_layers_p, new_layers_q) = rayon::join(
                || {
                    previous_layers
                        .layers_p
                        .par_iter()
                        .enumerate()
                        .map(|(memory_idx, previous_layer_p)| {
                            let subtable_idx = memory_idx / C;
                            let previous_layer_q = &previous_layers.layers_q[subtable_idx];

                            (0..len)
                                .into_par_iter()
                                .map(|i| {
                                    previous_layer_p[2 * i] * previous_layer_q[2 * i + 1]
                                        + previous_layer_p[2 * i + 1] * previous_layer_q[2 * i]
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                },
                || {
                    previous_layers
                        .layers_q
                        .par_iter()
                        .map(|previous_layer| {
                            (0..len)
                                .into_par_iter()
                                .map(|i| previous_layer[2 * i] * previous_layer[2 * i + 1])
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                },
            );
            layers.push(BatchedDenseRationalSumLayer::new(
                new_layers_p,
                new_layers_q,
            ));
        }

        Self { layers }
    }

    type CompanionCircuit = BatchedDenseRationalSum<F, 1>;

    fn d_construct(leaves: Self::Leaves) -> (Self, Option<Self::CompanionCircuit>) {
        let circuit = Self::construct(leaves);

        let claims = Net::send_to_master(&circuit.claims());
        if Net::am_master() {
            // Construct the final layers
            let claims = claims.unwrap();
            let leaves = (0..claims[0].len())
                .map(|i| claims.iter().map(|claim| (claim[i].p, claim[i].q)).unzip())
                .unzip();

            let final_circuit = Self::CompanionCircuit::construct(leaves);
            (circuit, Some(final_circuit))
        } else {
            (circuit, None)
        }
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn claims(&self) -> Vec<Fraction<F>> {
        let num_layers = <BatchedDenseRationalSum<F, C> as BatchedRationalSum<F>>::num_layers(self);
        let last_layers = &self.layers[num_layers - 1];
        assert_eq!(last_layers.layer_len, 2);
        last_layers
            .layers_p
            .iter()
            .enumerate()
            .map(|(i, layer_p)| {
                let layer_q = &last_layers.layers_q[i / C];
                Fraction {
                    p: layer_p[0] * layer_q[1] + layer_p[1] * layer_q[0],
                    q: layer_q[0] * layer_q[1],
                }
            })
            .collect()
    }

    fn layers(&'_ mut self) -> impl Iterator<Item = &'_ mut dyn BatchedRationalSumLayer<F>> {
        self.layers
            .iter_mut()
            .map(|layer| layer as &mut dyn BatchedRationalSumLayer<F>)
            .rev()
    }
}

/// Represents a single layer of a single grand product circuit using a sparse
/// vector, i.e. a vector containing (index, value) pairs.
/// with values {p: 0, q: 1} omitted
pub type SparseRationalSumLayer<F> = Vec<(usize, F)>;

/// A "dynamic density" grand product layer can switch from sparse
/// representation to dense representation once it's no longer sparse (after
/// binding).
#[derive(Debug, Clone, PartialEq)]
pub enum DynamicDensityRationalSumLayer<F: PrimeField> {
    Sparse(SparseRationalSumLayer<F>),
    Dense(DenseRationalSumLayer<F>),
}

/// This constant determines:
///     - whether the `layer_output` of a `DynamicDensityRationalSumLayer` is
///       dense or sparse
///     - when to switch from sparse to dense representation during the binding
///       of a `DynamicDensityRationalSumLayer`
/// If the layer has >DENSIFICATION_THRESHOLD fraction of non-1 values, it'll
/// switch to the dense representation. Value tuned experimentally.
const DENSIFICATION_THRESHOLD: f64 = 0.8;

impl<F: PrimeField> DynamicDensityRationalSumLayer<F> {
    /// Computes the grand product layer that is output by this layer.
    ///     L0'      R0'      L1'      R1'     <- output layer
    ///      Λ        Λ        Λ        Λ
    ///     / \      / \      / \      / \
    ///   L0   R0  L1   R1  L2   R2  L3   R3   <- this layer
    ///
    /// If the current layer is dense, the output layer will be dense.
    /// If the current layer is sparse, but already not very sparse (as
    /// parametrized by `DENSIFICATION_THRESHOLD`), the output layer will be
    /// dense. Otherwise, the output layer will be sparse.
    pub fn layer_output<const C: usize>(
        &self,
        output_len: usize,
        memory_index: usize,
        preprocessing: &Vec<Vec<F>>,
    ) -> Self {
        let subtable_index = memory_index / C;
        let layer_q = &preprocessing[subtable_index];

        match self {
            DynamicDensityRationalSumLayer::Sparse(sparse_layer) => {
                if (sparse_layer.len() as f64 / (output_len * 2) as f64) > DENSIFICATION_THRESHOLD {
                    // Current layer is already not very sparse, so make the next layer dense
                    let mut output_layer: DenseRationalSumLayer<F> = vec![F::zero(); output_len];
                    let mut next_index_to_process = 0usize;
                    for (j, (index, value)) in sparse_layer.iter().enumerate() {
                        if *index < next_index_to_process {
                            // Node was already multiplied with its sibling in a previous iteration
                            continue;
                        }
                        if index % 2 == 0 {
                            // Left node; try to find correspoding right node
                            let right = sparse_layer
                                .get(j + 1)
                                .cloned()
                                .unwrap_or((index + 1, F::zero()));
                            if right.0 == index + 1 {
                                output_layer[index / 2] =
                                    right.1 * layer_q[*index] + *value * layer_q[index + 1];
                            } else {
                                output_layer[index / 2] = *value * layer_q[index + 1];
                            }
                            next_index_to_process = index + 2;
                        } else {
                            // Right node; corresponding left node was not encountered in
                            // previous iteration
                            output_layer[index / 2] = *value * layer_q[index - 1];
                            next_index_to_process = index + 1;
                        }
                    }
                    DynamicDensityRationalSumLayer::Dense(output_layer)
                } else {
                    // Current layer is still pretty sparse, so make the next layer sparse
                    let mut output_layer: SparseRationalSumLayer<F> =
                        Vec::with_capacity(output_len);
                    let mut next_index_to_process = 0usize;
                    for (j, (index, value)) in sparse_layer.iter().enumerate() {
                        if *index < next_index_to_process {
                            // Node was already multiplied with its sibling in a previous iteration
                            continue;
                        }
                        if index % 2 == 0 {
                            // Left node; try to find correspoding right node
                            let right = sparse_layer
                                .get(j + 1)
                                .cloned()
                                .unwrap_or((index + 1, F::zero()));
                            if right.0 == index + 1 {
                                // Corresponding right node was found; multiply them together
                                output_layer.push((
                                    index / 2,
                                    right.1 * layer_q[*index] + *value * layer_q[index + 1],
                                ));
                            } else {
                                // Corresponding right node not found
                                output_layer.push((index / 2, *value * layer_q[index + 1]));
                            }
                            next_index_to_process = index + 2;
                        } else {
                            // Right node; corresponding left node was not encountered in
                            // previous iteration
                            output_layer.push((index / 2, *value * layer_q[index - 1]));
                            next_index_to_process = index + 1;
                        }
                    }
                    DynamicDensityRationalSumLayer::Sparse(output_layer)
                }
            },
            DynamicDensityRationalSumLayer::Dense(dense_layer) => {
                // If current layer is dense, next layer should also be dense.
                let output_layer: DenseRationalSumLayer<F> = (0..output_len)
                    .into_par_iter()
                    .map(|i| {
                        dense_layer[2 * i] * layer_q[2 * i + 1]
                            + dense_layer[2 * i + 1] * layer_q[2 * i]
                    })
                    .collect();
                DynamicDensityRationalSumLayer::Dense(output_layer)
            },
        }
    }
}

/// Represents a batch of `DynamicDensityRationalSumLayer`, all of which have
/// the same size `layer_len`. Note that within a single batch, some layers may
/// be represented by sparse vectors and others by dense vectors.
#[derive(Debug, Clone)]
pub struct BatchedSparseRationalSumLayer<F: PrimeField, const C: usize> {
    pub layer_len: usize,
    pub layers: Vec<DynamicDensityRationalSumLayer<F>>,
    pub preprocessing: Vec<Vec<F>>,
    pub preprocessing_next: Vec<Vec<F>>,
}

impl<F: PrimeField, const C: usize> BatchedSparseRationalSumLayer<F, C> {
    fn memory_to_subtable_index(i: usize) -> usize {
        i / C
    }
}

impl<F: PrimeField, const C: usize> BatchedRationalSumLayer<F>
    for BatchedSparseRationalSumLayer<F, C>
{
}
impl<F: PrimeField, const C: usize> BatchedCubicSumcheckInstance<F>
    for BatchedSparseRationalSumLayer<F, C>
{
    fn num_rounds(&self) -> usize {
        self.layer_len.log_2() - 1
    }

    /// Incrementally binds a variable of this batched layer's polynomials.
    /// If `self` is dense, we bind as in `BatchedDenseRationalSumLayer`,
    /// processing nodes 4 at a time to preserve the interleaved order:
    ///   0'  1'     2'  3'
    ///   |\ |\      |\ |\
    ///   | \| \     | \| \
    ///   |  \  \    |  \  \
    ///   |  |\  \   |  |\  \
    ///   0  1 2  3  4  5 6  7
    /// Left nodes have even indices, right nodes have odd indices.
    /// If `self` is sparse, we basically do the same thing but with more
    /// cases to check 😬
    // #[tracing::instrument(skip_all, name =
    // "BatchedSparseRationalSumLayer::bind")]
    fn bind(&mut self, r: &F) {
        debug_assert!(self.layer_len % 4 == 0);

        self.preprocessing_next =
            vec![unsafe_allocate_zero_vec::<F>(self.layer_len / 2); self.preprocessing.len()];

        (
            self.layers.par_chunks_mut(C),
            &mut self.preprocessing_next,
            &self.preprocessing,
        )
            .into_par_iter()
            .for_each(|(layers, preprocessing_next, preprocessing)| {
                preprocessing_next
                    .par_chunks_exact_mut(2)
                    .zip_eq(preprocessing.par_chunks_exact(4))
                    .for_each(|(next, cur)| {
                        next[0] = cur[0] + (cur[2] - cur[0]) * *r;
                        next[1] = cur[1] + (cur[3] - cur[1]) * *r;
                    });

                layers.par_iter_mut().for_each(|layer| match layer {
                    DynamicDensityRationalSumLayer::Sparse(sparse_layer) => {
                        let mut dense_bound_layer = if (sparse_layer.len() as f64
                            / self.layer_len as f64)
                            > DENSIFICATION_THRESHOLD
                        {
                            // Current layer is already not very sparse, so make the next
                            // layer dense
                            Some(vec![F::zero(); preprocessing_next.len()])
                        } else {
                            None
                        };

                        let mut num_bound = 0usize;
                        let mut push_to_bound_layer =
                            |sparse_layer: &mut Vec<(usize, F)>, dense_index: usize, value: F| {
                                match &mut dense_bound_layer {
                                    Some(ref mut dense_vec) => {
                                        debug_assert_eq!(dense_vec[dense_index], F::zero());
                                        dense_vec[dense_index] = value;
                                    },
                                    None => {
                                        sparse_layer[num_bound] = (dense_index, value);
                                    },
                                };
                                num_bound += 1;
                            };

                        let mut next_left_node_to_process = 0usize;
                        let mut next_right_node_to_process = 0usize;

                        for j in 0..sparse_layer.len() {
                            let (index, value) = sparse_layer[j];
                            if index % 2 == 0 && index < next_left_node_to_process {
                                // This left node was already bound with its sibling in a
                                // previous iteration
                                continue;
                            }
                            if index % 2 == 1 && index < next_right_node_to_process {
                                // This right node was already bound with its sibling in a
                                // previous iteration
                                continue;
                            }

                            let mut neighbors = vec![None, None];
                            for k in [j + 1, j + 2] {
                                if let Some((idx, val)) = sparse_layer.get(k) {
                                    if *idx == index + 1 {
                                        neighbors[0] = Some(*val);
                                    } else if *idx == index + 2 {
                                        neighbors[1] = Some(*val);
                                    } else {
                                        break;
                                    }
                                }
                            }

                            match index % 4 {
                                0 => {
                                    // Find sibling left node
                                    let sibling_value = neighbors[1].unwrap_or(F::zero());
                                    push_to_bound_layer(
                                        sparse_layer,
                                        index / 2,
                                        value + (sibling_value - value) * *r,
                                    );
                                    next_left_node_to_process = index + 4;
                                },
                                1 => {
                                    // Edge case: If this right node's neighbor is not 1 and
                                    // has _not_
                                    // been bound yet, we need to bind the neighbor first to
                                    // preserve
                                    // the monotonic ordering of the bound layer.
                                    if next_left_node_to_process <= index + 1 {
                                        if let Some(left_neighbor) = neighbors[0] {
                                            push_to_bound_layer(
                                                sparse_layer,
                                                index / 2,
                                                left_neighbor * *r,
                                            );
                                        }
                                        next_left_node_to_process = index + 3;
                                    }

                                    // Find sibling right node
                                    let sibling_value = neighbors[1].unwrap_or(F::zero());
                                    push_to_bound_layer(
                                        sparse_layer,
                                        index / 2 + 1,
                                        value + (sibling_value - value) * *r,
                                    );
                                    next_right_node_to_process = index + 4;
                                },
                                2 => {
                                    // Sibling left node wasn't encountered in previous
                                    // iteration,
                                    // so sibling must have value 1.
                                    push_to_bound_layer(sparse_layer, index / 2 - 1, value * *r);
                                    next_left_node_to_process = index + 2;
                                },
                                3 => {
                                    // Sibling right node wasn't encountered in previous
                                    // iteration,
                                    // so sibling must have value 1.
                                    push_to_bound_layer(sparse_layer, index / 2, value * *r);
                                    next_right_node_to_process = index + 2;
                                },
                                _ => unreachable!("?_?"),
                            }
                        }
                        if let Some(dense_vec) = dense_bound_layer {
                            *layer = DynamicDensityRationalSumLayer::Dense(dense_vec);
                        } else {
                            sparse_layer.truncate(num_bound);
                        }
                    },
                    DynamicDensityRationalSumLayer::Dense(dense_layer) => {
                        // If current layer is dense, next layer should also be dense.
                        let mut new_layer = unsafe_allocate_zero_vec::<F>(self.layer_len / 2);
                        new_layer
                            .par_chunks_exact_mut(2)
                            .zip_eq(dense_layer.par_chunks_exact(4))
                            .for_each(|(new_chunk, chunk)| {
                                new_chunk[0] = chunk[0] + *r * (chunk[2] - chunk[0]);
                                new_chunk[1] = chunk[1] + *r * (chunk[3] - chunk[1]);
                            });
                        *dense_layer = new_layer;
                    },
                })
            });
        self.layer_len /= 2;

        self.preprocessing = take(&mut self.preprocessing_next);
    }

    /// We want to compute the evaluations of the following univariate cubic
    /// polynomial at points {0, 1, 2, 3}:
    ///     Σ coeff[batch_index] * (Σ eq(r, x) * (right(x).p * left(x).q +
    /// (left(x).p + lambda * left(x).q) * right(x).q)) where the inner
    /// summation is over all but the "least significant bit" of the multilinear
    /// polynomials `eq`, `left`, and `right`. We denote this "least
    /// significant" variable x_b.
    ///
    /// Computing these evaluations requires processing pairs of adjacent
    /// coefficients of `eq`, `left`, and `right`.
    /// If `self` is dense, we process each layer 4 values at a time:
    ///                  layer = [L, R, L, R, L, R, ...]
    ///                           |  |  |  |
    ///    left(0, 0, 0, ..., x_b=0) |  |  right(0, 0, 0, ..., x_b=1)
    ///     right(0, 0, 0, ..., x_b=0)  left(0, 0, 0, ..., x_b=1)
    /// If `self` is sparse, we basically do the same thing but with some fancy
    /// optimizations and more cases to check 😬
    // #[tracing::instrument(skip_all, name =
    // "BatchedSparseRationalSumLayer::compute_cubic")]
    fn compute_cubic(&self, coeffs: &[F], eq_table: &[F], lambda: &F) -> (F, F) {
        let inner_func_delta =
            |left: Fraction<F>, right: Fraction<F>| right.p * left.q + left.p * right.q;

        (
            self.layers.par_chunks(C),
            coeffs.par_chunks(C),
            &self.preprocessing,
        )
            .into_par_iter()
            .map(|(layers, coeffs, preprocessing)| {
                let mut sum = (F::zero(), F::zero());
                for i in 0..preprocessing.len() / 4 {
                    let left = (preprocessing[4 * i], preprocessing[4 * i + 2]);
                    let right = (preprocessing[4 * i + 1], preprocessing[4 * i + 3]);

                    let m_left = left.1 - left.0;
                    let m_right = right.1 - right.0;

                    let left_eval_2 = left.1 + m_left;
                    let right_eval_2 = right.1 + m_right;

                    sum.0 += eq_table[i] * left.0 * right.0;
                    sum.1 += eq_table[i] * left_eval_2 * right_eval_2;
                }
                let default_sum = (sum.0 * *lambda, sum.1 * *lambda);

                layers
                    .par_iter()
                    .zip_eq(coeffs.par_iter())
                    .map(|(layer, coeff)| match layer {
                        // If sparse, we use the pre-emptively computed `eq_eval_sums` as a starting
                        // point:     eq_eval_sum := Σ eq_evals[i]
                        // What we ultimately want to compute:
                        //     Σ coeff[batch_index] * (Σ eq(r, x) * (right(x).p * left(x).q +
                        // (left(x).p
                        // + lambda * left(x).q) * right(x).q)) Note that if left[i]
                        // and right[i] are all 0s, the inner sum is:
                        //     Σ eq_evals[i] = eq_eval_sum * lambda
                        // To recover the actual inner sum, we find all the non-0
                        // left[i] and right[i] terms and compute the delta:
                        //     ∆ := Σ eq_evals[j] * ((right(x).p * left(x).q + (left(x).p + lambda *
                        // left(x).q) * right(x).q) - lambda) Then we can compute:
                        //    coeff[batch_index] * (eq_eval_sum + ∆)
                        // ...which is exactly the summand we want.
                        DynamicDensityRationalSumLayer::Sparse(sparse_layer) => {
                            let layer_q = preprocessing;
                            let get_default = |i| Fraction {
                                p: F::zero(),
                                q: layer_q[i],
                            };
                            // Computes:
                            //     ∆ := Σ eq_evals[j] * (left[j] * right[j] - 1)    ∀j where left[j]
                            // ≠ 1 or right[j] ≠ 1 for the evaluation
                            // points {0, 2, 3}
                            let mut sum = default_sum;

                            let mut next_index_to_process = 0usize;
                            for (j, (index, value)) in sparse_layer.iter().enumerate() {
                                if *index < next_index_to_process {
                                    // This node was already processed in a previous iteration
                                    continue;
                                }

                                let mut neighbors = vec![None, None, None];
                                for k in [j + 1, j + 2, j + 3] {
                                    if let Some((idx, val)) = sparse_layer.get(k) {
                                        if *idx == index + 1 {
                                            neighbors[0] = Some(*val);
                                        } else if *idx == index + 2 {
                                            neighbors[1] = Some(*val);
                                        } else if *idx == index + 3 {
                                            neighbors[2] = Some(*val);
                                            break;
                                        } else {
                                            break;
                                        }
                                    }
                                }
                                for k in 0..3 {
                                    if neighbors[k] == None {
                                        neighbors[k] = Some(F::zero());
                                    }
                                }

                                let find_neighbor = |i: usize| Fraction {
                                    p: neighbors[i - index - 1].unwrap(),
                                    q: layer_q[i],
                                };

                                // Recall that in the dense case, we process four values at a time:
                                //                  layer = [L, R, L, R, L, R, ...]
                                //                           |  |  |  |
                                //    left(0, 0, 0, ..., x_b=0) |  |  right(0, 0, 0, ..., x_b=1)
                                //     right(0, 0, 0, ..., x_b=0)  left(0, 0, 0, ..., x_b=1)
                                //
                                // In the sparse case, we do something similar, but some of the four
                                // values may be omitted from the sparse vector.
                                // We match on `index % 4` to determine which of the four values are
                                // present in the sparse vector, and infer the rest are 1.
                                let value = Fraction {
                                    p: *value,
                                    q: layer_q[*index],
                                };
                                let (left, right) = match index % 4 {
                                    0 => {
                                        let left = (value, find_neighbor(index + 2));
                                        let right =
                                            (find_neighbor(index + 1), find_neighbor(index + 3));
                                        next_index_to_process = index + 4;
                                        (left, right)
                                    },
                                    1 => {
                                        let left =
                                            (get_default(index - 1), find_neighbor(index + 1));
                                        let right = (value, find_neighbor(index + 2));
                                        next_index_to_process = index + 3;
                                        (left, right)
                                    },
                                    2 => {
                                        let left = (get_default(index - 2), value);
                                        let right =
                                            (get_default(index - 1), find_neighbor(index + 1));
                                        next_index_to_process = index + 2;
                                        (left, right)
                                    },
                                    3 => {
                                        let left = (get_default(index - 3), get_default(index - 1));
                                        let right = (get_default(index - 2), value);
                                        next_index_to_process = index + 1;
                                        (left, right)
                                    },
                                    _ => unreachable!("?_?"),
                                };

                                let m_left = left.1 - left.0;
                                let m_right = right.1 - right.0;

                                let left_eval_2 = left.1 + m_left;
                                let right_eval_2 = right.1 + m_right;

                                sum.0 += eq_table[index / 4]
                                    .mul_0_optimized(inner_func_delta(left.0, right.0));
                                sum.1 += eq_table[index / 4]
                                    .mul_0_optimized(inner_func_delta(left_eval_2, right_eval_2));
                            }

                            (*coeff * sum.0, *coeff * sum.1)
                        },
                        // If dense, we just compute
                        //     Σ coeff[batch_index] * (Σ eq_evals[i] * left[i] * right[i])
                        // directly in `self.compute_cubic`, without using `eq_eval_sums`.
                        DynamicDensityRationalSumLayer::Dense(dense_layer) => {
                            let layer_q = preprocessing;
                            // Computes:
                            //     coeff[batch_index] * (Σ eq_evals[i] * left[i] * right[i])
                            // for the evaluation points {0, 2, 3}
                            let evals = eq_table
                                .par_iter()
                                .zip_eq(dense_layer.par_chunks_exact(4))
                                .enumerate()
                                .map(|(i, (eq_eval, chunk))| {
                                    let left = (
                                        Fraction {
                                            p: chunk[0],
                                            q: layer_q[4 * i],
                                        },
                                        Fraction {
                                            p: chunk[2],
                                            q: layer_q[4 * i + 2],
                                        },
                                    );
                                    let right = (
                                        Fraction {
                                            p: chunk[1],
                                            q: layer_q[4 * i + 1],
                                        },
                                        Fraction {
                                            p: chunk[3],
                                            q: layer_q[4 * i + 3],
                                        },
                                    );

                                    let m_left = left.1 - left.0;
                                    let m_right = right.1 - right.0;

                                    let left_eval_2 = left.1 + m_left;
                                    let right_eval_2 = right.1 + m_right;

                                    (
                                        *eq_eval * inner_func_delta(left.0, right.0),
                                        *eq_eval * inner_func_delta(left_eval_2, right_eval_2),
                                    )
                                })
                                .reduce(
                                    || (F::zero(), F::zero()),
                                    |(sum_0, sum_2), (a, b)| (sum_0 + a, sum_2 + b),
                                );
                            (
                                *coeff * (evals.0 + default_sum.0),
                                *coeff * (evals.1 + default_sum.1),
                            )
                        },
                    })
                    .reduce(
                        || (F::zero(), F::zero()),
                        |(sum_0, sum_2), (a, b)| (sum_0 + a, sum_2 + b),
                    )
            })
            .reduce(
                || (F::zero(), F::zero()),
                |(sum_0, sum_2), (a, b)| (sum_0 + a, sum_2 + b),
            )
    }

    fn final_claims(&self) -> Vec<Vec<F>> {
        assert_eq!(self.layer_len, 2);
        let (left, right): (Vec<_>, Vec<_>) = self
            .layers
            .iter()
            .enumerate()
            .map(|(batch_index, layer)| match layer {
                DynamicDensityRationalSumLayer::Sparse(layer) => {
                    let subtable_index = Self::memory_to_subtable_index(batch_index);

                    let layer_q = &self.preprocessing[subtable_index];
                    let get_default = |i| Fraction {
                        p: F::zero(),
                        q: layer_q[i],
                    };

                    match layer.len() {
                        0 => (get_default(0), get_default(1)), // Neither left nor right claim
                        // is present, so they must both
                        // be 1
                        1 => {
                            if layer[0].0.is_zero() {
                                // Only left claim is present, so right claim must be 1
                                (
                                    Fraction {
                                        p: layer[0].1,
                                        q: layer_q[0],
                                    },
                                    get_default(1),
                                )
                            } else {
                                // Only right claim is present, so left claim must be 1
                                (
                                    get_default(0),
                                    Fraction {
                                        p: layer[0].1,
                                        q: layer_q[1],
                                    },
                                )
                            }
                        },
                        2 => (
                            Fraction {
                                p: layer[0].1,
                                q: layer_q[0],
                            },
                            Fraction {
                                p: layer[1].1,
                                q: layer_q[1],
                            },
                        ), // Both left and right claim are present
                        _ => panic!("Sparse layer length > 2"),
                    }
                },
                DynamicDensityRationalSumLayer::Dense(layer) => {
                    let subtable_index = Self::memory_to_subtable_index(batch_index);
                    let layer_q = &self.preprocessing[subtable_index];
                    (
                        Fraction {
                            p: layer[0],
                            q: layer_q[0],
                        },
                        Fraction {
                            p: layer[1],
                            q: layer_q[1],
                        },
                    )
                },
            })
            .unzip();

        let (left_p, left_q) = left.iter().map(|frac| (frac.p, frac.q)).unzip();
        let (right_p, right_q) = right.iter().map(|frac| (frac.p, frac.q)).unzip();
        vec![left_p, left_q, right_p, right_q]
    }

    fn compute_cubic_direct(
        &self,
        coeffs: &[F],
        evaluations: &[Vec<DenseMultilinearExtension<F>>],
        eq_table: &[F],
        lambda: &F,
    ) -> (F, F) {
        compute_cubic_direct(coeffs, evaluations, eq_table, lambda)
    }
}

pub fn sparse_preprocess<F: PrimeField>(leaves: Vec<Vec<F>>) -> Vec<Vec<Vec<F>>> {
    let num_layers = leaves[0].len().log_2();
    let mut layers: Vec<Vec<Vec<F>>> = Vec::with_capacity(num_layers);
    layers.push(leaves);

    // One more layer than usually present - so that we can always take a
    // preprocessing
    for i in 0..num_layers {
        let previous_layers = &layers[i];
        let len = previous_layers[0].len() / 2;
        // TODO(moodlezoup): parallelize over chunks instead of over batch
        let new_layers = previous_layers
            .par_iter()
            .map(|previous_layer| {
                (0..len)
                    .map(|i| previous_layer[2 * i] * previous_layer[2 * i + 1])
                    .collect::<Vec<_>>()
            })
            .collect();
        layers.push(new_layers);
    }
    layers
}

/// A batched grand product circuit.
/// Note that the circuit roots are not included in `self.layers` but included
/// in preprocessing
///        o
///      /   \
///     o     o  <- layers[layers.len() - 1]
///    / \   / \
///   o   o o   o  <- layers[layers.len() - 2]
///       ...
pub struct BatchedSparseRationalSum<F: PrimeField, const C: usize> {
    layers: Vec<BatchedSparseRationalSumLayer<F, C>>,
}

impl<F: PrimeField, const C: usize> BatchedRationalSum<F> for BatchedSparseRationalSum<F, C> {
    // (indices, p, q). p corresponds to indices (sparse), q is dense
    type Leaves = (Vec<Vec<usize>>, Vec<Vec<F>>, Vec<Vec<F>>);

    // #[tracing::instrument(skip_all, name =
    // "BatchedSparseRationalSum::construct")]
    fn construct(leaves: Self::Leaves) -> Self {
        let (indices_p, values_p, leaves_q) = leaves;
        let num_subtables = leaves_q.len();
        let leaves_len = leaves_q[0].len();
        let num_layers = leaves_q[0].len().log_2();
        let mut preprocessings = sparse_preprocess(leaves_q);

        let mut layers: Vec<BatchedSparseRationalSumLayer<F, C>> = Vec::with_capacity(num_layers);
        layers.push(BatchedSparseRationalSumLayer {
            layer_len: leaves_len,
            layers: (0..num_subtables * C)
                .into_par_iter()
                .map(|batch_index| {
                    let dimension_index = batch_index % C;
                    DynamicDensityRationalSumLayer::Sparse(
                        zip(&indices_p[dimension_index], &values_p[dimension_index])
                            .map(|(index, p)| (*index, *p))
                            .collect(),
                    )
                })
                .collect(),
            preprocessing: take(&mut preprocessings[0]),
            preprocessing_next: vec![],
        });

        for i in 0..num_layers - 1 {
            let previous_layers = &layers[i];
            let len = previous_layers.layer_len / 2;
            let new_layers = previous_layers
                .layers
                .par_iter()
                .enumerate()
                .map(|(memory_index, previous_layer)| {
                    previous_layer.layer_output::<C>(
                        len,
                        memory_index,
                        &previous_layers.preprocessing,
                    )
                })
                .collect();
            layers.push(BatchedSparseRationalSumLayer {
                layer_len: len,
                layers: new_layers,
                preprocessing: take(&mut preprocessings[i + 1]),
                preprocessing_next: vec![],
            });
        }

        Self { layers }
    }

    type CompanionCircuit = BatchedDenseRationalSum<F, 1>;

    fn d_construct(leaves: Self::Leaves) -> (Self, Option<Self::CompanionCircuit>) {
        let circuit = Self::construct(leaves);

        let claims = Net::send_to_master(&circuit.claims());
        if Net::am_master() {
            // Construct the final layers
            let claims = claims.unwrap();
            let leaves = (0..claims[0].len())
                .map(|i| claims.iter().map(|claim| (claim[i].p, claim[i].q)).unzip())
                .unzip();

            let final_circuit = Self::CompanionCircuit::construct(leaves);
            (circuit, Some(final_circuit))
        } else {
            (circuit, None)
        }
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn claims(&self) -> Vec<Fraction<F>> {
        let last_layers = &self.layers.last().unwrap();
        let final_claims = last_layers.final_claims();
        izip!(
            &final_claims[0],
            &final_claims[1],
            &final_claims[2],
            &final_claims[3]
        )
        .map(|(&left_p, &left_q, &right_p, &right_q)| {
            Fraction::rational_add(
                Fraction {
                    p: left_p,
                    q: left_q,
                },
                Fraction {
                    p: right_p,
                    q: right_q,
                },
            )
        })
        .collect()
    }

    fn layers(&'_ mut self) -> impl Iterator<Item = &'_ mut dyn BatchedRationalSumLayer<F>> {
        self.layers
            .iter_mut()
            .map(|layer| layer as &mut dyn BatchedRationalSumLayer<F>)
            .rev()
    }
}

#[cfg(test)]
mod rational_sum_tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_std::{test_rng, UniformRand};
    use rand_core::RngCore;

    #[test]
    fn dense_prove_verify() {
        const LAYER_SIZE: usize = 1 << 8;
        const BATCH_SIZE: usize = 4;
        const C: usize = 4;
        let mut rng = test_rng();
        let leaves_p: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
            std::iter::repeat_with(|| Fr::rand(&mut rng))
                .take(LAYER_SIZE)
                .collect()
        })
        .take(BATCH_SIZE * C)
        .collect();
        let leaves_q: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
            std::iter::repeat_with(|| Fr::rand(&mut rng))
                .take(LAYER_SIZE)
                .collect()
        })
        .take(BATCH_SIZE)
        .collect();

        let expected_claims: Vec<Fraction<Fr>> = leaves_p
            .iter()
            .enumerate()
            .map(|(i, layer_p)| {
                let layer_q = &leaves_q[i / C];
                zip(layer_p.iter(), layer_q.iter())
                    .map(|(p, q)| Fraction { p: *p, q: *q })
                    .reduce(Fraction::rational_add)
                    .unwrap()
            })
            .collect();

        let mut batched_circuit =
            <BatchedDenseRationalSum<Fr, C> as BatchedRationalSum<Fr>>::construct((
                leaves_p, leaves_q,
            ));
        let mut transcript = IOPTranscript::<Fr>::new(b"test_transcript");

        // I love the rust type system
        let claims =
            <BatchedDenseRationalSum<Fr, C> as BatchedRationalSum<Fr>>::claims(&batched_circuit);
        assert_eq!(expected_claims, claims);

        let (proof, r_prover) =
            <BatchedDenseRationalSum<Fr, C> as BatchedRationalSum<Fr>>::prove_rational_sum(
                &mut batched_circuit,
                &mut transcript,
            );

        let mut transcript = IOPTranscript::new(b"test_transcript");
        let (_, r_verifier) =
            BatchedDenseRationalSum::<Fr, C>::verify_rational_sum(&proof, &claims, &mut transcript);
        assert_eq!(r_prover, r_verifier);
    }

    #[test]
    fn sparse_prove_verify() {
        const LAYER_SIZE: usize = 1 << 8;
        const BATCH_SIZE: usize = 4;
        const C: usize = 4;
        let mut rng = test_rng();

        let baseline: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
            std::iter::repeat_with(|| Fr::rand(&mut rng))
                .take(LAYER_SIZE)
                .collect()
        })
        .take(BATCH_SIZE)
        .collect();

        let mut indices_p = vec![vec![]; C];
        let mut values_p = vec![vec![]; C];

        let leaves_p: Vec<Vec<Fr>> = (0..C)
            .map(|layer_idx| {
                (0..LAYER_SIZE)
                    .map(|index| {
                        if rng.next_u32() % 4 == 0 {
                            let mut p = Fr::rand(&mut rng);
                            while p == Fr::zero() {
                                p = Fr::rand(&mut rng);
                            }
                            indices_p[layer_idx].push(index);
                            values_p[layer_idx].push(p);
                            p
                        } else {
                            Fr::zero()
                        }
                    })
                    .collect()
            })
            .collect();

        let expected_claims: Vec<Fraction<Fr>> = (0..BATCH_SIZE * C)
            .map(|i| {
                let layer_p = &leaves_p[i % C];
                let layer_q = &baseline[i / C];
                zip(layer_p.iter(), layer_q.iter())
                    .map(|(p, q)| Fraction { p: *p, q: *q })
                    .reduce(Fraction::rational_add)
                    .unwrap()
            })
            .collect();

        let mut batched_circuit =
            <BatchedSparseRationalSum<Fr, C> as BatchedRationalSum<Fr>>::construct((
                indices_p,
                values_p,
                baseline.clone(),
            ));
        let mut transcript = IOPTranscript::new(b"test_transcript");

        // I love the rust type system
        let claims =
            <BatchedSparseRationalSum<Fr, C> as BatchedRationalSum<Fr>>::claims(&batched_circuit);
        assert_eq!(expected_claims, claims);

        let (proof, r_prover) =
            <BatchedSparseRationalSum<Fr, C> as BatchedRationalSum<Fr>>::prove_rational_sum(
                &mut batched_circuit,
                &mut transcript,
            );

        let mut transcript = IOPTranscript::<Fr>::new(b"test_transcript");
        let (_, r_verifier) = BatchedSparseRationalSum::<Fr, C>::verify_rational_sum(
            &proof,
            &claims,
            &mut transcript,
        );
        assert_eq!(r_prover, r_verifier);
    }
}

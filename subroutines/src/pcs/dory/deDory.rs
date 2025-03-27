//! Main module for distributed Dory commitment scheme

use crate::{
    pcs::{
        dory::batching::{batch_verify_internal, d_multi_open_internal},
        PCSError, PolynomialCommitmentScheme,
    },
    BatchProof,
};
use arithmetic::math::Math;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{borrow::Borrow, marker::PhantomData, rand::Rng, sync::Arc, vec::Vec, Zero};
use transcript::IOPTranscript;

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

use deDory::{
    deSetup::{PublicParameters, SubProverSetup, VerifierSetup},
    eval::{de_generate_eval_proof, verify_de_eval_proof, DoryEvalProof},
    DeDoryCommitment,
};

/// Dory Polynomial Commitment Scheme on multilinear polynomials.
pub struct DeDory<E: Pairing> {
    #[doc(hidden)]
    phantom: PhantomData<E>,
}

#[derive(Default, Clone, CanonicalSerialize, CanonicalDeserialize, PartialEq, Eq, Debug)]
pub struct DeDoryProof<E: Pairing> {
    proof: DoryEvalProof<E>,
    m: usize,
}

#[derive(Debug, Clone)]
pub enum DeDorySRS<E: Pairing> {
    Unprocessed(PublicParameters<E>),
    Processed((SubProverSetup<E>, VerifierSetup<E>)),
}

impl<E: Pairing> PolynomialCommitmentScheme<E> for DeDory<E> {
    // Parameters
    type ProverParam = SubProverSetup<E>;
    type VerifierParam = VerifierSetup<E>;
    type SRS = DeDorySRS<E>;
    // Polynomial and its associated types
    type Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>;
    type ProverCommitmentAdvice = Vec<E::G1Affine>;
    type Point = Vec<E::ScalarField>;
    type Evaluation = E::ScalarField;
    // Commitments and proofs
    type Commitment = PairingOutput<E>;
    type Proof = Option<DeDoryProof<E>>;
    type BatchProof = BatchProof<E, Self>;

    /// Build SRS for testing.
    ///
    /// - For univariate polynomials, `log_size` is the log of maximum degree.
    /// - For multilinear polynomials, `log_size` is the number of variables.
    ///
    /// WARNING: THIS FUNCTION IS FOR TESTING PURPOSE ONLY.
    /// THE OUTPUT SRS SHOULD NOT BE USED IN PRODUCTION.
    fn gen_srs_for_testing<R: Rng>(rng: &mut R, log_size: usize) -> Result<Self::SRS, PCSError> {
        Ok(DeDorySRS::Unprocessed(PublicParameters::rand(
            log_size, rng,
        )))
    }

    /// Trim the universal parameters to specialize the public parameters.
    /// Input both `supported_log_degree` for univariate and
    /// `supported_num_vars` for multilinear.
    fn trim(
        srs: impl Borrow<Self::SRS>,
        _supported_degree: Option<usize>,
        _supported_num_vars: Option<usize>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        Ok(match srs.borrow() {
            DeDorySRS::Unprocessed(pp) => (SubProverSetup::new(pp), VerifierSetup::new(pp)),
            DeDorySRS::Processed((prover, verifier)) => (prover.clone(), verifier.clone()),
        })
    }

    /// Generate a commitment for a polynomial.
    ///
    /// This function takes `2^num_vars` number of scalar multiplications over
    /// G1.
    fn d_commit(
        prover_param: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
    ) -> Result<(Option<Self::Commitment>, Self::ProverCommitmentAdvice), PCSError> {
        let m = Net::n_parties().log_2();
        let mut n = poly.num_vars + m;
        if (n - m) % 2 == 1 {
            n += 1;
        }
        let sub_prover_id = Net::party_id();

        Ok(DeDoryCommitment::deCommit(
            sub_prover_id,
            &poly.evaluations,
            Net::n_parties().log_2(),
            n,
            prover_param.borrow(),
        ))
    }

    fn batch_d_commit(
        prover_param: impl Borrow<Self::ProverParam>,
        polys: &[Self::Polynomial],
    ) -> Result<
        (
            Vec<Option<Self::Commitment>>,
            Vec<Self::ProverCommitmentAdvice>,
        ),
        PCSError,
    > {
        let m = Net::n_parties().log_2();
        let mut n = polys[0].num_vars + m;
        if (n - m) % 2 == 1 {
            n += 1;
        }
        let sub_prover_id = Net::party_id();

        Ok(DeDoryCommitment::multi_deCommit(
            sub_prover_id,
            polys.iter().map(|poly| &poly.evaluations[..]).collect(),
            m,
            n,
            prover_param.borrow(),
        ))
    }

    /// On input a polynomial `p` and a point `point`, outputs a proof for the
    /// same. This function does not need to take the evaluation value as an
    /// input.
    ///
    /// This function takes 2^{num_var +1} number of scalar multiplications over
    /// G1:
    /// - it prodceeds with `num_var` number of rounds,
    /// - at round i, we compute an MSM for `2^{num_var - i + 1}` number of G2
    ///   elements.
    fn open(
        prover_param: impl Borrow<Self::ProverParam>,
        polynomial: &Self::Polynomial,
        advice: &Self::ProverCommitmentAdvice,
        point: &Self::Point,
    ) -> Result<Self::Proof, PCSError> {
        let mut sub_prover_transcript = IOPTranscript::new(b"Distributed Dory Evaluation Proof");
        let m = Net::n_parties().log_2();
        let mut n = polynomial.num_vars + m;
        let point = if (n - m) % 2 == 1 {
            n += 1;
            vec![
                &point[..(n - 1 - m)],
                &[E::ScalarField::zero()],
                &point[(n - 1 - m)..],
            ]
            .concat()
        } else {
            point.to_vec()
        };
        let proof = de_generate_eval_proof(
            Net::party_id(),
            &mut sub_prover_transcript,
            &polynomial.evaluations,
            &advice,
            &point,
            n,
            m,
            prover_param.borrow(),
        );
        if Net::am_master() {
            Ok(Some(DeDoryProof {
                proof: proof.unwrap(),
                m,
            }))
        } else {
            Ok(None)
        }
    }

    /// Input a list of multilinear extensions, and a same number of points, and
    /// a transcript, compute a multi-opening for all the polynomials.
    fn d_multi_open(
        prover_param: impl Borrow<Self::ProverParam>,
        polynomials: Vec<Self::Polynomial>,
        advices: &[Self::ProverCommitmentAdvice],
        points: &[Self::Point],
        evals: &[Self::Evaluation],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<Option<BatchProof<E, Self>>, PCSError> {
        d_multi_open_internal(
            prover_param.borrow(),
            polynomials,
            advices,
            points,
            evals,
            transcript,
        )
    }

    /// Verifies that `value` is the evaluation at `x` of the polynomial
    /// committed inside `comm`.
    ///
    /// This function takes
    /// - num_var number of pairing product.
    /// - num_var number of MSM
    fn verify(
        verifier_param: &Self::VerifierParam,
        commitment: &Self::Commitment,
        point: &Self::Point,
        value: &E::ScalarField,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        let mut verifier_transcript = IOPTranscript::new(b"Distributed Dory Evaluation Proof");
        let mut n = point.len();
        let mut proof = proof.as_ref().unwrap().clone();
        let point = if (n - proof.m) % 2 == 1 {
            n += 1;
            vec![
                &point[..(n - 1 - proof.m)],
                &[E::ScalarField::zero()],
                &point[(n - 1 - proof.m)..],
            ]
            .concat()
        } else {
            point.to_vec()
        };
        if let Err(_) = verify_de_eval_proof(
            &mut verifier_transcript,
            &mut proof.proof,
            &commitment,
            *value,
            &point,
            n,
            proof.m,
            verifier_param,
        ) {
            Ok(false)
        } else {
            Ok(true)
        }
    }

    /// Verifies that `value_i` is the evaluation at `x_i` of the polynomial
    /// `poly_i` committed inside `comm`.
    fn batch_verify(
        verifier_param: &Self::VerifierParam,
        commitments: &[Self::Commitment],
        points: &[Self::Point],
        batch_proof: &Self::BatchProof,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError> {
        batch_verify_internal(verifier_param, commitments, points, batch_proof, transcript)
    }
}

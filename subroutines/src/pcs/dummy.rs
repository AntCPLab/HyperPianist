// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

use crate::pcs::{PCSError, PolynomialCommitmentScheme};
use ark_ec::pairing::Pairing;
use ark_poly::DenseMultilinearExtension;
use ark_std::{borrow::Borrow, marker::PhantomData, rand::Rng, vec::Vec, sync::Arc};
use transcript::IOPTranscript;

/// KZG Polynomial Commitment Scheme on multilinear polynomials.
pub struct DummyPCS<E: Pairing> {
    #[doc(hidden)]
    phantom: PhantomData<E>,
}

impl<E: Pairing> PolynomialCommitmentScheme<E> for DummyPCS<E> {
    // Parameters
    type ProverParam = ();
    type VerifierParam = ();
    type SRS = ();
    // Polynomial and its associated types
    type Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>;
    type ProverCommitmentAdvice = ();
    type Point = Vec<E::ScalarField>;
    type Evaluation = E::ScalarField;
    // Commitments and proofs
    type Commitment = ();
    type Proof = ();
    type BatchProof = ();

    /// Build SRS for testing.
    ///
    /// - For univariate polynomials, `log_size` is the log of maximum degree.
    /// - For multilinear polynomials, `log_size` is the number of variables.
    ///
    /// WARNING: THIS FUNCTION IS FOR TESTING PURPOSE ONLY.
    /// THE OUTPUT SRS SHOULD NOT BE USED IN PRODUCTION.
    fn gen_srs_for_testing<R: Rng>(_rng: &mut R, _log_size: usize) -> Result<Self::SRS, PCSError> {
        Ok(())
    }

    /// Trim the universal parameters to specialize the public parameters.
    /// Input both `supported_log_degree` for univariate and
    /// `supported_num_vars` for multilinear.
    fn trim(
        _srs: impl Borrow<Self::SRS>,
        _supported_degree: Option<usize>,
        _supported_num_vars: Option<usize>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        Ok(((), ()))
    }

    /// Generate a commitment for a polynomial.
    ///
    /// This function takes `2^num_vars` number of scalar multiplications over
    /// G1.
    fn commit(
        _prover_param: impl Borrow<Self::ProverParam>,
        _poly: &Self::Polynomial,
    ) -> Result<(Self::Commitment, Self::ProverCommitmentAdvice), PCSError> {
        Ok(((), ()))
    }

    fn d_commit(
        _prover_param: impl Borrow<Self::ProverParam>,
        _poly: &Self::Polynomial,
    ) -> Result<(Option<Self::Commitment>, Self::ProverCommitmentAdvice), PCSError> {
        Ok((Some(()), ()))
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
        _prover_param: impl Borrow<Self::ProverParam>,
        _polynomial: &Self::Polynomial,
        _advice: &Self::ProverCommitmentAdvice,
        _point: &Self::Point,
    ) -> Result<Self::Proof, PCSError> {
        Ok(())
    }

    /// Input a list of multilinear extensions, and a same number of points, and
    /// a transcript, compute a multi-opening for all the polynomials.
    fn multi_open(
        _prover_param: impl Borrow<Self::ProverParam>,
        _polynomials: Vec<Self::Polynomial>,
        _advices: &[Self::ProverCommitmentAdvice],
        _points: &[Self::Point],
        _evals: &[Self::Evaluation],
        _transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<Self::BatchProof, PCSError> {
        Ok(())
    }

    fn d_multi_open(
        _prover_param: impl Borrow<Self::ProverParam>,
        _polynomials: Vec<Self::Polynomial>,
        _advices: &[Self::ProverCommitmentAdvice],
        _points: &[Self::Point],
        _evals: &[Self::Evaluation],
        _transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<Option<Self::BatchProof>, PCSError> {
        Ok(Some(()))
    }

    /// Verifies that `value` is the evaluation at `x` of the polynomial
    /// committed inside `comm`.
    ///
    /// This function takes
    /// - num_var number of pairing product.
    /// - num_var number of MSM
    fn verify(
        _verifier_param: &Self::VerifierParam,
        _commitment: &Self::Commitment,
        _point: &Self::Point,
        _value: &E::ScalarField,
        _proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        Ok(true)
    }

    /// Verifies that `value_i` is the evaluation at `x_i` of the polynomial
    /// `poly_i` committed inside `comm`.
    fn batch_verify(
        _verifier_param: &Self::VerifierParam,
        _commitments: &[Self::Commitment],
        _points: &[Self::Point],
        _batch_proof: &Self::BatchProof,
        _transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError> {
        Ok(true)
    }
}

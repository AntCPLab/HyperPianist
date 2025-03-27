//! Main module for the (non-distributed) Dory commitment scheme

use crate::{
    pcs::{PCSError, PolynomialCommitmentScheme},
    BatchProof,
};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_poly::DenseMultilinearExtension;
use ark_std::{borrow::Borrow, marker::PhantomData, rand::Rng, sync::Arc, vec::Vec, Zero};
use transcript::IOPTranscript;

use deDory::{
    eval::{generate_eval_proof, verify_eval_proof, DoryEvalProof},
    setup::{ProverSetup, PublicParameters, VerifierSetup},
    DoryCommitment,
};

/// Dory Polynomial Commitment Scheme on multilinear polynomials.
pub struct Dory<E: Pairing> {
    #[doc(hidden)]
    phantom: PhantomData<E>,
}

impl<E: Pairing> PolynomialCommitmentScheme<E> for Dory<E> {
    // Parameters
    type ProverParam = ProverSetup<E>;
    type VerifierParam = VerifierSetup<E>;
    type SRS = PublicParameters<E>;
    // Polynomial and its associated types
    type Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>;
    type ProverCommitmentAdvice = Vec<E::G1Affine>;
    type Point = Vec<E::ScalarField>;
    type Evaluation = E::ScalarField;
    // Commitments and proofs
    type Commitment = PairingOutput<E>;
    type Proof = DoryEvalProof<E>;
    type BatchProof = BatchProof<E, Self>;

    /// Build SRS for testing.
    ///
    /// - For univariate polynomials, `log_size` is the log of maximum degree.
    /// - For multilinear polynomials, `log_size` is the number of variables.
    ///
    /// WARNING: THIS FUNCTION IS FOR TESTING PURPOSE ONLY.
    /// THE OUTPUT SRS SHOULD NOT BE USED IN PRODUCTION.
    fn gen_srs_for_testing<R: Rng>(rng: &mut R, log_size: usize) -> Result<Self::SRS, PCSError> {
        Ok(PublicParameters::rand(log_size, rng))
    }

    /// Trim the universal parameters to specialize the public parameters.
    /// Input both `supported_log_degree` for univariate and
    /// `supported_num_vars` for multilinear.
    fn trim(
        srs: impl Borrow<Self::SRS>,
        supported_degree: Option<usize>,
        supported_num_vars: Option<usize>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        assert!(supported_degree.is_none());
        assert!(!supported_num_vars.is_none());

        Ok((
            ProverSetup::from(srs.borrow()),
            VerifierSetup::from(srs.borrow()),
        ))
    }

    /// Generate a commitment for a polynomial.
    ///
    /// This function takes `2^num_vars` number of scalar multiplications over
    /// G1.
    fn commit(
        prover_param: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
    ) -> Result<(Self::Commitment, Self::ProverCommitmentAdvice), PCSError> {
        // Construct
        let mut n = poly.num_vars;
        if n % 2 == 1 {
            n += 1;
        }
        let comm = DoryCommitment::commit(&poly.evaluations, prover_param.borrow(), n);
        Ok((comm.comm, comm.T_vec_prime))
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
        let mut point = point.clone();
        if point.len() % 2 == 1 {
            point.push(E::ScalarField::zero());
        }
        let mut sub_prover_transcript = IOPTranscript::new(b"Dory Evaluation Proof");
        Ok(generate_eval_proof(
            &mut sub_prover_transcript,
            &polynomial.evaluations,
            &advice,
            &point,
            prover_param.borrow(),
        ))
    }

    // /// Input a list of multilinear extensions, and a same number of points, and
    // /// a transcript, compute a multi-opening for all the polynomials.
    // fn multi_open(
    //     prover_param: impl Borrow<Self::ProverParam>,
    //     polynomials: &[Self::Polynomial],
    //     points: &[Self::Point],
    //     evals: &[Self::Evaluation],
    //     transcript: &mut IOPTranscript<E::ScalarField>,
    // ) -> Result<BatchProof<E, Self>, PCSError> {
    //     multi_open_internal(
    //         prover_param.borrow(),
    //         polynomials,
    //         points,
    //         evals,
    //         transcript,
    //     )
    // }

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
        let mut point = point.clone();
        if point.len() % 2 == 1 {
            point.push(E::ScalarField::zero());
        }
        let mut verifier_transcript = IOPTranscript::new(b"Dory Evaluation Proof");
        if let Err(_) = verify_eval_proof(
            &mut verifier_transcript,
            &mut proof.clone(),
            commitment,
            *value,
            &point,
            verifier_param,
        ) {
            Ok(false)
        } else {
            Ok(true)
        }
    }

    // /// Verifies that `value_i` is the evaluation at `x_i` of the polynomial
    // /// `poly_i` committed inside `comm`.
    // fn batch_verify(
    //     verifier_param: &Self::VerifierParam,
    //     commitments: &[Self::Commitment],
    //     points: &[Self::Point],
    //     batch_proof: &Self::BatchProof,
    //     transcript: &mut IOPTranscript<E::ScalarField>,
    // ) -> Result<bool, PCSError> {
    //     batch_verify_internal(verifier_param, commitments, points, batch_proof,
    // transcript) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arithmetic::evaluate_opt;
    use ark_bls12_381::Bls12_381;
    use ark_ec::pairing::Pairing;
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::{test_rng, vec::Vec, UniformRand};

    type E = Bls12_381;
    type Fr = <E as Pairing>::ScalarField;

    fn test_single_helper<R: Rng>(
        params: &PublicParameters<E>,
        poly: &Arc<DenseMultilinearExtension<Fr>>,
        rng: &mut R,
    ) -> Result<(), PCSError> {
        let nv = poly.num_vars();
        assert_ne!(nv, 0);
        let (ck, vk) = Dory::trim(params, None, Some(nv))?;
        let point: Vec<_> = (0..nv).map(|_| Fr::rand(rng)).collect();
        let (com, advice) = Dory::commit(&ck, poly)?;
        let proof = Dory::open(&ck, poly, &advice, &point)?;
        let value = evaluate_opt(poly, &point);

        assert!(Dory::verify(&vk, &com, &point, &value, &proof)?);

        let value = Fr::rand(rng);
        assert!(!Dory::verify(&vk, &com, &point, &value, &proof)?);

        Ok(())
    }

    #[test]
    fn test_single_commit() -> Result<(), PCSError> {
        let mut rng = test_rng();

        let params = Dory::<E>::gen_srs_for_testing(&mut rng, 8)?;

        // normal polynomials
        for nv in 5..=8 {
            let poly1 = Arc::new(DenseMultilinearExtension::rand(nv, &mut rng));
            test_single_helper(&params, &poly1, &mut rng)?;
        }

        Ok(())
    }
}

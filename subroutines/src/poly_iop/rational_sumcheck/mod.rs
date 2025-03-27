//! This module implements the rational sum check protocol with or without
//! layered circuits

use crate::{
    poly_iop::{errors::PolyIOPErrors, PolyIOP},
    IOPProof, SumCheck,
};
use arithmetic::{bit_decompose, eq_eval, math::Math, VirtualPolynomial};
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_std::{end_timer, start_timer, One, Zero};
use itertools::izip;
use std::iter::zip;
use transcript::IOPTranscript;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

use super::sum_check::SumCheckSubClaim;

pub mod layered_circuit;

/// Non-layered-circuit version of batched RationalSumcheck.
/// Proves that \sum p(x) / q(x) = v.
pub trait RationalSumcheckSlow<F>: SumCheck<F>
where
    F: PrimeField,
{
    type RationalSumcheckSubClaim;
    type RationalSumcheckProof: CanonicalSerialize + CanonicalDeserialize;

    /// Initialize the system with a transcript
    ///
    /// This function is optional -- in the case where a RationalSumcheck is
    /// an building block for a more complex protocol, the transcript
    /// may be initialized by this complex protocol, and passed to the
    /// RationalSumcheck prover/verifier.
    fn init_transcript() -> Self::Transcript;

    /// Returns (proof, inv_g) for testing
    /// 
    /// IMPORTANT: if claimed_sums is set to equal to the length of fx,
    /// it is assumed that all instances of fx, gx, and g_inv are independent.
    /// If claimed_sums is a single element, it is assumed that all fx, gx
    /// and g_inv form one instance (sum)
    #[allow(clippy::type_complexity)]
    fn prove(
        fx: Vec<Self::VirtualPolynomial>,
        gx: Vec<Self::MultilinearExtension>,
        g_inv: Vec<Self::MultilinearExtension>,
        claimed_sums: Vec<F>,
        transcript: &mut IOPTranscript<F>,
    ) -> Result<Self::RationalSumcheckProof, PolyIOPErrors>;

    fn d_prove(
        fx: Vec<Self::VirtualPolynomial>,
        gx: Vec<Self::MultilinearExtension>,
        g_inv: Vec<Self::MultilinearExtension>,
        claimed_sums: Vec<F>,
        transcript: &mut IOPTranscript<F>,
    ) -> Result<Option<Self::RationalSumcheckProof>, PolyIOPErrors>;

    /// Verify that for witness multilinear polynomials (f1, ..., fk, g1, ...,
    /// gk) it holds that
    ///      `\prod_{x \in {0,1}^n} f1(x) * ... * fk(x)
    ///     = \prod_{x \in {0,1}^n} g1(x) * ... * gk(x)`
    fn verify(
        proof: &Self::RationalSumcheckProof,
        aux_info: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::RationalSumcheckSubClaim, PolyIOPErrors>;
}

#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct RationalSumcheckProof<F: PrimeField> {
    pub sum_check_proof: IOPProof<F>,
    pub num_polys: usize,
    pub claimed_sums: Vec<F>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct RationalSumcheckSubClaim<F: PrimeField> {
    pub sum_check_sub_claim: SumCheckSubClaim<F>,
    pub coeffs: Vec<F>,
    pub zerocheck_r: Vec<F>,
}

impl<F> RationalSumcheckSlow<F> for PolyIOP<F>
where
    F: PrimeField,
{
    type RationalSumcheckSubClaim = RationalSumcheckSubClaim<F>;
    type RationalSumcheckProof = RationalSumcheckProof<F>;

    fn init_transcript() -> Self::Transcript {
        IOPTranscript::<F>::new(b"Initializing RationalSumcheck transcript")
    }

    // Proves the Rational Sumcheck relation as a batched statement of multiple
    // independent instances f, g, g_inv
    fn prove(
        fx: Vec<Self::VirtualPolynomial>,
        gx: Vec<Self::MultilinearExtension>,
        g_inv: Vec<Self::MultilinearExtension>,
        claimed_sums: Vec<F>,
        transcript: &mut IOPTranscript<F>,
    ) -> Result<Self::RationalSumcheckProof, PolyIOPErrors> {
        let start = start_timer!(|| "rational_sumcheck prove");

        if fx.len() != gx.len() || gx.len() != g_inv.len()  {
            return Err(PolyIOPErrors::InvalidParameters(format!(
                "polynomials lengthes are not equal"
            )));
        }

        if claimed_sums.len() != g_inv.len() && claimed_sums.len() != 1 {
            return Err(PolyIOPErrors::InvalidParameters(format!(
                "claimed sums have invalid length"
            )));
        }

        let should_mix_sums = claimed_sums.len() != 1;

        transcript.append_serializable_element(b"rational_sumcheck_claims", &claimed_sums)?;

        let r = transcript.get_and_append_challenge_vectors(b"0check r", gx[0].num_vars)?;

        let coeffs = transcript
            .get_and_append_challenge_vectors(b"rational_sumcheck_coeffs", fx.len() * 2)?;

        // Zerocheck
        let mut sum_poly = VirtualPolynomial::new(gx[0].num_vars);
        let mut coeff_sum = F::zero();
        for (g_poly, g_inv_poly, coeff) in izip!(gx, g_inv.iter(), coeffs.iter()) {
            sum_poly.add_mle_list([g_poly, g_inv_poly.clone()], *coeff)?;
            coeff_sum += *coeff;
        }
        sum_poly.add_mle_list([], -coeff_sum)?;

        sum_poly = sum_poly.build_f_hat(&r)?;

        // Sumcheck
        let num_polys = fx.len();
        for (f_poly, g_inv_poly, coeff) in izip!(fx, g_inv, coeffs[num_polys..].iter())
        {
            let mut item = f_poly;
            if should_mix_sums {
                item.mul_by_mle(g_inv_poly, *coeff)?;
            } else {
                item.mul_by_mle(g_inv_poly, F::one())?;
            }
            sum_poly += &item;
        }

        let sum_check_proof =
            <PolyIOP<F> as SumCheck<F>>::prove(sum_poly, transcript)?;

        end_timer!(start);

        Ok(RationalSumcheckProof {
            sum_check_proof,
            num_polys,
            claimed_sums,
        })
    }

    fn d_prove(
        fx: Vec<Self::VirtualPolynomial>,
        gx: Vec<Self::MultilinearExtension>,
        g_inv: Vec<Self::MultilinearExtension>,
        claimed_sums: Vec<F>,
        transcript: &mut IOPTranscript<F>,
    ) -> Result<Option<Self::RationalSumcheckProof>, PolyIOPErrors> {
        let start = start_timer!(|| "rational_sumcheck prove");

        if fx.len() != gx.len() || gx.len() != g_inv.len() {
            return Err(PolyIOPErrors::InvalidParameters(format!(
                "polynomials lengthes are not equal"
            )));
        }

        if claimed_sums.len() != g_inv.len() && claimed_sums.len() != 1 {
            return Err(PolyIOPErrors::InvalidParameters(format!(
                "claimed sums have invalid length"
            )));
        }

        let should_mix_sums = claimed_sums.len() != 1;

        if Net::am_master() {
            transcript.append_serializable_element(b"rational_sumcheck_claims", &claimed_sums)?;
        }

        let length = fx[0].aux_info.num_variables;
        let num_party_vars = Net::n_parties().log_2();
        let r = if Net::am_master() {
            let r = transcript
                .get_and_append_challenge_vectors(b"0check r", length + num_party_vars)?;
            Net::recv_from_master_uniform(Some(r))
        } else {
            Net::recv_from_master_uniform(None)
        };

        let index_vec: Vec<F> = bit_decompose(Net::party_id() as u64, num_party_vars)
            .into_iter()
            .map(|x| F::from(x))
            .collect();

        let coeffs = if Net::am_master() {
            let coeffs = transcript
                .get_and_append_challenge_vectors(b"rational_sumcheck_coeffs", fx.len() * 2)?;
            Net::recv_from_master_uniform(Some(coeffs))
        } else {
            Net::recv_from_master_uniform(None)
        };

        // Zerocheck
        let mut sum_poly = VirtualPolynomial::new(gx[0].num_vars);
        let mut coeff_sum = F::zero();
        for (g_poly, g_inv_poly, coeff) in izip!(gx, g_inv.iter(), coeffs.iter()) {
            sum_poly.add_mle_list([g_poly, g_inv_poly.clone()], *coeff)?;
            coeff_sum += *coeff;
        }
        sum_poly.add_mle_list([], -coeff_sum)?;

        let coeff = eq_eval(&r[length..], &index_vec)?;
        sum_poly = sum_poly.build_f_hat_with_coeff(&r[..length], &coeff)?;

        // Sumcheck
        let num_polys = fx.len();
        for (f_poly, g_inv_poly, coeff) in izip!(fx, g_inv, coeffs[num_polys..].iter())
        {
            let mut item = f_poly;
            if should_mix_sums {
                item.mul_by_mle(g_inv_poly, *coeff)?;
            } else {
                item.mul_by_mle(g_inv_poly, F::one())?;
            }
            sum_poly += &item;
        }

        let sum_check_proof =
            <PolyIOP<F> as SumCheck<F>>::d_prove(sum_poly, transcript)?;

        end_timer!(start);

        if Net::am_master() {
            Ok(Some(RationalSumcheckProof {
                sum_check_proof: sum_check_proof.unwrap(),
                num_polys,
                claimed_sums,
            }))
        } else {
            Ok(None)
        }
    }

    fn verify(
        proof: &Self::RationalSumcheckProof,
        aux_info: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::RationalSumcheckSubClaim, PolyIOPErrors> {
        let start = start_timer!(|| "rational_sumcheck verify");

        transcript.append_serializable_element(b"rational_sumcheck_claims", &proof.claimed_sums)?;

        let zerocheck_r =
            transcript.get_and_append_challenge_vectors(b"0check r", aux_info.num_variables)?;

        let coeffs = transcript.get_and_append_challenge_vectors(
            b"rational_sumcheck_coeffs",
            proof.num_polys * 2,
        )?;

        let claimed_sum = 
            if proof.claimed_sums.len() == 1 {
                proof.claimed_sums[0]
            } else {
                zip(
                    proof.claimed_sums.iter(),
                    coeffs[proof.num_polys..].iter(),
                )
                .map(|(sum, coeff)| *sum * coeff)
                .sum::<F>()
            };

        let sum_check_sub_claim = <Self as SumCheck<F>>::verify(
            claimed_sum,
            &proof.sum_check_proof,
            aux_info,
            transcript,
        )?;

        end_timer!(start);

        Ok(RationalSumcheckSubClaim {
            sum_check_sub_claim,
            coeffs,
            zerocheck_r,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use arithmetic::{eq_eval, VPAuxInfo};
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_ff::{batch_inversion, One};
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::{test_rng, UniformRand, Zero};
    use itertools::MultiUnzip;
    use rand::RngCore;
    use std::{iter::zip, marker::PhantomData, sync::Arc};

    fn create_polys<R: RngCore>(
        num_vars: usize,
        rng: &mut R,
    ) -> (
        Arc<DenseMultilinearExtension<Fr>>,
        Arc<DenseMultilinearExtension<Fr>>,
        Arc<DenseMultilinearExtension<Fr>>,
    ) {
        let evals_p = std::iter::repeat_with(|| Fr::rand(rng))
            .take(1 << num_vars)
            .collect();
        let p = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars, evals_p,
        ));

        let evals_q = std::iter::repeat_with(|| {
            let mut val = Fr::zero();
            while val == Fr::zero() {
                val = Fr::rand(rng);
            }
            val
        })
        .take(1 << num_vars)
        .collect();

        let q = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars, evals_q,
        ));

        let mut g_inv = Arc::new(DenseMultilinearExtension::clone(&q));
        batch_inversion(&mut Arc::get_mut(&mut g_inv).unwrap().evaluations);

        (p, q, g_inv)
    }

    #[test]
    fn test_rational_sumcheck() -> Result<(), PolyIOPErrors> {
        let num_vars = 5;

        let mut rng = test_rng();

        let (p_polys, q_polys, q_inv_polys, expected_sums) =
            MultiUnzip::<(Vec<_>, Vec<_>, Vec<_>, Vec<_>)>::multiunzip(
                std::iter::repeat_with(|| {
                    let (p, q, q_inv) = create_polys(num_vars, &mut rng);
                    let expected_sum = zip(p.evaluations.iter(), q.evaluations.iter())
                        .map(|(p, q)| p / q)
                        .sum::<Fr>();
                    (p, q, q_inv, expected_sum)
                })
                .take(10),
            );

        let p_virt_polys = p_polys
            .iter()
            .map(|p| VirtualPolynomial::new_from_mle(&p, Fr::one()))
            .collect::<Vec<_>>();

        let mut transcript = <PolyIOP<Fr> as RationalSumcheckSlow<Fr>>::init_transcript();
        let proof = <PolyIOP<Fr> as RationalSumcheckSlow<Fr>>::prove(
            p_virt_polys,
            q_polys.iter().map(|poly| Arc::new(DenseMultilinearExtension::clone(poly))).collect(),
            q_inv_polys.iter().map(|poly| Arc::new(DenseMultilinearExtension::clone(poly))).collect(),
            expected_sums,
            &mut transcript,
        )?;

        // Apparently no one knows what's this for?
        let aux_info = VPAuxInfo {
            max_degree: 3,
            num_variables: num_vars,
            phantom: PhantomData::default(),
        };
        let mut transcript = <PolyIOP<Fr> as RationalSumcheckSlow<Fr>>::init_transcript();
        let subclaim = <PolyIOP<Fr> as RationalSumcheckSlow<Fr>>::verify(
            &proof,
            &aux_info,
            &mut transcript,
        )?;

        // Zerocheck subclaim
        let mut sum = Fr::zero();
        for (p, q, q_inv, coeff1, coeff2) in izip!(
            p_polys.iter(),
            q_polys.iter(),
            q_inv_polys.iter(),
            subclaim.coeffs.iter(),
            subclaim.coeffs[10..].iter()
        ) {
            sum += *coeff1
                * (q.evaluate(&subclaim.sum_check_sub_claim.point).unwrap()
                    * q_inv.evaluate(&subclaim.sum_check_sub_claim.point).unwrap()
                    - Fr::one())
                * eq_eval(&subclaim.sum_check_sub_claim.point, &subclaim.zerocheck_r)?
                + *coeff2
                    * (p.evaluate(&subclaim.sum_check_sub_claim.point).unwrap()
                        * q_inv.evaluate(&subclaim.sum_check_sub_claim.point).unwrap());
        }
        assert_eq!(sum, subclaim.sum_check_sub_claim.expected_evaluation);

        Ok(())
    }
}

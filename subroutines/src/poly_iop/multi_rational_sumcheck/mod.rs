use crate::{
    poly_iop::{errors::PolyIOPErrors, PolyIOP},
    SumcheckInstanceProof,
};
use arithmetic::{bit_decompose, build_eq_x_r, build_eq_x_r_with_coeff, eq_eval, math::Math, products_except_self};
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{end_timer, start_timer};
use std::{iter::zip, sync::Arc};
use transcript::IOPTranscript;

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

pub trait MultiRationalSumcheck<F>
where
    F: PrimeField,
{
    type MultilinearExtension;
    type MultiRationalSumcheckSubClaim;
    type Transcript;
    type MultiRationalSumcheckProof: CanonicalSerialize + CanonicalDeserialize;

    /// Initialize the system with a transcript
    ///
    /// This function is optional -- in the case where a MultiRationalSumcheck
    /// is an building block for a more complex protocol, the transcript
    /// may be initialized by this complex protocol, and passed to the
    /// MultiRationalSumcheck prover/verifier.
    fn init_transcript() -> Self::Transcript;

    /// h(x) = Sum_{i = 1}^n f_i(x) / g_i(x)
    #[allow(clippy::type_complexity)]
    fn prove(
        fx: &[F],
        gx: Vec<Self::MultilinearExtension>,
        h: Self::MultilinearExtension,
        claimed_sum: F,
        transcript: &mut IOPTranscript<F>,
    ) -> Result<(Self::MultiRationalSumcheckProof, Vec<F>), PolyIOPErrors>;

    fn d_prove(
        fx: &[F],
        gx: Vec<Self::MultilinearExtension>,
        h: Self::MultilinearExtension,
        claimed_sum: F,
        transcript: &mut IOPTranscript<F>,
    ) -> Result<Option<(Self::MultiRationalSumcheckProof, Vec<F>)>, PolyIOPErrors>;

    fn verify(
        proof: &Self::MultiRationalSumcheckProof,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::MultiRationalSumcheckSubClaim, PolyIOPErrors>;
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct MultiRationalSumcheckProof<F: PrimeField> {
    pub sumcheck_proof: SumcheckInstanceProof<F>,
    pub num_rounds: usize,
    pub num_polys: usize,
    pub claimed_sum: F,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct MultiRationalSumcheckSubClaim<F: PrimeField> {
    pub sumcheck_point: Vec<F>,
    pub sumcheck_expected_evaluation: F,
    pub zerocheck_r: Vec<F>,
    pub coeff: F,
}

// The first element in values is the eq_eval, second element is h
fn multi_rational_sumcheck_combine<F: PrimeField>(values: &[F], fx: &[F], coeff: &F) -> F {
    let eq_eval = values[0];
    let h_eval = values[1];
    let g_evals = &values[2..];
    let g_products = products_except_self(g_evals);

    // g_products is the product of all the g except self
    let mut sum = F::zero();
    for (g_product, f) in zip(&g_products, fx) {
        sum += *f * *g_product;
    }
    // Combined sumcheck on h and zerocheck for well-formation of h
    h_eval + *coeff * eq_eval * (h_eval * g_products[0] * g_evals[0] - sum)
}

impl<F> MultiRationalSumcheck<F> for PolyIOP<F>
where
    F: PrimeField,
{
    type MultilinearExtension = Arc<DenseMultilinearExtension<F>>;
    type Transcript = IOPTranscript<F>;
    type MultiRationalSumcheckSubClaim = MultiRationalSumcheckSubClaim<F>;
    type MultiRationalSumcheckProof = MultiRationalSumcheckProof<F>;

    fn init_transcript() -> Self::Transcript {
        IOPTranscript::<F>::new(b"Initializing MultiRationalSumcheck transcript")
    }

    // Proves the Rational Sumcheck relation as a batched statement of multiple
    // independent instances f, g, g_inv
    fn prove(
        fx: &[F],
        mut gx: Vec<Self::MultilinearExtension>,
        h: Self::MultilinearExtension,
        claimed_sum: F,
        transcript: &mut IOPTranscript<F>,
    ) -> Result<(Self::MultiRationalSumcheckProof, Vec<F>), PolyIOPErrors> {
        let start = start_timer!(|| "rational_sumcheck prove");

        if fx.len() != gx.len() {
            return Err(PolyIOPErrors::InvalidParameters(format!(
                "polynomials lengthes are not equal"
            )));
        }

        transcript.append_field_element(b"rational_sumcheck_claim", &claimed_sum)?;

        let num_vars = gx[0].num_vars;
        let num_polys = gx.len();
        let r = transcript.get_and_append_challenge_vectors(b"0check r", num_vars)?;
        let coeff = transcript.get_and_append_challenge(b"coeff")?;

        let eq_poly = build_eq_x_r(&r)?;
        let mut polys = vec![eq_poly, h];
        polys.append(&mut gx);
        let (sumcheck_proof, point, _) = SumcheckInstanceProof::<F>::prove_arbitrary(
            &claimed_sum,
            num_vars,
            &mut polys,
            |evals| multi_rational_sumcheck_combine(evals, fx, &coeff),
            num_polys + 2,
            transcript,
        );

        end_timer!(start);

        Ok((MultiRationalSumcheckProof {
            sumcheck_proof,
            num_rounds: num_vars,
            num_polys,
            claimed_sum,
        }, point))
    }

    fn d_prove(
        fx: &[F],
        mut gx: Vec<Self::MultilinearExtension>,
        h: Self::MultilinearExtension,
        claimed_sum: F,
        transcript: &mut IOPTranscript<F>,
    ) -> Result<Option<(Self::MultiRationalSumcheckProof, Vec<F>)>, PolyIOPErrors> {
        let start = start_timer!(|| "rational_sumcheck prove");

        if fx.len() != gx.len() {
            return Err(PolyIOPErrors::InvalidParameters(format!(
                "polynomials lengthes are not equal"
            )));
        }

        if Net::am_master() {
            transcript.append_field_element(b"rational_sumcheck_claim", &claimed_sum)?;
        }

        let length = gx[0].num_vars;
        let num_polys = gx.len();
        let num_party_vars = Net::n_parties().log_2();
        let r = if Net::am_master() {
            let r = transcript
                .get_and_append_challenge_vectors(b"0check r", length + num_party_vars)?;
            Net::recv_from_master_uniform(Some(r))
        } else {
            Net::recv_from_master_uniform(None)
        };
        let coeff = if Net::am_master() {
            let coeff = transcript.get_and_append_challenge(b"coeff")?;
            Net::recv_from_master_uniform(Some(coeff))
        } else {
            Net::recv_from_master_uniform(None)
        };

        let index_vec: Vec<F> = bit_decompose(Net::party_id() as u64, num_party_vars)
            .into_iter()
            .map(|x| F::from(x))
            .collect();
        let eq_coeff = eq_eval(&r[length..], &index_vec)?;

        let eq_poly = build_eq_x_r_with_coeff(&r[..length], &eq_coeff)?;
        let mut polys = vec![eq_poly, h];
        polys.append(&mut gx);
        let result = SumcheckInstanceProof::<F>::d_prove_arbitrary(
            &claimed_sum,
            length,
            &mut polys,
            |evals| multi_rational_sumcheck_combine(evals, fx, &coeff),
            num_polys + 2,
            transcript,
        );

        end_timer!(start);

        if Net::am_master() {
            let (sumcheck_proof, point, _) = result.unwrap();
            Ok(Some((MultiRationalSumcheckProof {
                sumcheck_proof,
                num_rounds: length + num_party_vars,
                num_polys,
                claimed_sum,
            }, point)))
        } else {
            Ok(None)
        }
    }

    fn verify(
        proof: &Self::MultiRationalSumcheckProof,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::MultiRationalSumcheckSubClaim, PolyIOPErrors> {
        let start = start_timer!(|| "rational_sumcheck verify");

        transcript.append_serializable_element(b"rational_sumcheck_claim", &proof.claimed_sum)?;

        let zerocheck_r =
            transcript.get_and_append_challenge_vectors(b"0check r", proof.num_rounds)?;

        let coeff = transcript.get_and_append_challenge(b"coeff")?;

        let (sumcheck_expected_evaluation, sumcheck_point) = proof.sumcheck_proof.verify(
            proof.claimed_sum,
            proof.num_rounds,
            proof.num_polys + 2,
            transcript,
        )?;

        end_timer!(start);

        Ok(MultiRationalSumcheckSubClaim {
            sumcheck_expected_evaluation,
            sumcheck_point,
            zerocheck_r,
            coeff,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use arithmetic::eq_eval;
    use ark_bls12_381::Fr;
    use ark_ff::batch_inversion;
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::{test_rng, UniformRand, Zero};
    use rand::RngCore;
    use std::sync::Arc;

    fn create_polys<R: RngCore>(
        num_vars: usize,
        num_polys: usize,
        rng: &mut R,
    ) -> (
        Vec<Fr>,
        Vec<Arc<DenseMultilinearExtension<Fr>>>,
        Arc<DenseMultilinearExtension<Fr>>,
    ) {
        let fx = std::iter::repeat_with(|| Fr::rand(rng))
            .take(num_polys)
            .collect::<Vec<_>>();

        let gx = std::iter::repeat_with(|| {
            let evals_g = std::iter::repeat_with(|| {
                let mut val = Fr::zero();
                while val == Fr::zero() {
                    val = Fr::rand(rng);
                }
                val
            })
            .take(1 << num_vars)
            .collect();

            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_vars, evals_g,
            ))
        })
        .take(num_polys)
        .collect::<Vec<_>>();

        let inv = gx
            .iter()
            .map(|poly| {
                let mut evals = poly.evaluations.clone();
                batch_inversion(&mut evals);
                evals
            })
            .collect::<Vec<_>>();

        let h_evals = (0..(1 << num_vars))
            .map(|i| {
                fx.iter()
                    .zip(inv.iter())
                    .map(|(f, inv)| *f * inv[i])
                    .sum::<Fr>()
            })
            .collect();

        (
            fx,
            gx,
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_vars, h_evals,
            )),
        )
    }

    #[test]
    fn test_multi_rational_sumcheck() -> Result<(), PolyIOPErrors> {
        let num_vars = 5;
        let num_polys = 7;

        let mut rng = test_rng();
        let (fx, gx, h) = create_polys(num_vars, num_polys, &mut rng);
        let claimed_sum = h.evaluations.iter().sum::<Fr>();

        let mut transcript = <PolyIOP<Fr> as MultiRationalSumcheck<Fr>>::init_transcript();
        let (proof, point) = <PolyIOP<Fr> as MultiRationalSumcheck<Fr>>::prove(
            &fx,
            gx.iter()
                .map(|poly| Arc::new(DenseMultilinearExtension::clone(&poly)))
                .collect(),
            Arc::new(DenseMultilinearExtension::clone(&h)),
            claimed_sum,
            &mut transcript,
        )?;
        let mut transcript = <PolyIOP<Fr> as MultiRationalSumcheck<Fr>>::init_transcript();
        let subclaim = <PolyIOP<Fr> as MultiRationalSumcheck<Fr>>::verify(&proof, &mut transcript)?;
        assert_eq!(point, subclaim.sumcheck_point);

        let eq = eq_eval(&subclaim.sumcheck_point, &subclaim.zerocheck_r)?;
        let g_evals = gx
            .iter()
            .map(|poly| poly.evaluate(&subclaim.sumcheck_point).unwrap())
            .collect::<Vec<_>>();
        let h_eval = h.evaluate(&subclaim.sumcheck_point).unwrap();
        let mut sum = h_eval * g_evals.iter().product::<Fr>();
        for i in 0..num_polys {
            let mut product_others = fx[i];
            for j in 0..num_polys {
                if j != i {
                    product_others *= g_evals[j];
                }
            }
            sum -= product_others;
        }
        assert_eq!(h_eval + subclaim.coeff * eq * sum, subclaim.sumcheck_expected_evaluation);
        Ok(())
    }
}

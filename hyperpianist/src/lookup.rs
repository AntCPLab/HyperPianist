use ark_ec::pairing::Pairing;
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use ark_std::log2;
use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};
use std::{iter::zip, marker::PhantomData, sync::Arc};
use subroutines::{
    pcs::prelude::PolynomialCommitmentScheme, JoltInstruction, LookupCheck, PolyIOP, PolyIOPErrors,
};
use transcript::IOPTranscript;

pub struct HyperPlonkLookupProverOpeningPoints<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub regular_advices: Vec<PCS::ProverCommitmentAdvice>,
    // Second item is advice index
    pub regular_openings: Vec<(PCS::Polynomial, usize, Vec<E::ScalarField>)>,
    pub witness_openings: Vec<Vec<E::ScalarField>>,
}

pub struct HyperPlonkLookupVerifierOpeningPoints<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub regular_openings: Vec<(PCS::Commitment, Vec<E::ScalarField>)>,
    pub witness_openings: Vec<Vec<E::ScalarField>>,
}

pub trait HyperPlonkLookupPlugin<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Ops: Sync;
    type Preprocessing: Clone + Sync;
    type Transcript;
    type Proof: Send + Sync + CanonicalSerialize + CanonicalDeserialize;

    fn preprocess() -> Self::Preprocessing;
    fn construct_witnesses(ops: &Self::Ops) -> Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>;
    fn num_witness_columns() -> Vec<usize>;
    fn max_num_variables() -> usize;
    fn prove(
        preprocessing: &Self::Preprocessing,
        pcs_param: &PCS::ProverParam,
        ops: &Self::Ops,
        transcript: &mut Self::Transcript,
    ) -> (Self::Proof, HyperPlonkLookupProverOpeningPoints<E, PCS>);
    fn d_prove(
        preprocessing: &Self::Preprocessing,
        pcs_param: &PCS::ProverParam,
        ops: &Self::Ops,
        transcript: &mut Self::Transcript,
    ) -> (
        Option<Self::Proof>,
        HyperPlonkLookupProverOpeningPoints<E, PCS>,
    );
    fn num_regular_openings(proof: &Self::Proof) -> usize;
    fn verify(
        proof: &Self::Proof,
        witness_openings: &[E::ScalarField],
        regular_openings: &[E::ScalarField],
        transcript: &mut Self::Transcript,
    ) -> Result<HyperPlonkLookupVerifierOpeningPoints<E, PCS>, PolyIOPErrors>;
}

pub struct HyperPlonkLookupPluginSingle<
    Instruction: JoltInstruction + Default,
    const C: usize,
    const M: usize,
> {
    marker: PhantomData<Instruction>,
}

impl<E, PCS, Instruction, const C: usize, const M: usize> HyperPlonkLookupPlugin<E, PCS>
    for HyperPlonkLookupPluginSingle<Instruction, C, M>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E, Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>>,
    Instruction: JoltInstruction + Default,
{
    type Ops = Vec<Instruction>;
    type Preprocessing =
        <PolyIOP<E::ScalarField> as LookupCheck<E, PCS, Instruction, C, M>>::Preprocessing;
    type Proof =
        <PolyIOP<E::ScalarField> as LookupCheck<E, PCS, Instruction, C, M>>::LookupCheckProof;
    type Transcript = IOPTranscript<E::ScalarField>;

    fn preprocess() -> Self::Preprocessing {
        <PolyIOP<E::ScalarField> as LookupCheck<E, PCS, Instruction, C, M>>::preprocess()
    }
    fn construct_witnesses(ops: &Self::Ops) -> Vec<Arc<DenseMultilinearExtension<E::ScalarField>>> {
        <PolyIOP<E::ScalarField> as LookupCheck<E, PCS, Instruction, C, M>>::construct_witnesses(
            ops,
        )
    }
    fn num_witness_columns() -> Vec<usize> {
        vec![3]
    }
    fn max_num_variables() -> usize {
        log2(M) as usize
    }

    fn prove(
        preprocessing: &Self::Preprocessing,
        pcs_param: &PCS::ProverParam,
        ops: &Self::Ops,
        transcript: &mut Self::Transcript,
    ) -> (Self::Proof, HyperPlonkLookupProverOpeningPoints<E, PCS>) {
        let alpha = transcript
            .get_and_append_challenge(b"lookup_alpha")
            .unwrap();
        let tau = transcript.get_and_append_challenge(b"lookup_tau").unwrap();

        let mut polys =
            <PolyIOP<E::ScalarField> as LookupCheck<E, PCS, Instruction, C, M>>::construct_polys(
                preprocessing,
                ops,
                &alpha,
            );

        #[cfg(feature = "rational_sumcheck_piop")]
        let (proof, advices, r_f, r_g, r_z, r_primary_sumcheck, f_inv, g_inv) =
            <PolyIOP<E::ScalarField> as LookupCheck<E, PCS, Instruction, C, M>>::prove(
                preprocessing,
                pcs_param,
                &mut polys,
                &alpha,
                &tau,
                transcript,
            )
            .unwrap();

        #[cfg(not(feature = "rational_sumcheck_piop"))]
        let (proof, advices, r_f, r_g, r_z, r_primary_sumcheck) =
            <PolyIOP<E::ScalarField> as LookupCheck<E, PCS, Instruction, C, M>>::prove(
                preprocessing,
                pcs_param,
                &mut polys,
                &alpha,
                &tau,
                transcript,
            )
            .unwrap();

        #[cfg(feature = "rational_sumcheck_piop")]
        let mut regular_openings = Vec::with_capacity(
            polys.dim.len() + polys.m.len() + 2 * polys.E_polys.len() + f_inv.len() + g_inv.len(),
        );

        #[cfg(not(feature = "rational_sumcheck_piop"))]
        let mut regular_openings =
            Vec::with_capacity(polys.dim.len() + polys.m.len() + 2 * polys.E_polys.len());

        for (i, poly) in polys.m.iter().enumerate() {
            regular_openings.push((
                poly.clone(),
                polys.dim.len() + polys.E_polys.len() + i,
                r_f.clone(),
            ));
        }
        for (i, poly) in polys.dim.iter().enumerate() {
            regular_openings.push((poly.clone(), i, r_g.clone()));
        }
        for (i, poly) in polys.E_polys.iter().enumerate() {
            regular_openings.push((poly.clone(), polys.dim.len() + i, r_g.clone()));
        }
        for (i, poly) in polys.E_polys.iter().enumerate() {
            regular_openings.push((poly.clone(), polys.dim.len() + i, r_z.clone()));
        }
        #[cfg(feature = "rational_sumcheck_piop")]
        {
            let offset = polys.dim.len() + polys.m.len() + polys.E_polys.len();
            for (i, poly) in f_inv.iter().enumerate() {
                regular_openings.push((poly.clone(), offset + i, r_f.clone()));
            }
            for (i, poly) in g_inv.iter().enumerate() {
                regular_openings.push((poly.clone(), offset + f_inv.len() + i, r_g.clone()));
            }
        }

        (
            proof,
            HyperPlonkLookupProverOpeningPoints {
                regular_advices: advices,
                regular_openings,
                witness_openings: vec![r_primary_sumcheck.clone(); 3],
            },
        )
    }

    fn d_prove(
        preprocessing: &Self::Preprocessing,
        pcs_param: &PCS::ProverParam,
        ops: &Self::Ops,
        transcript: &mut Self::Transcript,
    ) -> (
        Option<Self::Proof>,
        HyperPlonkLookupProverOpeningPoints<E, PCS>,
    ) {
        let (alpha, tau) = if Net::am_master() {
            let alpha = transcript
                .get_and_append_challenge(b"lookup_alpha")
                .unwrap();
            let tau = transcript.get_and_append_challenge(b"lookup_tau").unwrap();
            Net::recv_from_master_uniform(Some((alpha, tau)));
            (alpha, tau)
        } else {
            Net::recv_from_master_uniform(None)
        };

        let mut polys =
            <PolyIOP<E::ScalarField> as LookupCheck<E, PCS, Instruction, C, M>>::construct_polys(
                preprocessing,
                ops,
                &alpha,
            );

        polys.collect_m_polys();

        #[cfg(feature = "rational_sumcheck_piop")]
        let (proof, advices, r_f, r_g, r_z, r_primary_sumcheck, f_inv, g_inv) =
            <PolyIOP<E::ScalarField> as LookupCheck<E, PCS, Instruction, C, M>>::d_prove(
                preprocessing,
                pcs_param,
                &mut polys,
                &alpha,
                &tau,
                transcript,
            )
            .unwrap();

        #[cfg(not(feature = "rational_sumcheck_piop"))]
        let (proof, advices, r_f, r_g, r_z, r_primary_sumcheck) =
            <PolyIOP<E::ScalarField> as LookupCheck<E, PCS, Instruction, C, M>>::d_prove(
                preprocessing,
                pcs_param,
                &mut polys,
                &alpha,
                &tau,
                transcript,
            )
            .unwrap();

        #[cfg(feature = "rational_sumcheck_piop")]
        let mut regular_openings = Vec::with_capacity(
            polys.dim.len() + polys.m.len() + 2 * polys.E_polys.len() + f_inv.len() + g_inv.len(),
        );

        #[cfg(not(feature = "rational_sumcheck_piop"))]
        let mut regular_openings =
            Vec::with_capacity(polys.dim.len() + polys.m.len() + 2 * polys.E_polys.len());

        for (i, poly) in polys.m.iter().enumerate() {
            regular_openings.push((
                poly.clone(),
                polys.dim.len() + polys.E_polys.len() + i,
                r_f.clone(),
            ));
        }
        for (i, poly) in polys.dim.iter().enumerate() {
            regular_openings.push((poly.clone(), i, r_g.clone()));
        }
        for (i, poly) in polys.E_polys.iter().enumerate() {
            regular_openings.push((poly.clone(), polys.dim.len() + i, r_g.clone()));
        }
        for (i, poly) in polys.E_polys.iter().enumerate() {
            regular_openings.push((poly.clone(), polys.dim.len() + i, r_z.clone()));
        }
        #[cfg(feature = "rational_sumcheck_piop")]
        {
            let offset = polys.dim.len() + polys.m.len() + polys.E_polys.len();
            for (i, poly) in f_inv.iter().enumerate() {
                regular_openings.push((poly.clone(), offset + i, r_f.clone()));
            }
            for (i, poly) in g_inv.iter().enumerate() {
                regular_openings.push((poly.clone(), offset + f_inv.len() + i, r_g.clone()));
            }
        }

        (
            proof,
            HyperPlonkLookupProverOpeningPoints {
                regular_advices: advices,
                regular_openings,
                witness_openings: vec![r_primary_sumcheck.clone(); 3],
            },
        )
    }

    fn num_regular_openings(proof: &Self::Proof) -> usize {
        let mut len = proof.commitment.dim_commitment.len()
            + proof.commitment.m_commitment.len()
            + 2 * proof.commitment.E_commitment.len();

        #[cfg(feature = "rational_sumcheck_piop")]
        {
            len += proof.logup_checking.f_inv_comm.len() + proof.logup_checking.g_inv_comm.len();
        }

        len
    }
    fn verify(
        proof: &Self::Proof,
        witness_openings: &[E::ScalarField],
        regular_openings: &[E::ScalarField],
        transcript: &mut Self::Transcript,
    ) -> Result<HyperPlonkLookupVerifierOpeningPoints<E, PCS>, PolyIOPErrors> {
        let alpha = transcript
            .get_and_append_challenge(b"lookup_alpha")
            .unwrap();
        let tau = transcript.get_and_append_challenge(b"lookup_tau").unwrap();

        let subclaim = <PolyIOP<E::ScalarField> as LookupCheck<E, PCS, Instruction, C, M>>::verify(
            proof, transcript,
        )?;

        let mut offset = 0;
        let mut next_openings = |len| {
            let result = &regular_openings[offset..offset + len];
            offset += len;
            result
        };

        let m_openings = next_openings(proof.commitment.m_commitment.len());
        let dim_openings = next_openings(proof.commitment.dim_commitment.len());
        let E_openings = next_openings(2 * proof.commitment.E_commitment.len());

        #[cfg(feature = "rational_sumcheck_piop")]
        {
            let f_inv_openings = next_openings(proof.logup_checking.f_inv_comm.len());
            let g_inv_openings = next_openings(proof.logup_checking.g_inv_comm.len());
            <PolyIOP<E::ScalarField> as LookupCheck<E, PCS, Instruction, C, M>>::check_openings(
                &subclaim,
                dim_openings,
                E_openings,
                m_openings,
                &witness_openings,
                f_inv_openings,
                g_inv_openings,
                &alpha,
                &tau,
            )?;
        }

        #[cfg(not(feature = "rational_sumcheck_piop"))]
        <PolyIOP<E::ScalarField> as LookupCheck<E, PCS, Instruction, C, M>>::check_openings(
            &subclaim,
            dim_openings,
            E_openings,
            m_openings,
            &witness_openings,
            &alpha,
            &tau,
        )?;

        let mut regular_openings = Vec::with_capacity(Self::num_regular_openings(proof));

        #[cfg(feature = "rational_sumcheck_piop")]
        let r_f = &subclaim
            .logup_checking
            .f_subclaims
            .sum_check_sub_claim
            .point;
        #[cfg(feature = "rational_sumcheck_piop")]
        let r_g = &subclaim
            .logup_checking
            .g_subclaims
            .sum_check_sub_claim
            .point;

        #[cfg(not(feature = "rational_sumcheck_piop"))]
        let r_f = &subclaim.logup_checking.point_f;
        #[cfg(not(feature = "rational_sumcheck_piop"))]
        let r_g = &subclaim.logup_checking.point_g;

        for comm in proof.commitment.m_commitment.iter() {
            regular_openings.push((comm.clone(), r_f.clone()));
        }
        for comm in proof.commitment.dim_commitment.iter() {
            regular_openings.push((comm.clone(), r_g.clone()));
        }
        for comm in proof.commitment.E_commitment.iter() {
            regular_openings.push((comm.clone(), r_g.clone()));
        }
        for comm in proof.commitment.E_commitment.iter() {
            regular_openings.push((comm.clone(), subclaim.r_z.clone()));
        }
        #[cfg(feature = "rational_sumcheck_piop")]
        {
            for comm in proof.logup_checking.f_inv_comm.iter() {
                regular_openings.push((comm.clone(), r_f.clone()));
            }
            for comm in proof.logup_checking.g_inv_comm.iter() {
                regular_openings.push((comm.clone(), r_g.clone()));
            }
        }

        Ok(HyperPlonkLookupVerifierOpeningPoints {
            regular_openings,
            witness_openings: vec![subclaim.r_primary_sumcheck.clone(); 3],
        })
    }
}

#[macro_export]
macro_rules! combine_lookup_plugins {
    ($name:ident : $($plugin:ty),*) => {
        pub struct $name {}

        impl<E, PCS> $crate::lookup::HyperPlonkLookupPlugin<E, PCS> for $name
            where E: ark_ec::pairing::Pairing,
            PCS: subroutines::pcs::prelude::PolynomialCommitmentScheme<E,
                Polynomial = std::sync::Arc<ark_poly::DenseMultilinearExtension<E::ScalarField>>,
                Point = Vec<E::ScalarField>,
                Evaluation = E::ScalarField,
                BatchProof = subroutines::BatchProof<E, PCS>,
            > {
            type Ops = ($(Option<<$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::Ops>,)*);
            type Preprocessing = ($(<$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::Preprocessing,)*);
            type Proof = ($(Option<<$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::Proof>,)*);
            type Transcript = transcript::IOPTranscript<E::ScalarField>;

            fn preprocess() -> Self::Preprocessing {
                ($(<$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::preprocess(),)*)
            }
            fn construct_witnesses(ops: &Self::Ops) -> Vec<std::sync::Arc<ark_poly::DenseMultilinearExtension<E::ScalarField>>> {
                let witness_vecs : Vec<Vec<std::sync::Arc<ark_poly::DenseMultilinearExtension<E::ScalarField>>>> = vec![$(if let Some(ops) = &ops.${index()} {
                    <$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::construct_witnesses(ops)
                } else {
                    vec![]
                }),*];
                witness_vecs.concat()
            }
            fn num_witness_columns() -> Vec<usize> {
                let witness_columns : Vec<Vec<usize>> =
                    vec![$(<$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::num_witness_columns()),*];
                witness_columns.concat()
            }
            fn max_num_variables() -> usize {
                *vec![0, $(<$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::max_num_variables()),*].iter().max().unwrap()
            }
            fn prove(
                preprocessing: &Self::Preprocessing,
                pcs_param: &PCS::ProverParam,
                ops: &Self::Ops,
                transcript: &mut Self::Transcript,
            ) -> (Self::Proof, $crate::lookup::HyperPlonkLookupProverOpeningPoints<E, PCS>) {
                let mut all_openings = $crate::lookup::HyperPlonkLookupProverOpeningPoints {
                    regular_advices: vec![],
                    regular_openings: vec![],
                    witness_openings: vec![],
                };

                (($(
                    if let Some(ops) = &ops.${index()} {
                        let (proof, mut openings) = <$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::prove(&preprocessing.${index()}, pcs_param, ops, transcript);
                        all_openings.regular_openings.extend(openings.regular_openings.iter_mut()
                            .map(|opening| {
                                let (poly, advice_idx, point) = opening;
                                (std::mem::take(poly),  *advice_idx + all_openings.regular_advices.len(),
                                std::mem::take(point))
                            }));
                        all_openings.regular_advices.append(&mut openings.regular_advices);
                        all_openings.witness_openings.append(&mut openings.witness_openings);
                        Some(proof)
                    } else {
                        None
                    }
                ,)*), all_openings)
            }
            fn d_prove(
                preprocessing: &Self::Preprocessing,
                pcs_param: &PCS::ProverParam,
                ops: &Self::Ops,
                transcript: &mut Self::Transcript
            ) -> (Option<Self::Proof>, $crate::lookup::HyperPlonkLookupProverOpeningPoints<E, PCS>) {
                let mut all_openings = $crate::lookup::HyperPlonkLookupProverOpeningPoints {
                    regular_advices: vec![],
                    regular_openings: vec![],
                    witness_openings: vec![],
                };

                let proofs = ($(
                    if let Some(ops) = &ops.${index()} {
                        let (proof, mut openings) = <$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::d_prove(&preprocessing.${index()}, pcs_param, ops, transcript);
                        all_openings.regular_openings.extend(openings.regular_openings.iter_mut()
                            .map(|opening| {
                                let (poly, advice_idx, point) = opening;
                                (std::mem::take(poly),  *advice_idx + all_openings.regular_advices.len(),
                                std::mem::take(point))
                            }));
                        all_openings.regular_advices.append(&mut openings.regular_advices);
                        all_openings.witness_openings.append(&mut openings.witness_openings);
                        proof
                    } else {
                        None
                    }
                ,)*);
                if Net::am_master() {
                    (Some(proofs), all_openings)
                } else {
                    (None, all_openings)
                }
            }
            fn num_regular_openings(
                proof: &Self::Proof,
            ) -> usize {
                vec![
                    $(if let Some(proof) = &proof.${index()} {
                        <$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::num_regular_openings(&proof)
                    } else {
                        0
                    }),*
                ].iter().sum()
            }
            fn verify(
                proof: &Self::Proof,
                witness_openings: &[E::ScalarField],
                regular_openings: &[E::ScalarField],
                transcript: &mut Self::Transcript,
            ) -> Result<$crate::lookup::HyperPlonkLookupVerifierOpeningPoints<E, PCS>, subroutines::PolyIOPErrors> {
                let mut witness_index = 0;
                let mut regular_index = 0;
                let mut all_openings = $crate::lookup::HyperPlonkLookupVerifierOpeningPoints {
                    regular_openings: vec![],
                    witness_openings: vec![],
                };

                $(if let Some(proof) = &proof.${index()} {
                    let num_witnesses : usize = <$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::num_witness_columns().iter().sum();
                    let num_regular_openings = <$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::num_regular_openings(proof);

                    let mut openings = <$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::verify(proof,
                        &witness_openings[witness_index..witness_index + num_witnesses],
                        &regular_openings[regular_index..regular_index + num_regular_openings],
                        transcript)?;
                    all_openings.regular_openings.append(&mut openings.regular_openings);
                    all_openings.witness_openings.append(&mut openings.witness_openings);

                    witness_index += num_witnesses;
                    regular_index += num_regular_openings;
                })*

                Ok(all_openings)
            }
        }
    };
}

combine_lookup_plugins! { HyperPlonkLookupPluginNull : }

#[macro_export]
macro_rules! jolt_lookup {
    ($name:ident, $C:expr, $M:expr; $($inst:ty),*) => {
        $crate::combine_lookup_plugins! { $name : $($crate::lookup::HyperPlonkLookupPluginSingle<$inst, $C, $M>),* }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use subroutines::instruction::xor::XORInstruction;

    combine_lookup_plugins! { HyperPlonkLookupPluginTest1 : HyperPlonkLookupPluginSingle<XORInstruction, 4, 65536>}
    combine_lookup_plugins! { HyperPlonkLookupPluginTest2 : HyperPlonkLookupPluginSingle<XORInstruction, 4, 65536>,
    HyperPlonkLookupPluginSingle<XORInstruction, 4, 65536>}
    combine_lookup_plugins! { HyperPlonkLookupPluginTest4 : HyperPlonkLookupPluginSingle<XORInstruction, 4, 65536>,
    HyperPlonkLookupPluginSingle<XORInstruction, 4, 65536>,
    HyperPlonkLookupPluginSingle<XORInstruction, 4, 65536>,
    HyperPlonkLookupPluginSingle<XORInstruction, 4, 65536>}
    jolt_lookup! { JoltLookupTest1, 4, 65536; XORInstruction }
    jolt_lookup! { JoltLookupTest3, 4, 65536; XORInstruction, XORInstruction, XORInstruction }
}

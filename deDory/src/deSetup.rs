use crate::base::{impl_serde_for_ark_serde_unchecked, rand_util};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use num_traits::One;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::error::Error;

/// The public parameters for the Dory protocol. See section 5 of https://eprint.iacr.org/2020/1274.pdf for details.
///
/// Note: even though H_1 and H_2 are marked as blue, they are still needed.
///
/// Note: Gamma_1_fin is unused, so we leave it out.
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct PublicParameters<E: Pairing> {
    /// This is the vector of G1 elements that are used in the Dory protocol.
    /// That is, Γ_1,0 in the Dory paper.
    pub(super) Gamma_1: Vec<E::G1Affine>,
    /// This is the vector of G2 elements that are used in the Dory protocol.
    /// That is, Γ_2,0 in the Dory paper.
    pub(super) Gamma_2: Vec<E::G2Affine>,
    /// `H_1` = H_1 in the Dory paper. This could be used for blinding, but is
    /// currently only used in the Fold-Scalars algorithm.
    pub(super) H_1: E::G1Affine,
    /// `H_2` = H_2 in the Dory paper. This could be used for blinding, but is
    /// currently only used in the Fold-Scalars algorithm.
    pub(super) H_2: E::G2Affine,
    /// `Gamma_2_fin` = Gamma_2,fin in the Dory paper.
    pub(super) Gamma_2_fin: E::G2Affine,
    /// `max_num` is the maximum num that this setup will work for.
    pub(super) max_num: usize,
}

impl<E: Pairing> Default for PublicParameters<E> {
    fn default() -> Self {
        Self {
            Gamma_1: vec![],
            Gamma_2: vec![],
            H_1: E::G1Affine::default(),
            H_2: E::G2Affine::default(),
            Gamma_2_fin: E::G2Affine::default(),
            max_num: 0,
        }
    }
}

impl<E: Pairing> PublicParameters<E> {
    /// Generate random public parameters for testing purposes.
    pub fn rand(max_num: usize, rng: &mut impl ark_std::rand::Rng) -> Self {
        use ark_std::UniformRand;
        // Generate 2^max_num random group elements.
        let (Gamma_1, Gamma_2) = rand_util::rand_G_vecs::<E>(1 << max_num, rng);
        let (H_1, H_2) = (E::G1Affine::rand(rng), E::G2Affine::rand(rng));
        let Gamma_2_fin = E::G2Affine::rand(rng);

        Self {
            Gamma_1,
            Gamma_2,
            max_num,
            H_1,
            H_2,
            Gamma_2_fin,
        }
    }
}


#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct SubProverSetup<E: Pairing> {
    /// `Gamma_1[k]` = Γ_1,(m-k) in the Dory paper.
    pub(super) Gamma_1: Vec<E::G1Affine>,
    /// `Gamma_2[k]` = Γ_2,(m-k) in the Dory paper.
    pub(super) Gamma_2: Vec<E::G2Affine>,
    /// `H_1` = H_1 in the Dory paper. This could be used for blinding, but is
    /// currently only used in the Fold-Scalars algorithm.
    pub(super) H_1: E::G1Affine,
    /// `H_2` = H_2 in the Dory paper. This could be used for blinding, but is
    /// currently only used in the Fold-Scalars algorithm.
    pub(super) H_2: E::G2Affine,
    /// `Gamma_2_fin` = Gamma_2,fin in the Dory paper.
    pub(super) Gamma_2_fin: E::G2Affine,
    /// `max_num` is the maximum num that this setup will work for.
    pub(super) max_num: usize,
}

impl_serde_for_ark_serde_unchecked!(SubProverSetup);

impl<E: Pairing> SubProverSetup<E> {
    /// Create a new `ProverSetup` from the public parameters.

    pub fn new(
        pp: &PublicParameters<E>, 
    ) -> Self {
        assert_eq!(pp.Gamma_1.len(), 1 << pp.max_num);
        assert_eq!(pp.Gamma_2.len(), 1 << pp.max_num);

        SubProverSetup {
            Gamma_1: pp.Gamma_1.clone(),
            Gamma_2: pp.Gamma_2.clone(),
            H_1: pp.H_1,
            H_2: pp.H_2,
            Gamma_2_fin: pp.Gamma_2_fin,
            max_num: pp.max_num,
        }
    }

    /// Create a new `ProverSetup` from the public parameters and write it to a file.
    pub fn new_to_file(
        pp: &PublicParameters<E>, 
        file_path: &str
    ) -> Result<(), Box<dyn Error>> {
        let setup = Self::new(pp);
        let file = std::fs::File::create(file_path)?;
        let writer = std::io::BufWriter::new(file);
        bincode::serialize_into(writer, &setup)?;
        Ok(())
    }

    pub fn read_from_file(
        file_path: &str
    ) -> Result<Self, Box<dyn Error>> {
        let file = std::fs::File::open(file_path)?;
        let reader = std::io::BufReader::new(file);
        let setup = bincode::deserialize_from(reader)?;
        Ok(setup)
    }
}


/// The transparent setup information that the verifier must know to verify a
/// proof. This is public knowledge and must match with the prover's setup
/// information. See Section 3.3 of https://eprint.iacr.org/2020/1274.pdf for details.
///
///
/// Note:
/// We use num = m and k = m-i or m-j.
/// This indexing is more convenient for coding because lengths of the arrays
/// used are typically 2^k rather than 2^i or 2^j.
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct VerifierSetup<E: Pairing> {
    /// `Delta_1L[k]` = Δ_1L,(m-k) in the Dory paper, so `Delta_1L[0]` is
    /// unused. Note, this is the same as `Delta_2L`.
    pub(super) Delta_1L: Vec<PairingOutput<E>>,
    /// `Delta_1R[k]` = Δ_1R,(m-k) in the Dory paper, so `Delta_1R[0]` is
    /// unused.
    pub(super) Delta_1R: Vec<PairingOutput<E>>,
    /// `Delta_2L[k]` = Δ_2L,(m-k) in the Dory paper, so `Delta_2L[0]` is
    /// unused. Note, this is the same as `Delta_1L`.
    pub(super) Delta_2L: Vec<PairingOutput<E>>,
    /// `Delta_2R[k]` = Δ_2R,(m-k) in the Dory paper, so `Delta_2R[0]` is
    /// unused.
    pub(super) Delta_2R: Vec<PairingOutput<E>>,
    /// `chi[k]` = χ,(m-k) in the Dory paper.
    pub(super) chi: Vec<PairingOutput<E>>,
    /// `Gamma_1_0` is the Γ_1 used in Scalar-Product algorithm in the Dory
    /// paper.
    pub(super) Gamma_1_0: E::G1Affine,
    /// `Gamma_2_0` is the Γ_2 used in Scalar-Product algorithm in the Dory
    /// paper.
    pub(super) Gamma_2_0: E::G2Affine,
    /// `H_1` = H_1 in the Dory paper. This could be used for blinding, but is
    /// currently only used in the Fold-Scalars algorithm.
    pub(super) H_1: E::G1Affine,
    /// `H_2` = H_2 in the Dory paper. This could be used for blinding, but is
    /// currently only used in the Fold-Scalars algorithm.
    pub(super) H_2: E::G2Affine,
    /// `H_T` = H_T in the Dory paper.
    pub(super) H_T: PairingOutput<E>,
    /// `Gamma_2_fin` = Gamma_2,fin in the Dory paper.
    pub(super) Gamma_2_fin: E::G2Affine,
    /// `max_num` is the maximum num that this setup will work for
    pub(super) max_num: usize,
}

impl_serde_for_ark_serde_unchecked!(VerifierSetup);

impl<E: Pairing> VerifierSetup<E> {
    /// Create a new `VerifierSetup` from the public parameters.
    pub fn new(
        pp: &PublicParameters<E>,
    ) -> Self {
        assert_eq!(pp.Gamma_1.len(), 1 << pp.max_num);
        assert_eq!(pp.Gamma_2.len(), 1 << pp.max_num);
        let ((Delta_1L_2L, Delta_1R), (Delta_2R, chi)): ((Vec<_>, Vec<_>), (Vec<_>, Vec<_>)) =
            rayon::join(|| {
                (0..pp.max_num + 1)
                    .into_par_iter()
                    .map(|k| {
                        if k == 0 {
                            (
                                PairingOutput(One::one()),
                                PairingOutput(One::one()),
                            )
                        } else {
                            (
                                Pairing::multi_pairing(
                                    &pp.Gamma_1[..1 << (k - 1)],
                                    &pp.Gamma_2[..1 << (k - 1)],
                                ),
                                Pairing::multi_pairing(
                                    &pp.Gamma_1[1 << (k - 1)..1 << k],
                                    &pp.Gamma_2[..1 << (k - 1)],
                                ),
                            )
                        }
                    })
                    .unzip()
            }, || {
                (0..pp.max_num + 1)
                    .into_par_iter()
                    .map(|k| {
                        if k == 0 {
                            (
                                PairingOutput(One::one()),
                                Pairing::pairing(pp.Gamma_1[0], pp.Gamma_2[0]),
                            )
                        } else {
                            (
                                Pairing::multi_pairing(
                                    &pp.Gamma_1[..1 << (k - 1)],
                                    &pp.Gamma_2[1 << (k - 1)..1 << k],
                                ),
                                Pairing::multi_pairing(&pp.Gamma_1[..1 << k], &pp.Gamma_2[..1 << k]),
                            )
                        }
                    })
                    .unzip()
            });
        Self {
            Delta_1L: Delta_1L_2L.clone(),
            Delta_1R,
            Delta_2L: Delta_1L_2L,
            Delta_2R,
            chi,
            Gamma_1_0: pp.Gamma_1[0],
            Gamma_2_0: pp.Gamma_2[0],
            H_1: pp.H_1,
            H_2: pp.H_2,
            H_T: Pairing::pairing(pp.H_1, pp.H_2),
            Gamma_2_fin: pp.Gamma_2_fin,
            max_num: pp.max_num,
        }
    }

    /// Create a new `VerifierSetup` from the public parameters and write it to a file.
    pub fn new_to_file(
        pp: &PublicParameters<E>, 
        file_path: &str
    ) -> Result<(), Box<dyn Error>> {
        let setup = Self::new(pp);
        let file = std::fs::File::create(file_path).unwrap();
        let writer = std::io::BufWriter::new(file);
        bincode::serialize_into(writer, &setup)?;
        Ok(())
    }

    pub fn read_from_file(
        file_path: &str
    ) -> Result<Self, Box<dyn Error>> {
        let file = std::fs::File::open(file_path).unwrap();
        let reader = std::io::BufReader::new(file);
        let setup = bincode::deserialize_from(reader)?;
        Ok(setup)
    }
}
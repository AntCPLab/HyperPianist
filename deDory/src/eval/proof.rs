use super::transcript::MessageLabel;
use crate::base::impl_serde_for_ark_serde_checked;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use num_traits::Zero;
use std::mem;
use std::mem::take;
use thiserror::Error;
use transcript::IOPTranscript;

#[derive(Default, Clone, CanonicalSerialize, CanonicalDeserialize, PartialEq, Eq, Debug)]
/// The messages sent from the prover to the verifier in the interactive
/// protocol. This is, in essence, the proof.
///
/// This struct is effectively 4 queues.
/// The prover pushes messages to the front of a queue, and the verifier pops
/// messages from the back of a queue. However, this functionality is hidden
/// outside of `super`.
pub struct DoryEvalProof<E: Pairing> {
    /// The field elements sent from the prover to the verifier. The last
    /// element of the `Vec` is the first element sent.
    pub F_messages: Vec<E::ScalarField>,
    /// The G1 elements sent from the prover to the verifier. The last element
    /// of the `Vec` is the first element sent.
    pub G1_messages: Vec<E::G1Affine>,
    /// The G2 elements sent from the prover to the verifier. The last element
    /// of the `Vec` is the first element sent.
    pub G2_messages: Vec<E::G2Affine>,
    /// The GT elements sent from the prover to the verifier. The last element
    /// of the `Vec` is the first element sent.
    pub GT_messages: Vec<PairingOutput<E>>,

    pub F_read_index: usize,
    pub G1_read_index : usize,
    pub G2_read_index : usize,
    pub GT_read_index : usize,
}

impl_serde_for_ark_serde_checked!(DoryEvalProof);

impl<E: Pairing> DoryEvalProof<E> {
    pub(super) fn new() -> Self {
        Self {
            F_messages: Vec::new(),
            G1_messages: Vec::new(),
            G2_messages: Vec::new(),
            GT_messages: Vec::new(),
            F_read_index: 0,
            G1_read_index: 0,
            G2_read_index: 0,
            GT_read_index: 0,
        }
    }

    /// Pushes a field element from the prover onto the queue, and appends it to
    /// the transcript.
    pub(super) fn write_F_message(
        &mut self,
        transcript: &mut IOPTranscript<E::ScalarField>,
        message: E::ScalarField,
    ) {
        transcript
            .append_field_element(MessageLabel::DoryMessage.as_bytes(), &message)
            .unwrap();
        self.F_messages.push(message);
    }

    /// Pushes a G1 element from the prover onto the queue, and appends it to
    /// the transcript.
    pub(super) fn write_G1_message(
        &mut self,
        transcript: &mut IOPTranscript<E::ScalarField>,
        message: impl Into<E::G1Affine>,
    ) {
        let message = message.into();
        transcript
            .append_serializable_element(MessageLabel::DoryMessage.as_bytes(), &message)
            .unwrap();
        self.G1_messages.push(message);
    }

    /// Pushes a G2 element from the prover onto the queue, and appends it to
    /// the transcript.
    pub(super) fn write_G2_message(
        &mut self,
        transcript: &mut IOPTranscript<E::ScalarField>,
        message: impl Into<E::G2Affine>,
    ) {
        let message = message.into();
        transcript
            .append_serializable_element(MessageLabel::DoryMessage.as_bytes(), &message)
            .unwrap();
        self.G2_messages.push(message);
    }

    /// Pushes a GT element from the prover onto the queue, and appends it to
    /// the transcript.
    pub(super) fn write_GT_message(
        &mut self,
        transcript: &mut IOPTranscript<E::ScalarField>,
        message: PairingOutput<E>,
    ) {
        transcript
            .append_serializable_element(MessageLabel::DoryMessage.as_bytes(), &message)
            .unwrap();
        self.GT_messages.push(message);
    }

    /// Pops a field element from the verifier's queue, and appends it to the
    /// transcript.
    pub(super) fn read_F_message(
        &mut self,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> E::ScalarField {
        let message = take(&mut self.F_messages[self.F_read_index]);
        self.F_read_index += 1;
        transcript
            .append_serializable_element(MessageLabel::DoryMessage.as_bytes(), &message)
            .unwrap();
        message
    }

    /// Pops a G1 element from the verifier's queue, and appends it to the
    /// transcript.
    pub(super) fn read_G1_message(
        &mut self,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> E::G1Affine {
        let message = take(&mut self.G1_messages[self.G1_read_index]);
        self.G1_read_index += 1;
        transcript
            .append_serializable_element(MessageLabel::DoryMessage.as_bytes(), &message)
            .unwrap();
        message
    }

    /// Pops a G2 element from the verifier's queue, and appends it to the
    /// transcript.
    pub(super) fn read_G2_message(
        &mut self,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> E::G2Affine {
        let message = take(&mut self.G2_messages[self.G2_read_index]);
        self.G2_read_index += 1;
        transcript
            .append_serializable_element(MessageLabel::DoryMessage.as_bytes(), &message)
            .unwrap();
        message
    }

    /// Pops a GT element from the verifier's queue, and appends it to the
    /// transcript.
    pub(super) fn read_GT_message(
        &mut self,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> PairingOutput<E> {
        let message = take(&mut self.GT_messages[self.GT_read_index]);
        self.GT_read_index += 1;
        transcript
            .append_serializable_element(MessageLabel::DoryMessage.as_bytes(), &message)
            .unwrap();
        message
    }

    /// This is the F message that the verifier sends to the prover.
    /// This message is produces as a challenge from the transcript.
    ///
    /// While the message is a simple field element, we ensure that it is
    /// non-zero, and also return it's inverse.
    pub(super) fn get_challenge_scalar(
        &mut self,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> (E::ScalarField, E::ScalarField) {
        let mut message = E::ScalarField::zero();
        while message.is_zero() {
            message = transcript
                .get_and_append_challenge(MessageLabel::DoryChallenge.as_bytes())
                .unwrap();
        }
        let message_inv = message.inverse().unwrap();
        (message, message_inv)
    }

    pub fn get_size(&self) -> usize {
        let size: usize = if self.F_messages.len() > 0 {
            mem::size_of_val(&self.F_messages[0]) * self.F_messages.len()
        } else {
            0
        } + if self.G1_messages.len() > 0 {
            mem::size_of_val(&self.G1_messages[0]) * self.G1_messages.len()
        } else {
            0
        } + if self.G2_messages.len() > 0 {
            mem::size_of_val(&self.G2_messages[0]) * self.G2_messages.len()
        } else {
            0
        } + if self.GT_messages.len() > 0 {
            mem::size_of_val(&self.GT_messages[0]) * self.GT_messages.len()
        } else {
            0
        };
        size
    }
}

/// The error type for the Dory PCS.
#[derive(Error, Debug)]
pub enum DoryError {
    // /// This error occurs when the generators offset is invalid.
    // #[error("invalid generators offset: {0}")]
    // InvalidGeneratorsOffset(u64),
    /// This error occurs when the proof fails to verify.
    #[error("verification error")]
    VerificationError,
    /// This error occurs when the setup is too small.
    #[error("setup is too small: the setup is {0}, but the proof requires a setup of size {1}")]
    SmallSetup(usize, usize),
}

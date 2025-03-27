/// Labels for items in a merlin transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageLabel {
    /// Represents an inner product computation or its result.
    #[cfg(test)]
    InnerProduct,
    /// Represents a challenge in the computation of an inner product.
    #[cfg(test)]
    InnerProductChallenge,
    /// Denotes a sumcheck protocol message.
    Sumcheck,
    /// Represents a challenge in the sumcheck protocol.
    SumcheckChallenge,
    /// Represents a round evaluation in the sumcheck protocol.
    SumcheckRoundEvaluation,
    /// Represents a proof resulting from a query.
    QueryProof,
    /// Represents a commitment to a query.
    QueryCommit,
    /// Represents evaluations in an MLE context.
    QueryMleEvaluations,
    /// Represents a challenge in the context of MLE evaluations.
    QueryMleEvaluationsChallenge,
    /// Represents the data resulting from a query.
    QueryResultData,
    /// Represents a query for bit distribution data.
    QueryBitDistributions,
    /// Represents a challenge in a sumcheck query.
    QuerySumcheckChallenge,
    /// Represents a hash used for verification purposes.
    VerificationHash,
    /// Represents a message in the context of the Dory protocol.
    DoryMessage,
    /// Represents a challenge in the context of the Dory protocol.
    DoryChallenge,
    /// Represents challenges posted after result computation.
    PostResultChallenges,
    /// Represents a SQL query
    ProofExpr,
    /// Represents the length of a table.
    TableLength,
    /// Represents an offset for a generator.
    GeneratorOffset,
}

impl MessageLabel {
    /// Convert the label to a byte slice, which satisfies the requirements of a merlin label:
    /// "the labels should be distinct and none should be a prefix of any other."
    pub fn as_bytes(&self) -> &'static [u8] {
        match self {
            #[cfg(test)]
            MessageLabel::InnerProduct => b"ipp v1",
            #[cfg(test)]
            MessageLabel::InnerProductChallenge => b"ippchallenge v1",
            MessageLabel::Sumcheck => b"sumcheckproof v1",
            MessageLabel::SumcheckChallenge => b"sumcheckchallenge v1",
            MessageLabel::SumcheckRoundEvaluation => b"sumcheckroundevaluationscalars v1",
            MessageLabel::QueryProof => b"queryproof v1",
            MessageLabel::QueryCommit => b"querycommit v1",
            MessageLabel::QueryResultData => b"queryresultdata v1",
            MessageLabel::QueryBitDistributions => b"querybitdistributions v1",
            MessageLabel::QueryMleEvaluations => b"querymleevaluations v1",
            MessageLabel::QueryMleEvaluationsChallenge => b"querymleevaluationschallenge v1",
            MessageLabel::QuerySumcheckChallenge => b"querysumcheckchallenge v1",
            MessageLabel::VerificationHash => b"verificationhash v1",
            MessageLabel::DoryMessage => b"dorymessage v1",
            MessageLabel::DoryChallenge => b"dorychallenge v1",
            MessageLabel::PostResultChallenges => b"postresultchallenges v1",
            MessageLabel::ProofExpr => b"proofexpr v1",
            MessageLabel::TableLength => b"tablelength v1",
            MessageLabel::GeneratorOffset => b"generatoroffset v1",
        }
    }
}

#![feature(tcp_linger)]

pub mod multi;
pub mod two;

pub use two::DeTwoNet;
pub use multi::DeMultiNet;

pub mod channel;
pub use channel::DeSerNet;

#[derive(Clone, Debug)]
pub struct Stats {
    pub bytes_sent: usize,
    pub bytes_recv: usize,
    pub broadcasts: usize,
    pub to_master: usize,
    pub from_master: usize,
}

impl std::default::Default for Stats {
    fn default() -> Self {
        Self {
            bytes_sent: 0,
            bytes_recv: 0,
            broadcasts: 0,
            to_master: 0,
            from_master: 0,
        }
    }
}

pub trait DeNet {
    /// Am I the first party?
    #[inline]
    fn am_master() -> bool {
        Self::party_id() == 0
    }
    /// How many parties are there?
    fn n_parties() -> usize;
    /// What is my party number (0 to n-1)?
    fn party_id() -> usize;
    /// Initialize the network layer from a file.
    /// The file should contain one HOST:PORT setting per line, corresponding to the addresses of
    /// the parties in increasing order.
    ///
    /// Parties are zero-indexed.
    fn init_from_file(path: &str, party_id: usize);
    /// Is the network layer initalized?
    fn is_init() -> bool;
    /// Uninitialize the network layer, closing all connections.
    fn deinit();
    /// Set statistics to zero.
    fn reset_stats();
    /// Get statistics.
    fn stats() -> Stats;
    /// All parties send bytes to each other.
    fn broadcast_bytes(bytes: &[u8]) -> Vec<Vec<u8>>;
    /// All parties send bytes to the master.
    fn send_bytes_to_master(bytes: Vec<u8>) -> Option<Vec<Vec<u8>>>;
    /// All parties recv bytes from the master.
    /// Provide bytes iff you're the master!
    fn recv_bytes_from_master(bytes: Option<Vec<Vec<u8>>>) -> Vec<u8>;

    fn recv_bytes_from_master_uniform(bytes: Option<Vec<u8>>) -> Vec<u8>;

    /// Everyone sends bytes to the master, who recieves those bytes, runs a computation on them, and
    /// redistributes the resulting bytes.
    ///
    /// The master's computation is given by a function, `f`
    /// proceeds.
    #[inline]
    fn master_compute(bytes: Vec<u8>, f: impl Fn(Vec<Vec<u8>>) -> Vec<Vec<u8>>) -> Vec<u8> {
        let master_response = Self::send_bytes_to_master(bytes).map(f);
        Self::recv_bytes_from_master(master_response)
    }

    fn set_channel_id(channel_id: usize) {}
}

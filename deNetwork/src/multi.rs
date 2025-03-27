use crossbeam_channel::{Receiver, Select, Sender};
use lazy_static::lazy_static;
use log::debug;
use mio::{
    net::{TcpListener, TcpStream},
    Events, Interest, Poll, Token,
};
use rayon::prelude::*;
use std::{
    cell::Cell,
    collections::VecDeque,
    convert::TryInto,
    fs::File,
    io::{BufRead, BufReader, ErrorKind, Read, Write},
    net::SocketAddr,
    sync::{Mutex, RwLock},
    thread::{self, JoinHandle},
};

use ark_std::{end_timer, start_timer};

use super::{DeNet, Stats};

#[macro_use]
lazy_static! {
    static ref CONNECTIONS: RwLock<Connections> = RwLock::new(Connections::default());
    static ref STATS: Mutex<Stats> = Mutex::new(Stats::default());
}

/// Macro for locmaster the FieldChannel singleton in the current scope.
macro_rules! get_ch {
    () => {
        CONNECTIONS.read().expect("Poisoned FieldChannel")
    };
}

macro_rules! get_ch_mut {
    () => {
        CONNECTIONS.write().expect("Poisoned FieldChannel")
    };
}

#[derive(Debug)]
struct Peer {
    id: usize,
    addr: SocketAddr,
}

thread_local! {
    static CHANNEL_ID: Cell<usize> = Cell::new(0);
}

const MAX_NUM_CHANNELS: usize = 3;

enum Data {
    Single(Vec<u8>),
    Multi(Vec<Vec<u8>>),
    Shutdown,
}

#[derive(Default, Debug)]
struct Connections {
    id: usize,
    peers: Vec<Peer>,
    send_channel: Option<Sender<(usize, Data)>>,
    recv_channels: Vec<Receiver<Data>>,
    send_join_handle: Option<JoinHandle<()>>,
    recv_join_handle: Option<JoinHandle<()>>,
}

impl std::default::Default for Peer {
    fn default() -> Self {
        Self {
            id: 0,
            addr: "127.0.0.1:8000".parse().unwrap(),
        }
    }
}

fn parse_buffer(buf: &mut VecDeque<u8>, mut func: impl FnMut(usize, Vec<u8>) -> ()) {
    loop {
        if buf.len() < 9 {
            break;
        }

        let mut header = buf.range(..9);
        let channel_id = *header.next().unwrap();
        let m =
            u64::from_le_bytes(header.copied().collect::<Vec<_>>().try_into().unwrap()) as usize;
        if buf.len() < 9 + m {
            break;
        }

        buf.drain(..9);
        func(channel_id as usize, buf.drain(..m).collect())
    }
}

// Returns false if the socket is closed
fn read_to_buffer(
    stream: &mut impl Read,
    buf: &mut VecDeque<u8>,
    temp_buf: &mut [u8],
) -> Result<bool, std::io::Error> {
    loop {
        match stream.read(temp_buf) {
            Ok(0) => break Ok(false),
            Ok(size) => buf.extend(&temp_buf[..size]),
            Err(e) => match e.kind() {
                ErrorKind::WouldBlock => break Ok(true),
                _ => return Err(e),
            },
        }
    }
}

fn write_data(stream: &mut impl Write, channel_id: usize, data: &[u8]) {
    // This does not take into account WouldBlock errors (send buffer full?)
    // Hopefully we never write too much data at once
    let channel_id = [channel_id as u8];
    let bytes_size = (data.len() as u64).to_le_bytes();
    let actual_data = [&channel_id[..], &bytes_size[..], data].concat();
    stream.write_all(&actual_data).unwrap();
}

// These worker threads are globals
fn send_thread(
    own_id: usize,
    streams: &'static [Option<TcpStream>],
    channel: Receiver<(usize, Data)>,
) {
    let am_master = own_id == 0;

    loop {
        let (channel_id, data) = channel.recv().unwrap();
        match data {
            Data::Single(bytes_out) => {
                if am_master {
                    streams
                        .par_iter()
                        .enumerate()
                        .filter(|p| p.0 != own_id)
                        .for_each(|(_, stream)| {
                            // Write each sub-party's data to its own stream
                            let mut stream = stream.as_ref().unwrap();
                            write_data(&mut stream, channel_id, &bytes_out);
                        });
                } else {
                    let mut stream = streams[0].as_ref().unwrap();
                    write_data(&mut stream, channel_id, &bytes_out);
                }
            },

            Data::Multi(bytes_out) => {
                // Iterate over the sub-parties
                streams
                    .par_iter()
                    .enumerate()
                    .filter(|p| p.0 != own_id)
                    .for_each(|(id, stream)| {
                        // Write each sub-party's data to its own stream
                        let mut stream = stream.as_ref().unwrap();
                        write_data(&mut stream, channel_id, &bytes_out[id]);
                    });
            },

            Data::Shutdown => {
                if am_master {
                    streams
                        .par_iter()
                        .enumerate()
                        .filter(|p| p.0 != own_id)
                        .for_each(|(_, stream)| {
                            stream
                                .as_ref()
                                .unwrap()
                                .shutdown(std::net::Shutdown::Both)
                                .unwrap();
                        })
                } else {
                    streams[0]
                        .as_ref()
                        .unwrap()
                        .shutdown(std::net::Shutdown::Both)
                        .unwrap();
                }
                return;
            },
        }
    }
}

fn recv_thread(
    own_id: usize,
    mut poller: Poll,
    streams: &'static [Option<TcpStream>],
    channels: Vec<Sender<Data>>,
) {
    let am_master = own_id == 0;
    let mut tmp_buffer = vec![0u8; 1024];
    let mut events = Events::with_capacity(128);

    if am_master {
        // Keep a queue of the messages received from each peer
        let mut peer_buffers = vec![VecDeque::<u8>::new(); streams.len()];
        let mut channel_messages =
            vec![vec![VecDeque::<Vec<u8>>::new(); streams.len()]; MAX_NUM_CHANNELS];
        let mut num_peers_ready = vec![0usize; MAX_NUM_CHANNELS];
        let mut peer_closed = vec![false; streams.len()];
        loop {
            poller.poll(&mut events, None).unwrap();
            for event in &events {
                let peer_id = event.token().0;
                if peer_closed[peer_id] {
                    continue;
                }

                let mut stream = streams[peer_id].as_ref().unwrap();
                match read_to_buffer(&mut stream, &mut peer_buffers[peer_id], &mut tmp_buffer) {
                    Ok(false) => peer_closed[peer_id] = true,
                    Ok(true) => {},
                    Err(e) => {
                        println!("Read error: {e} from client {peer_id}");
                        peer_closed[peer_id] = true;
                    },
                };
                parse_buffer(&mut peer_buffers[peer_id], |channel_id, data| {
                    if channel_messages[channel_id][peer_id].is_empty() {
                        num_peers_ready[channel_id] += 1;
                    }
                    channel_messages[channel_id][peer_id].push_back(data);

                    // Everyone else have messages ready
                    if num_peers_ready[channel_id] == streams.len() - 1 {
                        let mut out_messages = vec![vec![]; streams.len()];
                        for (i, queue) in channel_messages[channel_id].iter_mut().enumerate() {
                            if i != own_id {
                                out_messages[i] = queue.pop_front().unwrap();
                                if queue.is_empty() {
                                    num_peers_ready[channel_id] -= 1;
                                }
                            }
                        }
                        channels[channel_id]
                            .send(Data::Multi(out_messages))
                            .unwrap();
                    }
                });
            }
        }
    } else {
        // Simply read from the stream
        let mut buffer = VecDeque::new();
        loop {
            poller.poll(&mut events, None).unwrap();
            if !events.is_empty() {
                let mut stream = streams[0].as_ref().unwrap();
                let mut should_return = false;
                match read_to_buffer(&mut stream, &mut buffer, &mut tmp_buffer) {
                    Ok(false) => should_return = true,
                    Ok(true) => {},
                    Err(e) => {
                        println!("Read error: {e}, recv thread now exiting");
                        should_return = true;
                    },
                };
                parse_buffer(&mut buffer, |channel_id, data| {
                    channels[channel_id].send(Data::Single(data)).unwrap()
                });
                if should_return {
                    return;
                }
            }
        }
    }
}

impl Connections {
    /// Given a path and the `id` of oneself, initialize the structure
    fn init_from_path(&mut self, path: &str, id: usize) {
        let f = BufReader::new(File::open(path).expect("host configuration path"));
        let mut peer_id = 0;
        for line in f.lines() {
            let line = line.unwrap();
            let trimmed = line.trim();
            if trimmed.len() > 0 {
                let addr: SocketAddr = trimmed
                    .parse()
                    .unwrap_or_else(|e| panic!("bad socket address: {}:\n{}", trimmed, e));
                let peer = Peer { id: peer_id, addr };
                self.peers.push(peer);
                peer_id += 1;
            }
        }
        assert!(id < self.peers.len());
        self.id = id;
    }

    fn connect_to_all(&mut self) {
        let timer = start_timer!(|| "Connecting");
        let n = self.peers.len();

        let mut streams = Box::new(Vec::new());
        streams.resize_with(n, Default::default);

        let poller = Poll::new().unwrap();
        if self.am_master() {
            for to_id in 1..n {
                debug!("to {}", to_id);
                let to_addr = self.peers[to_id].addr;
                debug!("Contacting {}", to_id);
                let mut stream = {
                    let std_stream = loop {
                        let mut ms_waited = 0;
                        match std::net::TcpStream::connect(to_addr) {
                            Ok(s) => break s,
                            Err(e) => match e.kind() {
                                std::io::ErrorKind::ConnectionRefused
                                | std::io::ErrorKind::ConnectionReset => {
                                    ms_waited += 10;
                                    std::thread::sleep(std::time::Duration::from_millis(10));
                                    if ms_waited % 3_000 == 0 {
                                        debug!("Still waiting");
                                    } else if ms_waited > 30_000 {
                                        panic!("Could not find peer in 30s");
                                    }
                                },
                                _ => {
                                    panic!("Error during FieldChannel::new: {}", e);
                                },
                            },
                        }
                    };
                    std_stream.set_nonblocking(true).unwrap();
                    std_stream
                        .set_linger(Some(std::time::Duration::from_secs(3)))
                        .unwrap();
                    TcpStream::from_std(std_stream)
                };
                stream.set_nodelay(true).unwrap();
                poller
                    .registry()
                    .register(&mut stream, Token(to_id), Interest::READABLE)
                    .unwrap();
                streams[to_id] = Some(stream);
            }
        } else {
            debug!("Awaiting 0");
            let listener = std::net::TcpListener::bind(self.peers[self.id].addr).unwrap();
            let (stream, _addr) = listener.accept().unwrap();
            stream.set_nonblocking(true).unwrap();
            stream
                .set_linger(Some(std::time::Duration::from_secs(3)))
                .unwrap();
            let mut stream = TcpStream::from_std(stream);
            stream.set_nodelay(true).unwrap();
            poller
                .registry()
                .register(&mut stream, Token(0), Interest::READABLE)
                .unwrap();
            streams[0] = Some(stream);
        }

        // Initialize channels
        let (send_send, send_recv) = crossbeam_channel::unbounded::<(usize, Data)>();
        self.send_channel = Some(send_send);

        let (recv_send, recv_recv): (Vec<_>, Vec<_>) = (0..MAX_NUM_CHANNELS)
            .map(|_| crossbeam_channel::unbounded::<Data>())
            .unzip();
        self.recv_channels = recv_recv;

        let streams_ref: &_ = Box::leak(streams);
        let own_id = self.id;
        self.send_join_handle = Some(thread::spawn(move || {
            send_thread(own_id, streams_ref, send_recv)
        }));
        self.recv_join_handle = Some(thread::spawn(move || {
            recv_thread(own_id, poller, streams_ref, recv_send)
        }));

        // Do a round with the master, to be sure everyone is ready
        let from_all = self.send_to_master(vec![self.id as u8]);
        self.recv_from_master(from_all);
        println!("deNetwork ready!");
        end_timer!(timer);
    }
    fn am_master(&self) -> bool {
        self.id == 0
    }
    fn broadcast(&self, _bytes_out: &[u8]) -> Vec<Vec<u8>> {
        unimplemented!("No longer supported");
    }
    fn send_to_master(&self, bytes_out: Vec<u8>) -> Option<Vec<Vec<u8>>> {
        let timer = start_timer!(|| format!("To master {}", bytes_out.len()));
        let channel_id = CHANNEL_ID.get();

        let r = if self.am_master() {
            let (bytes_recv, data) = match self.recv_channels[channel_id].recv().unwrap() {
                Data::Multi(mut data) => {
                    let bytes_recv = data.iter().map(|subdata| 9 + subdata.len()).sum::<usize>();
                    data[self.id] = bytes_out;
                    (bytes_recv, data)
                },
                _ => panic!("Unexpected single response"),
            };
            {
                let mut stats = STATS.lock().unwrap();
                stats.to_master += 1;
                stats.bytes_recv += bytes_recv;
            }
            Some(data)
        } else {
            {
                let mut stats = STATS.lock().unwrap();
                stats.to_master += 1;
                stats.bytes_sent += bytes_out.len() + 9;
            }
            self.send_channel
                .as_ref()
                .unwrap()
                .send((channel_id, Data::Single(bytes_out)))
                .unwrap();
            None
        };

        end_timer!(timer);
        r
    }

    fn recv_from_master(&self, bytes_out: Option<Vec<Vec<u8>>>) -> Vec<u8> {
        let timer = start_timer!(|| format!("From master"));

        let channel_id = CHANNEL_ID.get();

        let r = if self.am_master() {
            let data = bytes_out.unwrap();
            let bytes_sent = data.iter().map(|subdata| 9 + subdata.len()).sum::<usize>();
            {
                let mut stats = STATS.lock().unwrap();
                stats.from_master += 1;
                stats.bytes_sent += bytes_sent;
            }
            let own_data = data[self.id].clone();
            self.send_channel
                .as_ref()
                .unwrap()
                .send((channel_id, Data::Multi(data)))
                .unwrap();
            own_data
        } else {
            let data = match self.recv_channels[channel_id].recv().unwrap() {
                Data::Single(data) => data,
                _ => panic!("Unexpected multi response"),
            };
            {
                let mut stats = STATS.lock().unwrap();
                stats.from_master += 1;
                stats.bytes_recv += data.len() + 9;
            }
            data
        };

        end_timer!(timer);
        r
    }

    fn recv_from_master_uniform(&self, bytes_out: Option<Vec<u8>>) -> Vec<u8> {
        let timer = start_timer!(|| format!("From master"));

        let channel_id = CHANNEL_ID.get();

        let r = if self.am_master() {
            let data = bytes_out.unwrap();
            let bytes_sent = (data.len() + 9) * (self.peers.len() - 1);
            {
                let mut stats = STATS.lock().unwrap();
                stats.from_master += 1;
                stats.bytes_sent += bytes_sent;
            }
            let own_data = data.clone();
            self.send_channel
                .as_ref()
                .unwrap()
                .send((channel_id, Data::Single(data)))
                .unwrap();
            own_data
        } else {
            let data = match self.recv_channels[channel_id].recv().unwrap() {
                Data::Single(data) => data,
                _ => panic!("Unexpected multi response"),
            };
            {
                let mut stats = STATS.lock().unwrap();
                stats.from_master += 1;
                stats.bytes_recv += data.len() + 9;
            }
            data
        };

        end_timer!(timer);
        r
    }

    fn uninit(&mut self) {
        self.send_channel
            .as_ref()
            .unwrap()
            .send((0, Data::Shutdown))
            .unwrap();
        self.send_join_handle.take().unwrap().join().unwrap();
        // Do not join the receiver thread because I did not bother to send it a
        // signal that it should exit
        // self.recv_join_handle.take().unwrap().join().unwrap();
    }
}

pub struct DeMultiNet;

impl DeNet for DeMultiNet {
    #[inline]
    fn party_id() -> usize {
        get_ch!().id
    }

    #[inline]
    fn n_parties() -> usize {
        get_ch!().peers.len()
    }

    #[inline]
    fn init_from_file(path: &str, party_id: usize) {
        let mut ch = get_ch_mut!();
        ch.init_from_path(path, party_id);
        ch.connect_to_all();
    }

    #[inline]
    fn is_init() -> bool {
        Self::n_parties() > 0
    }

    #[inline]
    fn deinit() {
        thread::sleep(std::time::Duration::from_secs(3));
        get_ch_mut!().uninit()
    }

    #[inline]
    fn reset_stats() {
        let mut stats = STATS.lock().unwrap();
        *stats = Stats::default();
    }

    #[inline]
    fn stats() -> crate::Stats {
        STATS.lock().unwrap().clone()
    }

    #[inline]
    fn broadcast_bytes(bytes: &[u8]) -> Vec<Vec<u8>> {
        get_ch!().broadcast(bytes)
    }

    #[inline]
    fn send_bytes_to_master(bytes: Vec<u8>) -> Option<Vec<Vec<u8>>> {
        get_ch!().send_to_master(bytes)
    }

    #[inline]
    fn recv_bytes_from_master(bytes: Option<Vec<Vec<u8>>>) -> Vec<u8> {
        get_ch!().recv_from_master(bytes)
    }

    #[inline]
    fn recv_bytes_from_master_uniform(bytes: Option<Vec<u8>>) -> Vec<u8> {
        get_ch!().recv_from_master_uniform(bytes)
    }

    #[inline]
    fn set_channel_id(channel_id: usize) {
        CHANNEL_ID.set(channel_id);
    }
}

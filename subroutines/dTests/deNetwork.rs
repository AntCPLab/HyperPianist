mod common;

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

fn test_send() {
    let val = Net::send_to_master(&Net::party_id());
    if Net::am_master() {
        let val = val.unwrap();
        assert_eq!(val.len(), Net::n_parties());
        for i in 0..val.len() {
            assert_eq!(val[i], i);
        }
    } else {
        assert!(val == None);
    }
}

fn test_recv() {
    let val = Net::recv_from_master_uniform(if Net::am_master() {
        Some(123u32)
    } else {
        None
    });
    assert_eq!(val, 123u32);

    let mut data = vec![];
    for i in 0..Net::n_parties() {
        data.push(i as u32);
    }
    let val = Net::recv_from_master(if Net::am_master() {
        Some(data)
    } else {
        None
    });
    assert_eq!(val, Net::party_id() as u32);
}

fn main() {
    common::network_run(|| {
        test_send();
        test_recv();
        println!("Test OK");
    });
}

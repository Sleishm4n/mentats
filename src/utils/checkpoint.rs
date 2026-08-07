use crate::nn::network::Network;
use std::fs::create_dir_all;
use std::path::Path;

pub fn save_network(network: &Network, path: &str) {
    if let Some(parent) = Path::new(path).parent() {
        create_dir_all(parent).unwrap_or_else(|e| {
            panic!("Failed to create checkpoint directory {:?}: {}", parent, e)
        });
    }
    network
        .save(path)
        .unwrap_or_else(|e| panic!("Failed to save netwotrk to {:?}: {}", path, e));
}

pub fn load_network_or_panic(path: &str, context: &str) -> Network {
    Network::load(path)
        .unwrap_or_else(|e| panic!("Failed to load network from {} ({}). {}", path, e, context))
}

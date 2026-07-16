use std::fs::File;
use std::io::{self, BufReader, BufWriter};

use crate::{nn::Layer, optimiser::Optimiser, tensor::Tensor};
use crate::utils::model_io::{load_layer, read_u32, write_u32};

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Network {
        Network { layers }
    }

    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut current = input.clone();
        for layer in &mut self.layers {
            current = layer.forward_pass(&current);
        }
        current
    }

    pub fn backward(&mut self, input: &Tensor) -> Tensor {
        let mut current = input.clone();
        for layer in self.layers.iter_mut().rev() {
            current = layer.backward_pass(&current);
        }
        current
    }

    pub fn update(&mut self, optimiser: &mut dyn Optimiser) {
        let mut all_params: Vec<Tensor> = vec![];
        let mut all_grads: Vec<Tensor> = vec![];
        let mut counts: Vec<usize> = vec![];

        for layer in &mut self.layers {
            let params = layer.get_params();
            let grads = layer.get_grads();
            counts.push(params.len());
            all_params.extend(params);
            all_grads.extend(grads);
        }

        optimiser.step(&mut all_params, &all_grads);

        // redistribute back
        let mut idx = 0;
        for (layer, count) in self.layers.iter_mut().zip(counts.iter()) {
            layer.set_params(all_params[idx..idx + count].to_vec());
            idx += count;
        }
    }

    /// Format: [layer_count: u32][for each layer, in order: layer.save()]
    /// The count has to be written up front - without it `load` has no
    /// way to know when to stop reading from the flat byte stream.
    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        write_u32(&mut writer, self.layers.len() as u32)?;
        for layer in &self.layers {
            layer.save(&mut writer)?;
        }
        Ok(())
    }

    pub fn load(path: &str) -> io::Result<Network> {
        let mut reader = BufReader::new(File::open(path)?);
        let count = read_u32(&mut reader)? as usize;
        let mut layers = Vec::with_capacity(count);
        for _ in 0..count {
            layers.push(load_layer(&mut reader)?);
        }
        Ok(Network { layers })
    }
}
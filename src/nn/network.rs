//! The [`Network`] container: an ordered stack of [`Layer`]s plus the
//! forward / backward / update loop and model serialisation
use std::fs::File;
use std::io::{self, BufReader, BufWriter};

use crate::utils::model_io::{load_layer, read_u32, write_u32};
use crate::{nn::Layer, optimiser::Optimiser, tensor::Tensor};

/// An ordered stack of layers
///
/// A training step is always the same three calls: [`Network::forward`] to get
/// predictions, [`Network::backward`] with the loss gradient then
/// [`Network::update`] to let the optimiser apply the accumulated gradients
///
/// Build on with [`Network::new`]
pub struct Network {
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    /// Creates a network from layers, applied in the order given
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Network {
        Network { layers }
    }

    /// Runs `input` through every layer front to back and returns the
    /// final output
    ///
    /// Each layer caches what it needs for the backward pass, so a
    /// [`Network::backward`] call must always be preceded by a forward call
    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut current = input.clone();
        for layer in &mut self.layers {
            current = layer.forward_pass(&current);
        }
        current
    }

    /// Backpropagates the loss gradient through every layer back to front.
    ///
    /// `input` is the gradient of the loss with respect to the network's
    /// *output* (for example `d_mse(&prediction, &target)`). The returned
    /// tensor is the gradient with respect to the network's input, which is
    /// usually discarded, but is what chains an encoder to a decoder in the
    /// VAE examples. Parameter gradients are left on the individual layers for
    /// [`Network::update`] to collect.
    pub fn backward(&mut self, input: &Tensor) -> Tensor {
        let mut current = input.clone();
        for layer in self.layers.iter_mut().rev() {
            current = layer.backward_pass(&current);
        }
        current
    }

    /// Applies one optimiser step to every parameter in the network.
    ///
    /// Parameters and gradients from all layers are gathered into one flat
    /// list so the optimiser sees a single contiguous parameter set (which
    /// keeps per-parameter state such as Adam's moments stably indexed), then
    /// the updated values are redistributed back to their layers.
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

    /// Serialises the whole network to `path`.
    ///
    /// Format: `[layer_count: u32][for each layer, in order: layer.save()]`
    /// The count has to be written up front - without it `load` has no
    /// way to know when to stop reading from the flat byte stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or written to.
    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        write_u32(&mut writer, self.layers.len() as u32)?;
        for layer in &self.layers {
            layer.save(&mut writer)?;
        }
        Ok(())
    }

    /// Reconstructs a network previously written by [`Network::save`].
    ///
    /// Each layer is rebuilt from its type tag, so the architecture does not
    /// need to be known ahead of time.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened, the stream ends early,
    /// or an unknown layer tag is encountered.
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

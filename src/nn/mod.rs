//! Neural network components and abstractions.
//!
//! This module provides the core building blocks for constructing and
//! training neural networks, including layers, activation functions,
//! parameter initialisation, tensor reshaping, and sampling.
//! ## Modules
//!
//! - [`activation`] - Activation functions and activation layers.
//! - [`flatten`] - A layer for flattening tensors.
//! - [`init`] - Parameter initialisation methods.
//! - [`linear`] - Linear (fully connected) layers.
//! - [`network`] - Neural network construction and management.
//! - [`reshape`] - Layers for reshaping tensors.
//! - [`sampling`] - Sampling layers used by probabilistic models such as VAEs.
//! - [`softmax`] - Softmax activation and related functionality.
use crate::tensor::Tensor;
use std::io::{self, Write};

pub mod activation;
pub mod flatten;
pub mod init;
pub mod linear;
pub mod network;
pub mod reshape;
pub mod sampling;
pub mod softmax;

/// The common interface every layer implements
///
/// A [`crate::nn::network::Network`] is just a `Vec<Box<dyn Layer>>`, so a
/// layer only has to know how to move a tensor forward, push a gradient back,
/// expose its parameters to an optimiser and serialise itself
///
/// Layers are stateful: `forward_pass` caches whatever `backward_pass` needs,
/// so the two must always be called in order and in pairs
pub trait Layer {
    /// Runs the layer forward and caches any state the backward pass needs
    fn forward_pass(&mut self, input: &Tensor) -> Tensor;

    /// Propagates `d_output` (the gradient of the loss w.r.t. this layer's
    /// output) back to the layer's input, storing parameter gradients on the
    /// way
    fn backward_pass(&mut self, d_output: &Tensor) -> Tensor;

    /// Overwrites the layer's parameters, in the same order as
    /// [`Layer::get_params`]. Used by optimiser to apply update
    fn set_params(&mut self, params: Vec<Tensor>);

    ///Returns the layer's trainable parameters. Parameterless layers
    /// return an empty vector
    fn get_params(&self) -> Vec<Tensor>;

    /// Returns the gradients accumulated by the last [`Layer::backward_pass`],
    /// matching order of [`Layer::get_params`]
    fn get_grads(&self) -> Vec<Tensor>;

    /// Writes the layer's type tag and parameters to `writer`
    ///
    /// The tag constants live in [`crate::utils::model_io`] and tell
    /// [`crate::nn::network::Network::load`] which layer to reconstruct
    fn save(&self, writer: &mut dyn Write) -> io::Result<()>;
}

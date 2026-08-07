//! Optimisation
//!
//! Changes weights and learning rate during training of the model
//!
//! - [`adam`] - the main optimiser, an implementation of Adam optimiser from original paper

use crate::tensor::Tensor;

pub mod adam;

/// Defines the interface for paramter optimisation algs
///
/// # Arguments
///
/// * `params` - Mutable references to the model parameters that will be updated.
/// * `grads` - Gradients corresponding to each parameter in `params`.
pub trait Optimiser {
    fn step(&mut self, params: &mut Vec<Tensor>, grads: &[Tensor]);
}

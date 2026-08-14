//! Loss functions and their analytic derivatives
//!
//! Each loss comes in pair: `f(output, target) -> f32` for the scalar values
//! you log during training, and `d_f(output, target) -> Tensor` for the
//! gradient you hand to [`crate::nn::network::Network::backward`]
//!
//! - [`mse`] - mean squared error, for regression and VAE reconstruction
//! - [`cross_entropy`] - softmax and binary cross-entropy, for classification
//! - [`kl_divergence`] - the VAE latent regularisation term, with free bits

pub mod cross_entropy;
pub mod kl_divergence;
pub mod mse;

//! A neural network library built from scratch in Rust, with no external
//! machine learning dependencies. Tensors, layers, losses, and optimisers
//! are all implemented from first principles, the goal is to understand
//! the underlying math, not to wrap an existing framework.
//!
//! # Example
//!
//! A tiny feed-forward network, built and run one forward pass:
//!
//! ```
//! use mentats::{Network, Tensor};
//!
//! let mut network = Network::builder()
//!     .input(2)
//!     .linear(2)
//!     .sigmoid()
//!     .linear(1)
//!     .build();
//!
//! let input = Tensor::from_vec(vec![2, 1], vec![1.0, 0.0]);
//! let output = network.forward(&input);
//! ```
//!
//! See the `examples/` directory in the repository for full training loops
//! (XOR, MNIST classification, VAE, conditional VAE,  CNN-CVAE).
//!
//! # Modules
//!
//! - [`tensor`] - the core `Tensor` type: matmul, permute, elementwise maps, broadcasting.
//! - [`nn`] - layers, the [`NetworkBuilder`] and the [`nn::network::Network`] container.
//! - [`loss`] - MSE, cross-entropy, and KL divergence.
//! - [`optimiser`] - the Adam optimiser.
//! - [`data`] - dataset loaders (MNIST).
//! - [`utils`] - checkpointing, VAE sampling helpers, gradient checking.

pub mod data;
pub mod loss;
pub mod matrix;
pub mod nn;
pub mod optimiser;
pub mod tensor;
pub mod utils;

pub use tensor::core::Tensor;

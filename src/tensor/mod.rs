//! Tensor construction, indexing, and math operations.
//!
//! - [`core`] - the [`Tensor`] struct itself: shape, strides, `get`/`set`.
//! - [`init`] - random initialization (e.g. `rand_range`).
//! - [`ops`] - elementwise and matrix ops: `add`, `matmul`, `permute`, etc.
//! - [`batch_ops`] - batched variants: `matmul_batched`, `sum_batch`.

pub mod batch_ops;
pub mod core;
pub mod init;
pub mod ops;

pub use core::Tensor;

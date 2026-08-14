//! Mean squared error loss
use crate::tensor::Tensor;

/// Mean sqaured error between `output` and `target`
///
/// Averaged over every element, so the value does not scale with batch size
/// or feature count. Used as the reconstruction term in the VAE examples
///
/// # Panics
///
/// Panics if `target` has fewer elements than `output`
pub fn mse(output: &Tensor, target: &Tensor) -> f32 {
    let mut total: f32 = 0.0;

    for (index, element) in output.data.iter().enumerate() {
        total += (element - target.data[index]).powi(2);
    }

    total / (output.data.len()) as f32
}

/// Gradient of [`mse`] with respect to `output`: `2 * (output - target) / n`
///
/// Feed the result straight into [`crate::nn::network::Network::backward`]
///
/// # Panics
///
/// Panics if the shapes differ
pub fn d_mse(output: &Tensor, target: &Tensor) -> Tensor {
    output.zip_map(target, |a, b| 2.0 * (a - b) / output.data.len() as f32)
}

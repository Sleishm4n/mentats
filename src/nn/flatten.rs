//! Layer that flattens a multi-dimensional tensor into a column vector
use std::io::{self, Read, Write};

use crate::{
    nn::Layer,
    tensor::Tensor,
    utils::model_io::{write_u8, TAG_FLATTEN},
};

/// Collapses an input of any shape into a `[elements, 1]` column vector, and
/// restores the original shape on the backward pass
///
/// Used to bridge image-shaped data (`[28, 28]`) into the column vectors that
/// [`crate::nn::linear::LinearLayer`] expects. It has no parameters, the input
/// shape it records is only forward pass cache
pub struct FlattenLayer {
    input_shape: Option<Vec<usize>>,
}

impl Default for FlattenLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl FlattenLayer {
    /// Creates a new flatten layer
    pub fn new() -> Self {
        Self { input_shape: None }
    }

    /// Reads a layer back from `reader`, assuming that the [`TAG_FLATTEN`] byte
    /// has already been consumed
    ///
    /// Nothing is actually read: `input_shape` is just forward pass cache, not
    /// durable state
    ///
    /// # Errors
    ///
    /// Never returns an error, the signature matches the other layer loaders
    pub fn load(_reader: &mut dyn Read) -> io::Result<FlattenLayer> {
        Ok(FlattenLayer::new())
    }
}

impl Layer for FlattenLayer {
    fn forward_pass(&mut self, input: &Tensor) -> Tensor {
        self.input_shape = Some(input.shape.clone());
        Tensor::from_vec(vec![input.data.len(), 1], input.data.clone())
    }

    fn backward_pass(&mut self, d_output: &Tensor) -> Tensor {
        let input_shape: Vec<usize> = self
            .input_shape
            .as_ref()
            .expect("forward_pass must be called before backward_pass")
            .clone();

        let expected_elements: usize = input_shape.iter().product();
        assert_eq!(
            d_output.data.len(),
            expected_elements,
            "flatten backwards gradient size mismatch"
        );

        Tensor::from_vec(input_shape, d_output.data.clone())
    }

    fn set_params(&mut self, _params: Vec<Tensor>) {}

    fn get_params(&self) -> Vec<Tensor> {
        Vec::new()
    }

    fn get_grads(&self) -> Vec<Tensor> {
        Vec::new()
    }

    fn save(&self, writer: &mut dyn Write) -> io::Result<()> {
        write_u8(writer, TAG_FLATTEN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten_forward_to_column_vecor() {
        let mut layer = FlattenLayer::new();
        let input = Tensor::from_vec(vec![2, 2, 3], (0..12).map(|x| x as f32).collect());

        let output = layer.forward_pass(&input);

        assert_eq!(output.shape, vec![12, 1]);
        assert_eq!(output.data, input.data);
    }

    #[test]
    fn test_flatten_backward_restores_og_shape() {
        let mut layer = FlattenLayer::new();
        let input = Tensor::from_vec(vec![2, 2, 3], (0..12).map(|x| x as f32).collect());
        let _ = layer.forward_pass(&input);

        let grad = Tensor::from_vec(vec![12, 1], (0..12).rev().map(|x| x as f32).collect());
        let back = layer.backward_pass(&grad);

        assert_eq!(back.shape, vec![2, 2, 3]);
        assert_eq!(back.data, grad.data);
    }
}

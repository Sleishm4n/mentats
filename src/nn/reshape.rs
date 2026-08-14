//! Layer that reshapes a tensor to a fixed target shape.

use std::io::{self, Read, Write};

use crate::{
    nn::Layer,
    tensor::Tensor,
    utils::model_io::{read_shape, write_shape, write_u8, TAG_RESHAPE},
};

/// Reinterprets a tensor as `output_shape`, keeping the element count fixed.
///
/// The counterpart to [`crate::nn::flatten::FlattenLayer`], typically used at
/// the end of a decoder to turn a flat `[784, 1]` vector back into a `[28, 28]`
/// image. The backward pass reverses the reshape.
pub struct ReshapeLayer {
    /// The shape produced by the forward pass.
    pub output_shape: Vec<usize>,
    /// Input shape cached by the last forward pass, restored on backward.
    pub input_shape: Option<Vec<usize>>,
}

impl ReshapeLayer {
    /// Creates a layer that reshapes its input to `output_shape`.
    ///
    /// # Panics
    ///
    /// Panics if `output_shape` is empty.
    pub fn new(output_shape: Vec<usize>) -> Self {
        assert!(!output_shape.is_empty(), "output_shape cannot be empty");
        Self {
            output_shape,
            input_shape: None,
        }
    }

    /// Reads a layer back from `reader`, assuming the [`TAG_RESHAPE`] byte has
    /// already been consumed.
    ///
    /// `output_shape` is the one bit of config this layer needs to
    /// reconstruct; `input_shape` is just forward-pass cache.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream ends early or the shape is malformed.
    pub fn load(reader: &mut dyn Read) -> io::Result<ReshapeLayer> {
        let output_shape = read_shape(reader)?;
        Ok(ReshapeLayer::new(output_shape))
    }
}

impl Layer for ReshapeLayer {
    fn forward_pass(&mut self, input: &Tensor) -> Tensor {
        let input_elements: usize = input.shape.iter().product();
        let output_elements: usize = self.output_shape.iter().product();
        assert_eq!(
            input_elements, output_elements,
            "reshape size mismatch between input and output"
        );

        self.input_shape = Some(input.shape.clone());
        Tensor::from_vec(self.output_shape.clone(), input.data.clone())
    }

    fn backward_pass(&mut self, d_output: &Tensor) -> Tensor {
        let input_shape = self
            .input_shape
            .as_ref()
            .expect("forward_pass must be called before backward_pass")
            .clone();

        let expected_elements: usize = input_shape.iter().product();
        assert_eq!(
            d_output.data.len(),
            expected_elements,
            "backward reshape gradient size mismatch"
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
        write_u8(writer, TAG_RESHAPE)?;
        write_shape(writer, &self.output_shape)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reshape_forward_changes_shape_only() {
        let mut layer = ReshapeLayer::new(vec![3, 2]);
        let input = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let output = layer.forward_pass(&input);

        assert_eq!(output.shape, vec![3, 2]);
        assert_eq!(output.data, input.data);
    }

    #[test]
    fn test_reshape_backward_restores_input_shape() {
        let mut layer = ReshapeLayer::new(vec![3, 2]);
        let input = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let _ = layer.forward_pass(&input);

        let grad = Tensor::from_vec(vec![3, 2], vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let back = layer.backward_pass(&grad);

        assert_eq!(back.shape, vec![2, 3]);
        assert_eq!(back.data, grad.data);
    }

    #[test]
    #[should_panic(expected = "reshape size mismatch")]
    fn test_reshape_forward_panics_on_size_mismatch() {
        let mut layer = ReshapeLayer::new(vec![4, 2]);
        let input = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let _ = layer.forward_pass(&input);
    }
}

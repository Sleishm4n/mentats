use std::io::{self, Read, Write};

use crate::{
    nn::Layer,
    tensor::Tensor,
    utils::model_io::{write_u8, TAG_FLATTEN},
};

pub struct FlattenLayer {
    input_shape: Option<Vec<usize>>,
}

impl Default for FlattenLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl FlattenLayer {
    pub fn new() -> Self {
        Self { input_shape: None }
    }

    /// input_shape is just forward-pass cache, not durable state.
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
        vec![]
    }

    fn get_grads(&self) -> Vec<Tensor> {
        vec![]
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

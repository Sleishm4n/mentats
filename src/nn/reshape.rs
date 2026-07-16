use crate::{nn::Layer, tensor::Tensor};

pub struct ReshapeLayer {
    pub output_shape: Vec<usize>,
    pub input_shape: Option<Vec<usize>>,
}

impl ReshapeLayer {
    pub fn new(output_shape: Vec<usize>) -> Self {
        assert!(!output_shape.is_empty(), "output_shape cannot be empty");
        Self {
            output_shape,
            input_shape: None,
        }
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
        vec![]
    }

    fn get_grads(&self) -> Vec<Tensor> {
        vec![]
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

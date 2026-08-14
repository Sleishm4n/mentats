//! Softmax layer, converting scores into a probability distribution.

use std::io::{self, Read, Write};

use crate::{
    nn::Layer,
    tensor::Tensor,
    utils::model_io::{write_u8, TAG_SOFTMAX},
};

/// Normalises its input into a probability distribution that sums to 1.
///
/// Computed with the max-subtraction trick for numerical stability. Intended
/// to sit at the end of a classifier, paired with
/// [`crate::loss::cross_entropy`]. It has no trainable parameters.
pub struct SoftmaxLayer {
    /// Input cached by the last forward pass.
    pub input: Option<Tensor>,
}

impl Default for SoftmaxLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl SoftmaxLayer {
    /// Creates a new softmax layer.
    pub fn new() -> Self {
        SoftmaxLayer { input: None }
    }

    /// Reads a layer back from `reader`, assuming the [`TAG_SOFTMAX`] byte has
    /// already been consumed.
    ///
    /// Nothing is actually read, the layer carries no durable state.
    ///
    /// # Errors
    ///
    /// Never returns an error, the signature matches the other layer loaders.
    pub fn load(_reader: &mut dyn Read) -> io::Result<SoftmaxLayer> {
        Ok(SoftmaxLayer::new())
    }
}

impl Layer for SoftmaxLayer {
    fn forward_pass(&mut self, input: &Tensor) -> Tensor {
        let max = input.tensor_max();
        let exps = input.map(|x| (x - max).exp());
        let sum = exps.data.iter().sum::<f32>();
        let output = exps.map(|x| x / sum);
        self.input = Some(output.clone());
        output
    }

    fn backward_pass(&mut self, d_output: &Tensor) -> Tensor {
        let s = self.input.as_ref().unwrap();
        let dot: f32 = d_output.zip_map(s, |a, b| a * b).data.iter().sum();
        let shifted = d_output.map(|x| x - dot);
        s.zip_map(&shifted, |a, b| a * b)
    }

    fn get_params(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn get_grads(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn set_params(&mut self, _params: Vec<Tensor>) {}

    fn save(&self, writer: &mut dyn Write) -> io::Result<()> {
        write_u8(writer, TAG_SOFTMAX)
    }
}

use crate::tensor::Tensor;
use std::io::{self, Write};

pub mod activation;
pub mod flatten;
pub mod init;
pub mod linear;
pub mod network;
pub mod reshape;
pub mod sampling;
pub mod softmax;

pub trait Layer {
    fn forward_pass(&mut self, input: &Tensor) -> Tensor;
    fn backward_pass(&mut self, d_output: &Tensor) -> Tensor;
    fn set_params(&mut self, params: Vec<Tensor>);
    fn get_params(&self) -> Vec<Tensor>;
    fn get_grads(&self) -> Vec<Tensor>;

    fn save(&self, writer: &mut dyn Write) -> io::Result<()>;
}

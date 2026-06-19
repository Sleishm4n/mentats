use crate::tensor::Tensor;

pub mod activation;
pub mod linear;
pub mod network;
pub mod init;
pub mod softmax;

pub trait Layer {
    fn forward_pass(&mut self, input: &Tensor) -> Tensor;
    fn backward_pass(&mut self, d_output: &Tensor) -> Tensor;
    fn set_params(&mut self, params: Vec<Tensor>);
    fn get_params(&self) -> Vec<Tensor>;
    fn get_grads(&self) -> Vec<Tensor>; 
}
use crate::tensor::Tensor;

pub mod adam;

pub trait Optimiser {
    fn step(&mut self, params: &mut Vec<Tensor>, grads: &[Tensor]);
}

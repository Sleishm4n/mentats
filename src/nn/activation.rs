use std::io::{self, Read, Write};

use crate::{
    nn::Layer,
    tensor::Tensor,
    utils::model_io::{read_u8, write_u8, TAG_ACTIVATION},
};

pub fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn tanh(x: f32) -> f32 {
    (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())
}

pub fn d_relu(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn d_sigmoid(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub fn d_tanh(x: f32) -> f32 {
    1.0 - tanh(x).powi(2)
}

const ACT_RELU: u8 = 0;
const ACT_SIGMOID: u8 = 1;
const ACT_TANH: u8 = 2;

#[derive(Clone, Copy, PartialEq)]
pub enum ActivationKind {
    Relu,
    Sigmoid,
    Tanh,
}

impl ActivationKind {
    fn id(self) -> u8 {
        match self {
            ActivationKind::Relu => ACT_RELU,
            ActivationKind::Sigmoid => ACT_SIGMOID,
            ActivationKind::Tanh => ACT_TANH,
        }
    }

    fn function(self) -> fn(f32) -> f32 {
        match self {
            ActivationKind::Relu => relu,
            ActivationKind::Sigmoid => sigmoid,
            ActivationKind::Tanh => tanh,
        }
    }

    fn derivative(self) -> fn(f32) -> f32 {
        match self {
            ActivationKind::Relu => d_relu,
            ActivationKind::Sigmoid => d_sigmoid,
            ActivationKind::Tanh => d_tanh,
        }
    }
}

pub struct ActivationLayer {
    pub kind: ActivationKind,
    pub input: Option<Tensor>,
}

impl ActivationLayer {
    pub fn new(kind: ActivationKind) -> Self {
        ActivationLayer { kind, input: None }
    }

    pub fn load(reader: &mut dyn Read) -> io::Result<ActivationLayer> {
        let id = read_u8(reader)?;
        let kind = match id {
            ACT_RELU => ActivationKind::Relu,
            ACT_SIGMOID => ActivationKind::Sigmoid,
            ACT_TANH => ActivationKind::Tanh,
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown activation id: {other}"),
                ))
            }
        };
        Ok(ActivationLayer { kind, input: None })
    }
}

impl Layer for ActivationLayer {
    fn forward_pass(&mut self, input: &Tensor) -> Tensor {
        self.input = Some(input.clone());
        input.map(self.kind.function())
    }

    fn backward_pass(&mut self, d_output: &Tensor) -> Tensor {
        let input = self.input.as_ref().unwrap();
        input
            .map(self.kind.derivative())
            .zip_map(d_output, |d, g| d * g)
    }

    fn get_params(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn get_grads(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn set_params(&mut self, _params: Vec<Tensor>) {}

    fn save(&self, writer: &mut dyn Write) -> io::Result<()> {
        write_u8(writer, TAG_ACTIVATION)?;
        write_u8(writer, self.kind.id())?;
        Ok(())
    }
}

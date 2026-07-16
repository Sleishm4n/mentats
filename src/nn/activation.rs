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

pub struct ActivationLayer {
    pub function: fn(f32) -> f32,
    pub derivative: fn(f32) -> f32,
    pub input: Option<Tensor>,
}

const ACT_RELU: u8 = 0;
const ACT_SIGMOID: u8 = 1;
const ACT_TANH: u8 = 2;

impl ActivationLayer {
    pub fn new(function: fn(f32) -> f32, derivative: fn(f32) -> f32) -> Self {
        ActivationLayer {
            function,
            derivative,
            input: None,
        }
    }

    fn activation_id(&self) -> io::Result<u8> {
        // fn pointers compare by address, so this correctly identifies
        // which named activation was used to construct this layer.
        if self.function as usize == relu as *const () as usize {
            Ok(ACT_RELU)
        } else if self.function as usize == sigmoid as *const () as usize {
            Ok(ACT_SIGMOID)
        } else if self.function as usize == tanh as *const () as usize {
            Ok(ACT_TANH)
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot save ActivationLayer: unrecognized activation function \
                 (only relu/sigmoid/tanh are supported)",
            ))
        }
    }

    pub fn load(reader: &mut dyn Read) -> io::Result<ActivationLayer> {
        let id = read_u8(reader)?;
        let (function, derivative): (fn(f32) -> f32, fn(f32) -> f32) = match id {
            ACT_RELU => (relu, d_relu),
            ACT_SIGMOID => (sigmoid, d_sigmoid),
            ACT_TANH => (tanh, d_tanh),
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown activation id: {other}"),
                ))
            }
        };
        Ok(ActivationLayer {
            function,
            derivative,
            input: None,
        })
    }
}

impl Layer for ActivationLayer {
    fn forward_pass(&mut self, input: &Tensor) -> Tensor {
        self.input = Some(input.clone());
        input.map(self.function)
    }

    fn backward_pass(&mut self, d_output: &Tensor) -> Tensor {
        let input = self.input.as_ref().unwrap();
        input.map(self.derivative).zip_map(d_output, |d, g| d * g)
    }

    fn get_params(&self) -> Vec<Tensor> {
        vec![]
    }
    fn get_grads(&self) -> Vec<Tensor> {
        vec![]
    }
    fn set_params(&mut self, _params: Vec<Tensor>) {}

    fn save(&self, writer: &mut dyn Write) -> io::Result<()> {
        write_u8(writer, TAG_ACTIVATION)?;
        write_u8(writer, self.activation_id()?)?;
        Ok(())
    }
}

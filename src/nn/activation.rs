//! Activation functions, their derivatives and the [`ActivationLayer`] that
//! applies them elementwise
//!
//! The raw `f32 -> f32` functions are public so they can be used directly (
//! in tests, or in hand-made loops), but inside a network they are selected
//! through the [`ActivationKind`], which is what makes the layer serialisable:
//! only a wee id byte is written to disk, not a whole function pointer
use std::io::{self, Read, Write};

use crate::{
    nn::Layer,
    tensor::Tensor,
    utils::model_io::{read_u8, write_u8, TAG_ACTIVATION},
};

pub const LEAKY_RELU_ALPHA: f32 = 0.01;

/// Rectified linear unit: `max(0, x)`
pub fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Leaky Rectified linear unit (LReLU): `max(alpha * x, x)`
///
/// Uses [`LEAKY_RELU_ALPHA`] as `alpha`.
pub fn leaky_relu(x: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        LEAKY_RELU_ALPHA * x
    }
}

/// Logistic sigmoid: `1 / (1 + e^-x)`. squashing to `(0, 1)`
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Hyperbolic tangent, squashing to `(-1, 1)`
pub fn tanh(x: f32) -> f32 {
    (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())
}

/// Derivative of [`relu`]. Undefined at zero, taken here as 0
pub fn d_relu(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

/// Derivative of [`leaky_relu`].
///
/// Returns `1.0` for positive inputs and [`LEAKY_RELU_ALPHA`] for
/// non-positive inputs. At `x = 0`, the derivative is undefined when
/// `LEAKY_RELU_ALPHA != 1`, this implementation uses `LEAKY_RELU_ALPHA`
/// by convention.
pub fn d_leaky_relu(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        LEAKY_RELU_ALPHA
    }
}

/// Derivative of [`sigmoid`]: `s(x) * (1 - s(x))`
pub fn d_sigmoid(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

/// Derivative of [`tanh`]: `1 - tanh(x)^2`
pub fn d_tanh(x: f32) -> f32 {
    1.0 - tanh(x).powi(2)
}

const ACT_RELU: u8 = 0;
const ACT_SIGMOID: u8 = 1;
const ACT_TANH: u8 = 2;
const ACT_LEAKY_RELU: u8 = 3;

/// Which activation an [`ActivationLayer`] applies
///
/// Storing the choice as an enum rather than a pair of function pointers is
/// what lets a layer be saved and reloaded, each variant maps to a stable
/// id byte in the checkpoint format
#[derive(Clone, Copy, PartialEq)]
pub enum ActivationKind {
    /// [`relu`]
    Relu,
    /// [`sigmoid`]
    Sigmoid,
    /// [`tanh`]
    Tanh,
    /// [`leaky_relu`]
    LeakyRelu,
}

impl ActivationKind {
    /// The stable id byte written to checkpoints for this variant
    fn id(self) -> u8 {
        match self {
            ActivationKind::Relu => ACT_RELU,
            ActivationKind::Sigmoid => ACT_SIGMOID,
            ActivationKind::Tanh => ACT_TANH,
            ActivationKind::LeakyRelu => ACT_LEAKY_RELU,
        }
    }

    /// The forward function for this variant
    fn function(self) -> fn(f32) -> f32 {
        match self {
            ActivationKind::Relu => relu,
            ActivationKind::Sigmoid => sigmoid,
            ActivationKind::Tanh => tanh,
            ActivationKind::LeakyRelu => leaky_relu,
        }
    }

    /// The derivative of [`ActivationKind::function`] for this variant
    fn derivative(self) -> fn(f32) -> f32 {
        match self {
            ActivationKind::Relu => d_relu,
            ActivationKind::Sigmoid => d_sigmoid,
            ActivationKind::Tanh => d_tanh,
            ActivationKind::LeakyRelu => d_leaky_relu,
        }
    }
}

/// Applies an [`ActivationKind`] elementwise, with no trainable params
///
/// The pre-activation input is cached on the forward pass, the backward
/// pass needs it to evaluate the derivative at the same point
pub struct ActivationLayer {
    /// The activation function this layer applies
    pub kind: ActivationKind,
    /// Pre-activation input cached by the last forward pass
    pub input: Option<Tensor>,
}

impl ActivationLayer {
    /// Creates a layer applying the given activation
    pub fn new(kind: ActivationKind) -> Self {
        ActivationLayer { kind, input: None }
    }

    /// Reads a layer back from `reader`, assuming the [`TAG_ACTIVATION`] byte
    /// has already been consumed
    ///
    /// # Errors
    ///
    /// Returns an error if the stream ends early or the activation id is not
    /// recognised
    pub fn load(reader: &mut dyn Read) -> io::Result<ActivationLayer> {
        let id = read_u8(reader)?;
        let kind = match id {
            ACT_RELU => ActivationKind::Relu,
            ACT_SIGMOID => ActivationKind::Sigmoid,
            ACT_TANH => ActivationKind::Tanh,
            ACT_LEAKY_RELU => ActivationKind::LeakyRelu,
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

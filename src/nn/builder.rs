//! Fluent construction of a [`Network`] via [`NetworkBuilder`]

use crate::nn::{
    activation::{ActivationKind, ActivationLayer},
    conv::Conv2DLayer,
    flatten::FlattenLayer,
    linear::LinearLayer,
    network::Network,
    padding::Pad2DLayer,
    pooling::MaxPool2DLayer,
    reshape::ReshapeLayer,
    softmax::SoftmaxLayer,
    upsample::Upsample2DLayer,
    Layer,
};
use std::error::Error;
use std::fmt;

#[derive(Debug, PartialEq, Eq)]
pub enum BuildError {
    /// Attempted to build a network with no layers
    EmptyNetwork,
    /// `.linear()` was called with no `.input(...)` dimension or prior layer output
    MissingInputShape,
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::EmptyNetwork => write!(f, "Cannot build an empty network with 0 layers"),
            BuildError::MissingInputShape => write!(
                f,
                "Missing input shape: call .input(features) before adding a linear layer"
            ),
        }
    }
}

impl Error for BuildError {}

/// Fluent builder for [`Network`]
///
/// Tracks the current output dimension automatically so you never need
/// to manually wire `in_features` for each [`LinearLayer`]
///
/// # Example
/// ```ignore
/// let encoder = NetworkBuilder::new(784)
///     .linear(256).relu
///     .linear(128).relu
///     .linear(latent_dim * 2)
///     .build();
/// ```
pub struct NetworkBuilder {
    /// Collection of layers making up the network
    pub layers: Vec<Box<dyn Layer>>,
    /// Latest size of output
    pub current_features: Option<usize>,
}

impl Default for NetworkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkBuilder {
    /// Creates an empty builder
    pub fn new() -> NetworkBuilder {
        Self {
            layers: Vec::new(),
            current_features: None,
        }
    }

    /// Sets the initial input feature size (e.g. 784 for MNIST or 2 for XOR)
    pub fn input(mut self, features: usize) -> NetworkBuilder {
        self.current_features = Some(features);
        self
    }

    /// Appends a [`LinearLayer`] with Xavier uniform weights
    ///
    /// # Panics
    ///
    /// Panics if no input dimension has been set via [`NetworkBuilder::input`]
    /// or established by a previous layer
    pub fn linear(mut self, out_features: usize) -> NetworkBuilder {
        let in_features = self.current_features.expect("cannot add a linear layer without setting input features first (call .input(...) first)", );
        self.layers
            .push(Box::new(LinearLayer::new_rand(in_features, out_features)));
        self.current_features = Some(out_features);
        self
    }

    /// Appends a [`LinearLayer`] with Kaiming normal weights
    ///
    /// # Panics
    ///
    /// Panics if no input dimension has been set
    pub fn linear_kaiming(mut self, out_features: usize) -> NetworkBuilder {
        let in_features = self.current_features.expect("cannot add a linear layer without setting input features first (call .input(...) first)", );
        self.layers.push(Box::new(LinearLayer::new_kaiming(
            in_features,
            out_features,
        )));
        self.current_features = Some(out_features);
        self
    }

    /// Appends a ReLU activation layer
    pub fn relu(mut self) -> NetworkBuilder {
        self.layers
            .push(Box::new(ActivationLayer::new(ActivationKind::Relu)));
        self
    }

    /// Appends a LeakyReLU activation layer
    pub fn leaky_relu(mut self) -> NetworkBuilder {
        self.layers
            .push(Box::new(ActivationLayer::new(ActivationKind::LeakyRelu)));
        self
    }

    /// Appends a Sigmoid activation layer
    pub fn sigmoid(mut self) -> NetworkBuilder {
        self.layers
            .push(Box::new(ActivationLayer::new(ActivationKind::Sigmoid)));
        self
    }

    /// Appends a Tanh activation layer
    pub fn tanh(mut self) -> NetworkBuilder {
        self.layers
            .push(Box::new(ActivationLayer::new(ActivationKind::Tanh)));
        self
    }

    /// Appends a Softmax layer
    pub fn softmax(mut self) -> NetworkBuilder {
        self.layers.push(Box::new(SoftmaxLayer::new()));
        self
    }

    /// Appends a Flatten layer
    pub fn flatten(mut self) -> NetworkBuilder {
        self.layers.push(Box::new(FlattenLayer::new()));
        self
    }

    /// Appends a Reshape layer with the specified target shape
    pub fn reshape(mut self, target_shape: Vec<usize>) -> NetworkBuilder {
        self.layers.push(Box::new(ReshapeLayer::new(target_shape)));
        self
    }

    /// Appends a Conv2D layer with Kaiming normal initialization
    pub fn conv2d_kaiming(
        mut self,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
    ) -> NetworkBuilder {
        self.layers.push(Box::new(Conv2DLayer::new_kaiming(
            in_channels,
            out_channels,
            kernel_size,
        )));
        self
    }

    /// Appends standard 2x2 max pooling with stride 2
    pub fn max_pool2d(mut self) -> NetworkBuilder {
        self.layers.push(Box::new(MaxPool2DLayer::new_stand()));
        self
    }

    /// Appends 1-pixel zero-padding (for 3x3 same-padding)
    pub fn pad2d_1(mut self) -> NetworkBuilder {
        self.layers.push(Box::new(Pad2DLayer::new_1()));
        self
    }

    /// Appends standard 2x nearest-neighbor upsampling
    pub fn upsample2d_2x(mut self) -> NetworkBuilder {
        self.layers.push(Box::new(Upsample2DLayer::new_2x()));
        self
    }

    /// Appends any custom [`Layer`]
    pub fn add_layer<L: Layer + 'static>(mut self, layer: L) -> NetworkBuilder {
        self.layers.push(Box::new(layer));
        self
    }

    /// Builds and returns the configured [`Network`]
    ///
    /// # Panics
    ///
    /// Panics if no layers were added to the builder
    pub fn build(self) -> Network {
        assert!(
            !self.layers.is_empty(),
            "cannot build a network with 0 layers"
        );
        Network::new(self.layers)
    }
}

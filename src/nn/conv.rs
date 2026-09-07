//! 2D Convolutional layer
//!
//! Performs valid 2D cross-correlation over unbatched 3D tensors:
//! `[in_channels, height, width] -> [out_channels, out_h, out_w]`.
use crate::{nn::init::kaiming_normal_conv, Tensor};

/// A 2D convolutional layer computing valid cross-correlation
///
/// Weights are stored as `[out_channels, in_channels, kh, kw]` and bias as
/// `[out_channels, 1]`. The layer accepts an unbatched `[in_channels, height, width]`
/// tensor and caches its input for the backward pass
#[derive(Clone)]
pub struct Conv2DLayer {
    /// Weight filter tensor shape: `[out_channels, in_channels, kh, kw]`
    pub weight: Tensor,
    /// Bias column vector, shape: `[out_channels, 1]`
    pub bias: Tensor,
    /// Number of expected input channels
    pub in_channels: usize,
    /// Number of output feature maps/filters
    pub out_channels: usize,
    /// Kernel dimensions `(kh, kw)`
    pub kernel_size: (usize, usize),
    /// Input cached by last forward pass, needed for gradients computations
    pub input: Option<Tensor>, // cached `[in_channels, h_in, w_in]`
    /// Weight gradient from last backward pass
    pub d_weight: Option<Tensor>,
    /// Bias gradient from the last backward pass
    pub d_bias: Option<Tensor>,
}

impl Conv2DLayer {
    /// Creates a layer with all weights and biases set to zero
    ///
    /// Useful for tests and parameter loading
    /// Prefer to use `[Conv2DLayer::new_kaiming]`
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
    ) -> Conv2DLayer {
        assert!(in_channels > 0, "in_channels must be > 0");
        assert!(out_channels > 0, "out_channels must be > 0");
        assert!(
            kernel_size.0 > 0 && kernel_size.1 > 0,
            "kernel dimensions must be > 0"
        );
        Conv2DLayer {
            weight: Tensor::new(vec![
                out_channels,
                in_channels,
                kernel_size.0,
                kernel_size.1,
            ]),
            bias: Tensor::new(vec![out_channels, 1]),
            in_channels,
            out_channels,
            kernel_size,
            input: None,
            d_weight: None,
            d_bias: None,
        }
    }

    /// Creates a layer with Kaiming normal weights and zero biases.
    ///
    /// Recommended for layers followed by ReLU activation.
    pub fn new_kaiming(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
    ) -> Conv2DLayer {
        Conv2DLayer {
            weight: kaiming_normal_conv(in_channels, out_channels, kernel_size.0, kernel_size.1),
            bias: Tensor::new(vec![out_channels, 1]),
            in_channels,
            out_channels,
            kernel_size,
            input: None,
            d_weight: None,
            d_bias: None,
        }
    }

    /// Computes valid 2D cross-correlation, caching `input` for the backward pass.
    ///
    /// # Panics
    ///
    /// Panics if `input` is not rank 3, if `input.shape[0] != in_channels`,
    /// or if the input height or width is smaller than the kernel.
    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape.len(),
            3,
            "Input must be 3D [channels, height, width]"
        );
        assert_eq!(
            input.shape[0], self.in_channels,
            "Input channel count mismatch"
        );

        let h_in = input.shape[1];
        let w_in = input.shape[2];
        let (kh, kw) = self.kernel_size;

        assert!(
            h_in >= kh,
            "Input height ({h_in}) must be >= kernel height ({kh})"
        );
        assert!(
            w_in >= kw,
            "Input width ({w_in}) must be >= kernel width ({kw})"
        );

        let out_h = h_in - kh + 1;
        let out_w = w_in - kw + 1;

        self.input = Some(input.clone());

        let mut output = Tensor::new(vec![self.out_channels, out_h, out_w]);

        for oc in 0..self.out_channels {
            let b = self.bias.get(&[oc, 0]);
            for i in 0..out_h {
                for j in 0..out_w {
                    let mut sum = b;
                    for ic in 0..self.in_channels {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let x = input.get(&[ic, i + ki, j + kj]);
                                let w = self.weight.get(&[oc, ic, ki, kj]);
                                sum += x * w;
                            }
                        }
                    }
                    output.set(&[oc, i, j], sum);
                }
            }
        }

        output
    }

    /// Computes gradients with respect to bias, weight and input
    ///
    /// Caches `d_bias` and `d_weight` on the layer for optimiser updates
    /// and returns `(d_weight, d_bias, d_input)`
    ///
    /// # Panics
    ///
    /// Panics if no forward pass has been run yet
    pub fn backward(&mut self, d_output: &Tensor) -> (Tensor, Tensor, Tensor) {
        let d_weight = self.weight_grad(d_output);
        let d_bias = self.bias_grad(d_output);
        let d_input = self.input_grad(d_output);

        self.d_bias = Some(d_bias.clone());
        self.d_weight = Some(d_weight.clone());

        (d_weight, d_bias, d_input)
    }

    /// Computes the bias gradient by summing `d_output`
    ///
    /// Returns a tensor of shape `[out_channels, 1]`
    fn bias_grad(&self, d_output: &Tensor) -> Tensor {
        let out_h = d_output.shape[1];
        let out_w = d_output.shape[2];
        let mut d_bias = Tensor::new(vec![self.out_channels, 1]);
        for oc in 0..self.out_channels {
            let mut sum = 0.0;
            for i in 0..out_h {
                for j in 0..out_w {
                    sum += d_output.get(&[oc, i, j]);
                }
            }
            d_bias.set(&[oc, 0], sum);
        }
        d_bias
    }

    /// Computes filter weight gradients through cross-correlation of cached input
    /// and `d_output`
    ///
    /// Returns a tensor of shape `[out_channels, in_channels, kh, kw]`
    ///
    /// # Panics
    ///
    /// Panics if `self.input` is `None` (forward pass was not called)
    fn weight_grad(&self, d_output: &Tensor) -> Tensor {
        let input = self
            .input
            .as_ref()
            .expect("forward must be called before backward");
        let (kh, kw) = self.kernel_size;
        let out_h = d_output.shape[1];
        let out_w = d_output.shape[2];
        let mut d_weight = Tensor::new(vec![self.out_channels, self.in_channels, kh, kw]);

        for oc in 0..self.out_channels {
            for ic in 0..self.in_channels {
                for ki in 0..kh {
                    for kj in 0..kw {
                        let mut sum = 0.0;
                        for i in 0..out_h {
                            for j in 0..out_w {
                                let dout = d_output.get(&[oc, i, j]);
                                let x = input.get(&[ic, i + ki, j + kj]);
                                sum += dout * x;
                            }
                        }
                        d_weight.set(&[oc, ic, ki, kj], sum);
                    }
                }
            }
        }
        d_weight
    }

    /// Computes gradients with respect to the input tensor through scatter accumulation
    ///
    /// Returns a tensor of shape `[in_channels, h_in, w_in]`
    ///
    /// # Panics
    ///
    /// Panics if `self.input` is `None` (forward pass was not called)
    fn input_grad(&self, d_output: &Tensor) -> Tensor {
        let input = self
            .input
            .as_ref()
            .expect("forward must be called before backward");
        let (kh, kw) = self.kernel_size;
        let h_in = input.shape[1];
        let w_in = input.shape[2];
        let out_h = d_output.shape[1];
        let out_w = d_output.shape[2];
        let mut d_input = Tensor::new(vec![self.in_channels, h_in, w_in]);
        for oc in 0..self.out_channels {
            for i in 0..out_h {
                for j in 0..out_w {
                    let dout = d_output.get(&[oc, i, j]);
                    for ic in 0..self.in_channels {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let w = self.weight.get(&[oc, ic, ki, kj]);
                                let prev = d_input.get(&[ic, i + ki, j + kj]);
                                d_input.set(&[ic, i + ki, j + kj], prev + dout * w);
                            }
                        }
                    }
                }
            }
        }
        d_input
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_conv_forward_hand_calculated() {
        // 1 input channel, 1 output channel, 2x2 kernel
        let mut conv = Conv2DLayer::new(1, 1, (2, 2));
        // Weight = [[1.0, 0.0],
        //           [0.0, 1.0]]
        conv.weight = Tensor::from_vec(vec![1, 1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]);
        // Bias = [0.5]
        conv.bias = Tensor::from_vec(vec![1, 1], vec![0.5]);
        // Input 3x3:
        // [[1.0, 2.0, 3.0],
        //  [4.0, 5.0, 6.0],
        //  [7.0, 8.0, 9.0]]
        let input = Tensor::from_vec(
            vec![1, 3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let output = conv.forward(&input);
        // Expected output shape: [1, 2, 2]
        assert_eq!(output.shape, vec![1, 2, 2]);
        // Expected calculations:
        // (0, 0): (1*1 + 2*0 + 4*0 + 5*1) + 0.5 = 6.0 + 0.5 = 6.5
        // (0, 1): (2*1 + 3*0 + 5*0 + 6*1) + 0.5 = 8.0 + 0.5 = 8.5
        // (1, 0): (4*1 + 5*0 + 7*0 + 8*1) + 0.5 = 12.0 + 0.5 = 12.5
        // (1, 1): (5*1 + 6*0 + 8*0 + 9*1) + 0.5 = 14.0 + 0.5 = 14.5
        let expected = vec![6.5, 8.5, 12.5, 14.5];
        assert_eq!(output.data, expected);
        // Verify cached input
        assert!(conv.input.is_some());
        assert_eq!(conv.input.unwrap().data, input.data);
    }

    #[test]
    fn test_conv_forward_multi_channel() {
        // 2 input channels, 2 output channels, 2x2 kernel
        let mut conv = Conv2DLayer::new(2, 2, (2, 2));
        // Let all weights be 1.0, biases be 0.0
        conv.weight = Tensor::from_vec(vec![2, 2, 2, 2], vec![1.0; 16]);
        conv.bias = Tensor::from_vec(vec![2, 1], vec![0.0, 0.0]);
        // Input 2 channels of 2x2, all 1.0
        let input = Tensor::from_vec(vec![2, 2, 2], vec![1.0; 8]);
        let output = conv.forward(&input);
        // Output size: [2, 1, 1]
        assert_eq!(output.shape, vec![2, 1, 1]);
        // Each output position sums across 2 input channels * 4 kernel pixels = 8.0
        assert_eq!(output.data, vec![8.0, 8.0]);
    }

    #[test]
    #[should_panic(expected = "Input height (2) must be >= kernel height (3)")]
    fn test_conv_forward_panics_on_undersized_height() {
        let mut conv = Conv2DLayer::new(1, 1, (3, 3));
        let input = Tensor::new(vec![1, 2, 3]);
        conv.forward(&input);
    }
}

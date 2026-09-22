//! 2D Convolutional layer
//!
//! Performs valid 2D cross-correlation over unbatched 3D tensors:
//! `[in_channels, height, width] -> [out_channels, out_h, out_w]`.
use std::io::Read;
use std::io::{self};

use crate::{
    nn::{init::kaiming_normal_conv, Layer},
    utils::model_io::{read_tensor, write_tensor, write_u8, TAG_CONV2D},
    Tensor,
};

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
        self.input = Some(input.clone());

        match input.shape.len() {
            3 => {
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
            4 => {
                assert_eq!(
                    input.shape[1], self.in_channels,
                    "Input channel count mismatch"
                );
                let batch_size = input.shape[0];
                let h_in = input.shape[2];
                let w_in = input.shape[3];
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

                let mut output = Tensor::new(vec![batch_size, self.out_channels, out_h, out_w]);

                for b in 0..batch_size {
                    for oc in 0..self.out_channels {
                    let bias_val = self.bias.get(&[oc, 0]);
                    for i in 0..out_h {
                        for j in 0..out_w {
                            let mut sum = bias_val;
                            for ic in 0..self.in_channels {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let x = input.get(&[b, ic, i + ki, j + kj]);
                                        let w = self.weight.get(&[oc, ic, ki, kj]);
                                        sum += x * w;
                                    }
                                }
                            }
                            output.set(&[b, oc, i, j], sum);
                        }
                    }
                }
                }

                output
            }
            _ => panic!("Conv2DLayer expects 3D [channels, height, width] or 4D [batch, channels, height, width]"),
        }
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
        match d_output.shape.len() {
            3 => {
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
            4 => {
                let batch_size = d_output.shape[0];
                let out_h = d_output.shape[2];
                let out_w = d_output.shape[3];
                let mut d_bias = Tensor::new(vec![self.out_channels, 1]);

                for oc in 0..self.out_channels {
                    let mut sum = 0.0;
                    for b in 0..batch_size {
                        for i in 0..out_h {
                            for j in 0..out_w {
                                sum += d_output.get(&[b, oc, i, j]);
                            }
                        }
                    }
                    d_bias.set(&[oc, 0], sum);
                }

                d_bias
            }
            _ => panic!("Conv2DLayer expects 3D [channels, height, width] or 4D [batch, channels, height, width]"),
        }
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
        match d_output.shape.len() {
            3 => {
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
            4 => {
                let batch_size = d_output.shape[0];

                let out_h = d_output.shape[2];
                let out_w = d_output.shape[3];
                let mut d_weight = Tensor::new(vec![self.out_channels, self.in_channels, kh, kw]);

                for oc in 0..self.out_channels {
                    for ic in 0..self.in_channels {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let mut sum = 0.0;
                                for b in 0..batch_size  {
                                    for i in 0..out_h {
                                        for j in 0..out_w {
                                            let dout = d_output.get(&[b, oc, i, j]);
                                            let x = input.get(&[b, ic, i + ki, j + kj]);
                                            sum += dout * x;
                                        }
                                    }
                                }
                                d_weight.set(&[oc, ic, ki, kj], sum);
                            }
                        }
                    }
                }
                d_weight
            }
            _ => panic!("Conv2DLayer expects 3D [channels, height, width] or 4D [batch, channels, height, width]"),
        }
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
        match d_output.shape.len() {
            3 => {
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
            4 => {
                let batch_size =d_output.shape[0];

                let h_in  = input.shape[2];
                let w_in = input.shape[3];
                let out_h = d_output.shape[2];
                let out_w = d_output.shape[3];

                let mut d_input = Tensor::new(vec![batch_size, self.in_channels, h_in, w_in]);

                for oc in 0..self.out_channels {
                    for b in 0..batch_size {
                        for i in 0..out_h {
                            for j in 0..out_w {
                                let dout = d_output.get(&[b, oc, i, j]);
                                for ic in 0..self.in_channels {
                                    for ki in 0..kh {
                                        for kj in 0..kw {
                                            let w = self.weight.get(&[oc, ic, ki, kj]);
                                            let prev = d_input.get(&[b, ic, i + ki, j + kj]);
                                            d_input.set(&[b, ic, i + ki, j + kj], prev + dout * w);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                d_input
            }
            _ => panic!("Conv2DLayer expects 3D [channels, height, width] or 4D [batch, channels, height, width]"),
        }
    }

    pub fn load(reader: &mut dyn Read) -> io::Result<Conv2DLayer> {
        let weight = read_tensor(reader)?;
        let bias = read_tensor(reader)?;
        let out_channels = weight.shape[0];
        let in_channels = weight.shape[1];
        let kernel_size = (weight.shape[2], weight.shape[3]);
        Ok(Conv2DLayer {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            input: None,
            d_weight: None,
            d_bias: None,
        })
    }
}

impl Layer for Conv2DLayer {
    fn forward_pass(&mut self, input: &Tensor) -> Tensor {
        self.forward(input)
    }

    fn backward_pass(&mut self, d_output: &Tensor) -> Tensor {
        let (_, _, d_input) = self.backward(d_output);
        d_input
    }

    fn set_params(&mut self, params: Vec<Tensor>) {
        self.weight = params[0].clone();
        self.bias = params[1].clone();
    }

    fn get_params(&self) -> Vec<Tensor> {
        let params = vec![self.weight.clone(), self.bias.clone()];
        params
    }

    fn get_grads(&self) -> Vec<Tensor> {
        let grads = vec![
            self.d_weight.clone().unwrap_or_else(|| {
                Tensor::new(vec![
                    self.out_channels,
                    self.in_channels,
                    self.kernel_size.0,
                    self.kernel_size.1,
                ])
            }),
            self.d_bias
                .clone()
                .unwrap_or_else(|| Tensor::new(vec![self.out_channels, 1])),
        ];
        grads
    }

    fn save(&self, writer: &mut dyn std::io::prelude::Write) -> std::io::Result<()> {
        write_u8(writer, TAG_CONV2D)?;
        write_tensor(writer, &self.weight)?;
        write_tensor(writer, &self.bias)?;
        Ok(())
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

    #[test]
    fn test_conv_forward_and_backward_batched() {
        let mut conv_single = Conv2DLayer::new(1, 2, (2, 2));
        conv_single.weight = Tensor::from_vec(
            vec![2, 1, 2, 2],
            vec![1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 1.5, 2.0],
        );
        conv_single.bias = Tensor::from_vec(vec![2, 1], vec![0.1, -0.2]);
        let mut conv_batched = conv_single.clone();

        let sample0 = Tensor::from_vec(
            vec![1, 3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let sample1 = Tensor::from_vec(
            vec![1, 3, 3],
            vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        );

        let out0 = conv_single.forward(&sample0);
        let (dw0, db0, din0) = conv_single.backward(&Tensor::from_vec(vec![2, 2, 2], vec![1.0; 8]));

        let out1 = conv_single.forward(&sample1);
        let (dw1, db1, din1) = conv_single.backward(&Tensor::from_vec(vec![2, 2, 2], vec![2.0; 8]));

        let mut batch_data = sample0.data.clone();
        batch_data.extend(&sample1.data);
        let batched_in = Tensor::from_vec(vec![2, 1, 3, 3], batch_data);

        let batched_out = conv_batched.forward(&batched_in);
        assert_eq!(batched_out.shape, vec![2, 2, 2, 2]);

        let sample_out_size = 2 * 2 * 2;
        assert_eq!(&batched_out.data[0..sample_out_size], &out0.data[..]);
        assert_eq!(&batched_out.data[sample_out_size..], &out1.data[..]);

        let mut batched_dout_data = vec![1.0; 8];
        batched_dout_data.extend(vec![2.0; 8]);
        let batched_dout = Tensor::from_vec(vec![2, 2, 2, 2], batched_dout_data);

        let (dw_b, db_b, din_b) = conv_batched.backward(&batched_dout);

        for i in 0..db_b.data.len() {
            assert!((db_b.data[i] - (db0.data[i] + db1.data[i])).abs() < 1e-5);
        }

        for i in 0..dw_b.data.len() {
            assert!((dw_b.data[i] - (dw0.data[i] + dw1.data[i])).abs() < 1e-5);
        }

        assert_eq!(din_b.shape, vec![2, 1, 3, 3]);
        let sample_in_size = 3 * 3;
        for i in 0..sample_in_size {
            assert!((din_b.data[i] - din0.data[i]).abs() < 1e-5);
            assert!((din_b.data[sample_in_size + i] - din1.data[i]).abs() < 1e-5);
        }
    }
}

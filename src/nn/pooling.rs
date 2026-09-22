//! 2D Max Pooling layer
//!
//! Downsamples dimensions over unbatched 3D tensors
//! `[channels, height, width] -> [channels, out_h, out_w]`.

use std::io::{self, Read, Write};

use crate::{
    nn::Layer,
    tensor::Tensor,
    utils::model_io::{read_u32, write_u32, write_u8, TAG_MAXPOOL2D},
};

/// A 2D max pooling layer that extracts the maximum value over sliding windows
///
/// Works independently per channel
#[derive(Clone)]
pub struct MaxPool2DLayer {
    /// Pooling kernel dimensions `(kh, kw)`
    pub kernel_size: (usize, usize),
    /// Step size across height and width
    pub stride: usize,
    /// Coordinates of each windows maximum,
    /// ordered row majot across `[channels, out_h, out_w]`
    pub argmax: Option<Vec<(usize, usize)>>,
    /// Input shape cached from the last forward pass
    pub input_shape: Option<Vec<usize>>,
}

impl MaxPool2DLayer {
    /// Creates a new max pooling layer with the given kernelt dimensions and stide
    ///
    /// # Panics
    ///
    /// Panics if `kernel_size.0 == 0`, `kernel_size.1 == 0` or `stride == 0`
    pub fn new(kernel_size: (usize, usize), stride: usize) -> MaxPool2DLayer {
        assert!(stride > 0, "stride must be > 0");
        assert!(
            kernel_size.0 > 0 && kernel_size.1 > 0,
            "kernel dimensions must be > 0"
        );
        MaxPool2DLayer {
            kernel_size,
            stride,
            argmax: None,
            input_shape: None,
        }
    }

    /// Creates a standard 2x2 pooling layer with stride 2
    pub fn new_stand() -> MaxPool2DLayer {
        Self::new((2, 2), 2)
    }

    /// Computes 2D max pooling, caching `argmax` positions and `input_shape` for backward
    ///
    /// # Panics
    ///
    /// Panics if `input` is not rank 3 or if input height/ width is smaller
    /// then the kernel size
    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        match input.shape.len() {
            3 => {
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

                let out_h = (h_in - kh) / self.stride + 1;
                let out_w = (w_in - kw) / self.stride + 1;

                let channels = input.shape[0];
                let mut output = Tensor::new(vec![channels, out_h, out_w]);
                let mut argmax = Vec::with_capacity(channels * out_h * out_w);

                for c in 0..channels {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let h_start = oh * self.stride;
                            let h_end = h_start + kh;
                            let w_start = ow * self.stride;
                            let w_end = w_start + kw;

                            let mut max_val = input.get(&[c, h_start, w_start]);
                            let mut max_pos = (h_start, w_start);
                            for i in h_start..h_end {
                                for j in w_start..w_end {
                                    let val = input.get(&[c, i, j]);
                                    if val > max_val {
                                        max_val = val;
                                        max_pos = (i, j);
                                    }
                                }
                            }

                            output.set(&[c, oh, ow], max_val);
                            argmax.push(max_pos);
                        }
                    }
                }
                self.argmax = Some(argmax);
                self.input_shape = Some(input.shape.clone());

                output
            }
            4 => {
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

                let out_h = (h_in - kh) / self.stride + 1;
                let out_w = (w_in - kw) / self.stride + 1;

                let channels = input.shape[1];
                let mut output = Tensor::new(vec![batch_size, channels, out_h, out_w]);
                let mut argmax = Vec::with_capacity(batch_size * channels * out_h * out_w);

                for b in 0..batch_size {
                    for c in 0..channels {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let h_start = oh * self.stride;
                                let h_end = h_start + kh;
                                let w_start = ow * self.stride;
                                let w_end = w_start + kw;

                                let mut max_val = input.get(&[b, c, h_start, w_start]);
                                let mut max_pos = (h_start, w_start);
                                for i in h_start..h_end {
                                    for j in w_start..w_end {
                                        let val = input.get(&[b, c, i, j]);
                                        if val > max_val {
                                            max_val = val;
                                            max_pos = (i, j);
                                        }
                                    }
                                }

                                output.set(&[b, c, oh, ow], max_val);
                                argmax.push(max_pos);
                            }
                        }
                    }
                }
                self.argmax = Some(argmax);
                self.input_shape = Some(input.shape.clone());

                output
            }
            _ => panic!("MaxPool2DLayer::forward requires a rank 3 or 4 tensor"),
        }
    }

    /// Routes upstream gradients back to the locations of the maximum activations.
    ///
    /// Accumulates gradients to handle overlapping windows.
    ///
    /// # Panics
    ///
    /// Panics if `forward` was not called before `backward`.
    pub fn backward(&mut self, d_output: &Tensor) -> Tensor {
        let input_shape = self
            .input_shape
            .as_ref()
            .expect("forward must be called before backward");

        let argmax = self
            .argmax
            .as_ref()
            .expect("forward must be called before backward");

        let mut d_input = Tensor::new(input_shape.clone());

        match d_output.shape.len() {
            3 => {
                let out_h = d_output.shape[1];
                let out_w = d_output.shape[2];
                let channels = input_shape[0];

                let mut idx = 0;
                for c in 0..channels {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let (max_h, max_w) = argmax[idx];
                            idx += 1;

                            let dout = d_output.get(&[c, oh, ow]);

                            let prev = d_input.get(&[c, max_h, max_w]);
                            d_input.set(&[c, max_h, max_w], prev + dout);
                        }
                    }
                }
                d_input
            }
            4 => {
                let batch_size = d_output.shape[0];

                let out_h = d_output.shape[2];
                let out_w = d_output.shape[3];
                let channels = input_shape[1];

                let mut idx = 0;
                for b in 0..batch_size {
                    for c in 0..channels {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let (max_h, max_w) = argmax[idx];
                                idx += 1;

                                let dout = d_output.get(&[b, c, oh, ow]);

                                let prev = d_input.get(&[b, c, max_h, max_w]);
                                d_input.set(&[b, c, max_h, max_w], prev + dout);
                            }
                        }
                    }
                }
                d_input
            }
            _ => panic!("MaxPool2DLayer::backward requires a rank 3 or 4 tensor"),
        }
    }

    pub fn load(reader: &mut dyn Read) -> io::Result<MaxPool2DLayer> {
        let kh = read_u32(reader)? as usize;
        let kw = read_u32(reader)? as usize;
        let stride = read_u32(reader)? as usize;
        Ok(MaxPool2DLayer::new((kh, kw), stride))
    }
}

impl Layer for MaxPool2DLayer {
    fn save(&self, writer: &mut dyn Write) -> io::Result<()> {
        write_u8(writer, TAG_MAXPOOL2D)?;
        write_u32(writer, self.kernel_size.0 as u32)?;
        write_u32(writer, self.kernel_size.1 as u32)?;
        write_u32(writer, self.stride as u32)?;
        Ok(())
    }

    fn forward_pass(&mut self, input: &Tensor) -> Tensor {
        self.forward(input)
    }

    fn backward_pass(&mut self, d_output: &Tensor) -> Tensor {
        self.backward(d_output)
    }

    fn set_params(&mut self, _params: Vec<Tensor>) {}

    fn get_params(&self) -> Vec<Tensor> {
        Vec::new()
    }

    fn get_grads(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_maxpool_forward_and_backward() {
        let mut pool = MaxPool2DLayer::new_stand(); // 2x2, stride 2
                                                    // 1 channel, 4x4 input
        let input = Tensor::from_vec(
            vec![1, 4, 4],
            vec![
                1.0, 3.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 3.0, 2.0, 1.0, 0.0, -1.0, 4.0, 9.0, 2.0,
            ],
        );
        let out = pool.forward(&input);
        assert_eq!(out.shape, vec![1, 2, 2]);
        // Window (0,0): max is 6.0 (at row 1, col 1)
        // Window (0,1): max is 8.0 (at row 1, col 3)
        // Window (1,0): max is 4.0 (at row 3, col 1)
        // Window (1,1): max is 9.0 (at row 3, col 2)
        assert_eq!(out.data, vec![6.0, 8.0, 4.0, 9.0]);
        // Upstream gradient: 2x2
        let d_out = Tensor::from_vec(vec![1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let d_in = pool.backward(&d_out);
        assert_eq!(d_in.shape, vec![1, 4, 4]);
        // All zeros except at the argmax positions:
        assert_eq!(d_in.get(&[0, 1, 1]), 1.0);
        assert_eq!(d_in.get(&[0, 1, 3]), 2.0);
        assert_eq!(d_in.get(&[0, 3, 1]), 3.0);
        assert_eq!(d_in.get(&[0, 3, 2]), 4.0);
    }

    #[test]
    fn test_maxpool_batched_forward_and_backward() {
        let mut pool_single = MaxPool2DLayer::new_stand();
        let mut pool_batched = MaxPool2DLayer::new_stand();

        let sample0 = Tensor::from_vec(
            vec![1, 4, 4],
            vec![
                1.0, 3.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 3.0, 2.0, 1.0, 0.0, -1.0, 4.0, 9.0, 2.0,
            ],
        );

        let dout0 = Tensor::from_vec(vec![1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);

        let sample1 = Tensor::from_vec(
            vec![1, 4, 4],
            vec![
                9.0, 1.0, 8.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 3.0, 2.0, 5.0, 4.0, 8.0,
            ],
        );

        let dout1 = Tensor::from_vec(vec![1, 2, 2], vec![5.0, 6.0, 7.0, 8.0]);

        let out0 = pool_single.forward(&sample0);
        let din0 = pool_single.backward(&dout0);
        let out1 = pool_single.forward(&sample1);
        let din1 = pool_single.backward(&dout1);

        let mut batch_in_data = sample0.data.clone();
        batch_in_data.extend(&sample1.data);
        let batched_in = Tensor::from_vec(vec![2, 1, 4, 4], batch_in_data);

        let batched_out = pool_batched.forward(&batched_in);
        assert_eq!(batched_out.shape, vec![2, 1, 2, 2]);

        assert_eq!(&batched_out.data[0..4], &out0.data[..]);
        assert_eq!(&batched_out.data[4..8], &out1.data[..]);

        let mut batch_dout_data = dout0.data.clone();
        batch_dout_data.extend(&dout1.data);
        let batched_dout = Tensor::from_vec(vec![2, 1, 2, 2], batch_dout_data);

        let batched_din = pool_batched.backward(&batched_dout);
        assert_eq!(batched_din.shape, vec![2, 1, 4, 4]);

        assert_eq!(&batched_din.data[0..16], &din0.data[..]);
        assert_eq!(&batched_din.data[16..32], &din1.data[..]);
    }
}

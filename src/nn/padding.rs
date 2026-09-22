//! 2D Zero Padding Layer
//!
//! Adds padding border of zeros around an unbatched 3D tensor
//! `[channels, height, width] -> [channels, height + 2*ph, width + 2*pw]`
use std::{
    io::{self, Read, Write},
    panic,
};

use crate::{
    nn::Layer,
    utils::model_io::{read_u32, write_u32, write_u8, TAG_PAD2D},
    Tensor,
};

/// A 2D zero-padding layer
///
/// Places the input tensor in the centre of a bigger zero filled tensor
/// Commonly placed immediately before a [`crate::nn::conv::Conv2DLayer`] to achieve
/// "same" spatial dimensions across convolutions
#[derive(Clone)]
pub struct Pad2DLayer {
    /// Padding added to top/bottom (ph) and left/right (pw)
    pub padding: (usize, usize),
    /// Cached input shape for validation
    pub input_shape: Option<Vec<usize>>,
}

impl Pad2DLayer {
    /// Creates a padding layer with specified vertical and horizontal padding
    pub fn new(padding: (usize, usize)) -> Pad2DLayer {
        assert!(
            padding.0 > 0 && padding.1 > 0,
            "Padding must be greater than 0"
        );
        Pad2DLayer {
            padding,
            input_shape: None,
        }
    }

    /// Creates a padding layer with symmetric vertical and horizontal padding
    pub fn new_same(pad: usize) -> Self {
        Self::new((pad, pad))
    }

    /// Standard 1-pixel border padding for 3x3 "same" convolutions
    pub fn new_1() -> Self {
        Self::new((1, 1))
    }

    /// Embeds the input into a zero-padded tensor
    ///
    /// # Panics
    ///
    /// Panics if `input` is not rank 3
    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        self.input_shape = Some(input.shape.clone());

        match input.shape.len() {
            3 => {
                let h_in = input.shape[1];
                let w_in = input.shape[2];
                let channels = input.shape[0];
                let (ph, pw) = self.padding;

                let out_h = h_in + (2 * ph);
                let out_w = w_in + (2 * pw);

                let mut output = Tensor::new(vec![channels, out_h, out_w]);

                for c in 0..channels {
                    for ih in 0..h_in {
                        for iw in 0..w_in {
                            let val = input.get(&[c, ih, iw]);
                            output.set(&[c, ih + ph, iw + pw], val);
                        }
                    }
                }

                output
            }
            4 => {
                let batch_size = input.shape[0];

                let channels = input.shape[1];
                let h_in = input.shape[2];
                let w_in = input.shape[3];
                let (ph, pw) = self.padding;

                let out_h = h_in + (2 * ph);
                let out_w = w_in + (2 * pw);

                let mut output = Tensor::new(vec![batch_size, channels, out_h, out_w]);

                for b in 0..batch_size {
                    for c in 0..channels {
                        for ih in 0..h_in {
                            for iw in 0..w_in {
                                let val = input.get(&[b, c, ih, iw]);
                                output.set(&[b, c, ih + ph, iw + pw], val);
                            }
                        }
                    }
                }

                output
            }
            _ => panic!("Pad2DLayer::forward requires a rank 3 or 4 tensor"),
        }
    }

    /// Slices the unpadded centre region out of `d_output`
    ///
    /// # Panics
    ///
    /// Panics if `forward` was not called before to `backward`
    pub fn backward(&mut self, d_output: &Tensor) -> Tensor {
        let input_shape = self
            .input_shape
            .as_ref()
            .expect("forward must be called before backward");

        let mut d_input = Tensor::new(input_shape.clone());

        match d_output.shape.len() {
            3 => {
                let h_in = input_shape[1];
                let w_in = input_shape[2];
                let channels = input_shape[0];
                let (ph, pw) = self.padding;

                for c in 0..channels {
                    for ih in 0..h_in {
                        for iw in 0..w_in {
                            let val = d_output.get(&[c, ih + ph, iw + pw]);
                            d_input.set(&[c, ih, iw], val);
                        }
                    }
                }

                d_input
            }
            4 => {
                let batch_size = input_shape[0];

                let h_in = input_shape[2];
                let w_in = input_shape[3];
                let channels = input_shape[1];
                let (ph, pw) = self.padding;

                for b in 0..batch_size {
                    for c in 0..channels {
                        for ih in 0..h_in {
                            for iw in 0..w_in {
                                let val = d_output.get(&[b, c, ih + ph, iw + pw]);
                                d_input.set(&[b, c, ih, iw], val);
                            }
                        }
                    }
                }

                d_input
            }
            _ => panic!("Pad2DLayer::backward requires a rank 3 or 4 tensor"),
        }
    }

    pub fn load(reader: &mut dyn Read) -> io::Result<Pad2DLayer> {
        let ph = read_u32(reader)? as usize;
        let pw = read_u32(reader)? as usize;
        Ok(Pad2DLayer::new((ph, pw)))
    }
}

impl Layer for Pad2DLayer {
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

    fn save(&self, writer: &mut dyn Write) -> io::Result<()> {
        write_u8(writer, TAG_PAD2D)?;
        write_u32(writer, self.padding.0 as u32)?;
        write_u32(writer, self.padding.1 as u32)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_forward_and_backward_batched() {
        let mut pad_single = Pad2DLayer::new_1();
        let mut pad_batched = Pad2DLayer::new_1();

        let sample0 = Tensor::from_vec(vec![1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let dout0 = Tensor::from_vec(vec![1, 4, 4], (1..=16).map(|x| x as f32).collect());

        let sample1 = Tensor::from_vec(vec![1, 2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        let dout1 = Tensor::from_vec(vec![1, 4, 4], (17..=32).map(|x| x as f32).collect());

        let out0 = pad_single.forward(&sample0);
        let din0 = pad_single.backward(&dout0);
        let out1 = pad_single.forward(&sample1);
        let din1 = pad_single.backward(&dout1);

        let mut batch_in_data = sample0.data.clone();
        batch_in_data.extend(&sample1.data);
        let batched_in = Tensor::from_vec(vec![2, 1, 2, 2], batch_in_data);
        
        let batched_out = pad_batched.forward(&batched_in);
        assert_eq!(batched_out.shape, vec![2, 1, 4, 4]);
        assert_eq!(&batched_out.data[0..16], &out0.data[..]);
        assert_eq!(&batched_out.data[16..32], &out1.data[..]);

        let mut batch_dout_data = dout0.data.clone();
        batch_dout_data.extend(&dout1.data);
        let batched_dout = Tensor::from_vec(vec![2, 1, 4, 4], batch_dout_data);
        
        let batched_din = pad_batched.backward(&batched_dout);
        assert_eq!(batched_din.shape, vec![2, 1, 2, 2]);
        assert_eq!(&batched_din.data[0..4], &din0.data[..]);
        assert_eq!(&batched_din.data[4..8], &din1.data[..]);
    }
}
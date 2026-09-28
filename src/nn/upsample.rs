//! 2D Nearest Neighbour Upsampling Layer
//!
//! Expands the spatial dimensions over unbatched 3D tensors
//! `[channels, height, width] -> [channels, height * scale, width * scale]`
use std::{
    io::{self, Read, Write},
    panic,
};

use crate::{
    nn::Layer,
    tensor::Tensor,
    utils::model_io::{read_u32, write_u32, write_u8, TAG_UPSAMPLE2D},
};

/// A 2D nearest neighbour upsampling layer
///
/// Duplicates each spatial element into a `(S x S)`, where `S` is `scale_factor`,
/// block across all channels.
#[derive(Clone)]
pub struct Upsample2DLayer {
    /// Integer scaling multiplier for height and width (e.g 2 doubles the size)
    pub scale_factor: usize,
    /// Cached input shape from forward pass (used for backward validation)
    pub input_shape: Option<Vec<usize>>,
}

impl Upsample2DLayer {
    /// Creates a new upsampler with given integer scaling factor
    ///
    /// # Panics
    ///
    /// Panics if `scale_factor == 0`
    pub fn new(scale_factor: usize) -> Upsample2DLayer {
        assert!(scale_factor > 0, "scale_factor must be > 0");
        Upsample2DLayer {
            scale_factor,
            input_shape: None,
        }
    }

    /// Creates a standard 2x nearest-neighbor upsampling layer.
    pub fn new_2x() -> Upsample2DLayer {
        Self::new(2)
    }

    /// Expands spatial dimension by cloning cells into `(scale x scale)` blocks
    ///
    ///  # Panics
    ///
    /// Panics if `input` is not rank 3
    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        self.input_shape = Some(input.shape.clone());

        match input.shape.len() {
            3 => {
                let h_in = input.shape[1];
                let w_in = input.shape[2];
                let channels = input.shape[0];
                let scale = self.scale_factor;

                let out_h = h_in * scale;
                let out_w = w_in * scale;

                let mut output = Tensor::new(vec![channels, out_h, out_w]);

                let in_data = &input.data;
                let out_data = &mut output.data;

                for c in 0..channels {
                    let in_plane_offset = c * (h_in * w_in);
                    let out_plane_offset = c * (out_h * out_w);

                    for ih in 0..h_in {
                        let in_row_start = in_plane_offset + ih * w_in;
                        let out_row_base = out_plane_offset + (ih * scale) * out_w;

                        for iw in 0..w_in {
                            let val = in_data[in_row_start + iw];
                            let out_col_start = out_row_base + iw * scale;
                            for sw in 0..scale {
                                out_data[out_col_start + sw] = val;
                            }
                        }

                        for sh in 1..scale {
                            let copy_row_start = out_row_base + sh * out_w;
                            out_data
                                .copy_within(out_row_base..out_row_base + out_w, copy_row_start);
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
                let scale = self.scale_factor;

                let out_h = h_in * scale;
                let out_w = w_in * scale;

                let mut output = Tensor::new(vec![batch_size, channels, out_h, out_w]);

                let in_data = &input.data;
                let out_data = &mut output.data;

                let in_spatial_size = h_in * w_in;
                let out_spatial_size = out_h * out_w;
                let total_planes = batch_size * channels;

                for plane in 0..total_planes {
                    let in_plane_offset = plane * in_spatial_size;
                    let out_plane_offset = plane * out_spatial_size;

                    for ih in 0..h_in {
                        let in_row_start = in_plane_offset + ih * w_in;
                        let out_row_base = out_plane_offset + (ih * scale) * out_w;

                        for iw in 0..w_in {
                            let val = in_data[in_row_start + iw];
                            let out_col_start = out_row_base + iw * scale;
                            for sw in 0..scale {
                                out_data[out_col_start + sw] = val;
                            }
                        }

                        for sh in 1..scale {
                            let copy_row_start = out_row_base + sh * out_w;
                            out_data
                                .copy_within(out_row_base..out_row_base + out_w, copy_row_start);
                        }
                    }
                }

                output
            }
            _ => panic!("Upsample2DLayer::forward requires a rank 3 or 4 tensor"),
        }
    }

    /// Sums incoming gradients across each `(scale x scale)` block bac to the input
    ///
    /// # Panics
    ///
    /// Panics if `forward` was not called before `backward`
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
                let scale = self.scale_factor;

                let out_w = d_output.shape[2];

                let dout_data = &d_output.data;
                let din_data = &mut d_input.data;

                for c in 0..channels {
                    let in_plane_offset = c * (h_in * w_in);
                    let out_plane_offset = c * (d_output.shape[1] * out_w);

                    for ih in 0..h_in {
                        let in_row_start = in_plane_offset + ih * w_in;
                        let out_row_base = out_plane_offset + (ih * scale) * out_w;

                        for iw in 0..w_in {
                            let out_col_start = out_row_base + iw * scale;
                            let mut sum = 0.0;
                            for sh in 0..scale {
                                let row_offset = out_col_start + sh * out_w;
                                for sw in 0..scale {
                                    sum += dout_data[row_offset + sw];
                                }
                            }
                            din_data[in_row_start + iw] = sum;
                        }
                    }
                }

                d_input
            }
            4 => {
                let batch_size = input_shape[0];
                let channels = input_shape[1];
                let h_in = input_shape[2];
                let w_in = input_shape[3];
                let scale = self.scale_factor;

                let out_w = d_output.shape[3];

                let dout_data = &d_output.data;
                let din_data = &mut d_input.data;

                let in_spatial_size = h_in * w_in;
                let out_spatial_size = d_output.shape[2] * out_w;
                let total_planes = batch_size * channels;

                for plane in 0..total_planes {
                    let in_plane_offset = plane * in_spatial_size;
                    let out_plane_offset = plane * out_spatial_size;

                    for ih in 0..h_in {
                        let in_row_start = in_plane_offset + ih * w_in;
                        let out_row_base = out_plane_offset + (ih * scale) * out_w;

                        for iw in 0..w_in {
                            let out_col_start = out_row_base + iw * scale;
                            let mut sum = 0.0;
                            for sh in 0..scale {
                                let row_offset = out_col_start + sh * out_w;
                                for sw in 0..scale {
                                    sum += dout_data[row_offset + sw];
                                }
                            }
                            din_data[in_row_start + iw] = sum;
                        }
                    }
                }

                d_input
            }
            _ => panic!("Upsample2DLayer::backward requires a rank 3 or 4 tensor"),
        }
    }

    pub fn load(reader: &mut dyn Read) -> io::Result<Upsample2DLayer> {
        let scale_factor = read_u32(reader)? as usize;
        Ok(Upsample2DLayer::new(scale_factor))
    }
}

impl Layer for Upsample2DLayer {
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
        write_u8(writer, TAG_UPSAMPLE2D)?;
        write_u32(writer, self.scale_factor as u32)?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use std::assert_eq;

    use super::*;

    #[test]
    fn test_upsample_forward_and_backward() {
        let mut layer = Upsample2DLayer::new_2x();

        let input = Tensor::from_vec(vec![1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let output = layer.forward(&input);

        assert_eq!(output.shape, vec![1, 4, 4]);
        let expected_data = vec![
            1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
        ];
        assert_eq!(output.data, expected_data);

        let d_output = Tensor::from_vec(vec![1, 4, 4], vec![1.0; 16]);
        let d_input = layer.backward(&d_output);

        assert_eq!(d_input.shape, vec![1, 2, 2]);
        assert_eq!(d_input.data, vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_upsample_forward_and_backward_batched() {
        let mut up_single = Upsample2DLayer::new_2x();
        let mut up_batched = Upsample2DLayer::new_2x();

        let sample0 = Tensor::from_vec(vec![1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let dout0 = Tensor::from_vec(vec![1, 4, 4], (1..=16).map(|x| x as f32).collect());

        let sample1 = Tensor::from_vec(vec![1, 2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        let dout1 = Tensor::from_vec(vec![1, 4, 4], (17..=32).map(|x| x as f32).collect());

        let out0 = up_single.forward(&sample0);
        let din0 = up_single.backward(&dout0);
        let out1 = up_single.forward(&sample1);
        let din1 = up_single.backward(&dout1);

        let mut batch_in_data = sample0.data.clone();
        batch_in_data.extend(&sample1.data);
        let batched_in = Tensor::from_vec(vec![2, 1, 2, 2], batch_in_data);

        let batched_out = up_batched.forward(&batched_in);
        assert_eq!(batched_out.shape, vec![2, 1, 4, 4]);
        assert_eq!(&batched_out.data[0..16], &out0.data[..]);
        assert_eq!(&batched_out.data[16..32], &out1.data[..]);

        let mut batch_dout_data = dout0.data.clone();
        batch_dout_data.extend(&dout1.data);
        let batched_dout = Tensor::from_vec(vec![2, 1, 4, 4], batch_dout_data);

        let batched_din = up_batched.backward(&batched_dout);
        assert_eq!(batched_din.shape, vec![2, 1, 2, 2]);
        assert_eq!(&batched_din.data[0..4], &din0.data[..]);
        assert_eq!(&batched_din.data[4..8], &din1.data[..]);
    }
}

use std::{
    io::{self, Read, Write},
    vec,
};

use crate::{
    nn::init::{kaiming_normal, xavier_uniform},
    nn::Layer,
    tensor::Tensor,
    utils::model_io::{read_tensor, write_tensor, write_u8, TAG_LINEAR},
};

#[derive(Clone)]
pub struct LinearLayer {
    pub weight: Tensor,
    pub bias: Tensor,
    pub in_features: usize,
    pub out_features: usize,
    pub input: Option<Tensor>,
    pub d_weight: Option<Tensor>,
    pub d_bias: Option<Tensor>,
}

impl LinearLayer {
    pub fn new(in_features: usize, out_features: usize) -> LinearLayer {
        LinearLayer {
            weight: Tensor::new(vec![out_features, in_features]),
            bias: Tensor::new(vec![out_features, 1]),
            in_features,
            out_features,
            input: None,
            d_weight: None,
            d_bias: None,
        }
    }

    pub fn new_rand(in_features: usize, out_features: usize) -> LinearLayer {
        LinearLayer {
            weight: xavier_uniform(in_features, out_features),
            bias: Tensor::new(vec![out_features, 1]),
            in_features,
            out_features,
            input: None,
            d_weight: None,
            d_bias: None,
        }
    }

    pub fn new_kaiming(in_features: usize, out_features: usize) -> LinearLayer {
        LinearLayer {
            weight: kaiming_normal(in_features, out_features),
            bias: Tensor::new(vec![out_features, 1]),
            in_features,
            out_features,
            input: None,
            d_weight: None,
            d_bias: None,
        }
    }

    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        self.input = Some(input.clone());

        match input.shape.len() {
            2 => self.weight.matmul(input).add(&self.bias),
            3 => {
                let batched_matmul_result = self.weight.matmul_batched_broadcast(input);
                let batch_size = input.shape[0];

                let bias_broadcasted = self._broadcast_bias_batched(&self.bias, batch_size);
                batched_matmul_result.add(&bias_broadcasted)
            }
            _ => panic!("LinearLayer only supports 2D or 3D inputs"),
        }
    }

    fn _broadcast_bias_batched(&self, bias: &Tensor, batch_size: usize) -> Tensor {
        assert_eq!(bias.shape.len(), 2, "bias must be 2D");

        let out_feat = bias.shape[0];
        let mut broadcasted = Tensor::new(vec![batch_size, out_feat, 1]);

        for b in 0..batch_size {
            for o in 0..out_feat {
                broadcasted.set(&[b, o, 0], bias.get(&[o, 0]));
            }
        }
        broadcasted
    }

    pub fn backward(&self, d_output: &Tensor) -> (Tensor, Tensor, Tensor) {
        let d_w = d_output.matmul(&self.input.as_ref().unwrap().transpose());
        let d_b = d_output.clone();
        let d_x = self.weight.transpose().matmul(d_output);
        (d_w, d_b, d_x)
    }

    pub fn get_weights_and_bias(&self) -> (&Tensor, &Tensor) {
        (&self.weight, &self.bias)
    }

    pub fn load(reader: &mut dyn Read) -> io::Result<LinearLayer> {
        let weight = read_tensor(reader)?;
        let bias = read_tensor(reader)?;
        let out_features = weight.shape[0];
        let in_features = weight.shape[1];
        Ok(LinearLayer {
            weight,
            bias,
            in_features,
            out_features,
            input: None,
            d_weight: None,
            d_bias: None,
        })
    }
}

impl Layer for LinearLayer {
    fn forward_pass(&mut self, input: &Tensor) -> Tensor {
        self.forward(input)
    }

    fn backward_pass(&mut self, d_output: &Tensor) -> Tensor {
        let input = self.input.as_ref().unwrap();

        match input.shape.len() {
            2 => {
                // standard backwards pass
                self.d_weight = Some(d_output.matmul(&self.input.as_ref().unwrap().transpose()));
                self.d_bias = Some(d_output.clone());
                self.weight.transpose().matmul(d_output)
            }
            3 => {
                // batched inputs
                let batch_size = input.shape[0];
                let in_features = input.shape[1];

                let mut d_w_acc = vec![0.0f32; self.out_features * in_features];

                for b in 0..batch_size {
                    let d_out_b =
                        &d_output.data[b * self.out_features..(b + 1) * self.out_features];
                    let in_b = &input.data[b * in_features..(b + 1) * in_features];

                    for i in 0..self.out_features {
                        let g = d_out_b[i];
                        for j in 0..in_features {
                            d_w_acc[i * in_features + j] += g * in_b[j];
                        }
                    }
                }

                self.d_weight = Some(Tensor::from_vec(
                    vec![self.out_features, in_features],
                    d_w_acc,
                ));
                self.d_bias = Some(d_output.sum_batch());

                self.weight.transpose().matmul_batched_broadcast(d_output)
            }
            _ => panic!("LinearLayer only supports 2D or 3D inputs"),
        }
    }

    fn get_params(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
    fn get_grads(&self) -> Vec<Tensor> {
        vec![
            self.d_weight
                .as_ref()
                .cloned()
                .unwrap_or_else(|| Tensor::new(vec![self.out_features, self.in_features])),
            self.d_bias
                .as_ref()
                .cloned()
                .unwrap_or_else(|| Tensor::new(vec![self.out_features, 1])),
        ]
    }
    fn set_params(&mut self, params: Vec<Tensor>) {
        self.weight = params[0].clone();
        self.bias = params[1].clone();
    }

    fn save(&self, writer: &mut dyn Write) -> io::Result<()> {
        write_u8(writer, TAG_LINEAR)?;
        write_tensor(writer, &self.weight)?;
        write_tensor(writer, &self.bias)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward() {
        let mut layer = LinearLayer::new(3, 1);
        layer.weight = Tensor::from_vec(vec![1, 3], vec![1.0, 1.0, 1.0]);
        let input = Tensor::from_vec(vec![3, 1], vec![2.0, 2.0, 2.0]);
        let forward_res = layer.forward(&input);
        assert_eq!(forward_res.data, vec![6.0]);
    }

    #[test]
    fn test_backward() {
        let mut layer = LinearLayer::new(2, 1);
        layer.weight = Tensor::from_vec(vec![1, 2], vec![1.0, 0.0]);
        let input = Tensor::from_vec(vec![2, 1], vec![1.0, 0.0]);
        let d_output = &Tensor::from_vec(vec![1, 1], vec![1.0]);
        layer.forward(&input);
        let (d_w, d_b, d_x) = layer.backward(d_output);
        assert_eq!(d_w.data, vec![1.0, 0.0]);
        assert_eq!(d_b.data, vec![1.0]);
        assert_eq!(d_x.data, vec![1.0, 0.0]);
    }

    #[test]
    fn test_backward_nontrivial() {
        let mut layer = LinearLayer::new(2, 2);
        layer.weight = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let input = Tensor::from_vec(vec![2, 1], vec![5.0, 6.0]);
        let d_output = Tensor::from_vec(vec![2, 1], vec![1.0, 1.0]);
        layer.forward(&input);
        let (d_w, d_b, d_x) = layer.backward(&d_output);
        assert_eq!(d_w.data, vec![5.0, 6.0, 5.0, 6.0]);
        assert_eq!(d_b.data, vec![1.0, 1.0]);
        assert_eq!(d_x.data, vec![4.0, 6.0]);
    }
}

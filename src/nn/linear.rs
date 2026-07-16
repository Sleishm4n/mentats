use std::io::{self, Read, Write};

use crate::{
    nn::Layer, tensor::Tensor,
    nn::init::{kaiming_normal, xavier_uniform},
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
            weight: xavier_uniform(out_features, in_features),
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
        self.weight.matmul(input).add(&self.bias)
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
        self.d_weight = Some(d_output.matmul(&self.input.as_ref().unwrap().transpose()));
        self.d_bias = Some(d_output.clone());
        let d_x = self.weight.transpose().matmul(d_output);
        d_x
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
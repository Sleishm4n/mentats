use std::{io::{self, Read, Write}, vec};

use crate::{
    nn::Layer,
    tensor::Tensor,
    utils::model_io::{read_u32, write_u32, write_u8, TAG_SAMPLER},
};
use rand::Rng;

pub struct GaussianSampler {
    pub latent_dim: usize,
    pub mu: Option<Tensor>,
    pub log_var: Option<Tensor>,
    pub eps: Option<Tensor>,
}

impl GaussianSampler {
    pub fn new(latent_dim: usize) -> Self {
        Self {
            latent_dim,
            mu: None,
            log_var: None,
            eps: None,
        }
    }

    pub fn load(reader: &mut dyn Read) -> io::Result<Self> {
        let latent_dim = read_u32(reader)? as usize;
        Ok(Self::new(latent_dim))
    }

    fn sample_stanard_normal(size: usize) -> Tensor {
        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(size);

        for _ in 0..size {
            let u1 = loop {
                let candidate: f32 = rng.gen_range(0.0..1.0);

                if candidate > 0.0 {
                    break candidate;
                }
            };
            let u2: f32 = rng.gen_range(0.0..1.0);

            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            data.push(r * theta.cos());
        }

        Tensor::from_vec(vec![size, 1], data)
    }
}

impl Layer for GaussianSampler {
    fn forward_pass(&mut self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape[0],
            self.latent_dim * 2,
            "input must be [latent_dim*2, batch_size"
        );

        let batch_size = input.shape[1];

        let mu = Tensor::from_vec(
            vec![self.latent_dim, batch_size],
            input.data[0..self.latent_dim * batch_size].to_vec(),
        );
        let log_var = Tensor::from_vec(
            vec![self.latent_dim, batch_size],
            input.data[0..self.latent_dim * batch_size].to_vec(),
        );

        let eps = Self::sample_stanard_normal(self.latent_dim * batch_size);
        let eps_reshaped = Tensor::from_vec(vec![self.latent_dim, batch_size], 
            eps.data.clone());

        let sigma = log_var.map(|x| (0.5 * x).exp());

        let z = mu.add(&sigma.zip_map(&eps_reshaped, |s, e| s * e));

        self.mu = Some(mu);
        self.log_var = Some(log_var);
        self.eps = Some(eps_reshaped);

        z
    }
    
    fn backward_pass(&mut self, d_output: &Tensor) -> Tensor {
        let log_var = self.log_var.as_ref().unwrap();
        let eps = self.eps.as_ref().unwrap();

        let d_mu = d_output.clone();

        let sigma = log_var.map(|x| (0.5 * x).exp());
        let d_sigma = d_output.zip_map(eps, |dz, e| dz * e);

        let d_log_var = d_sigma.zip_map(&sigma, |ds, s| ds * s * 0.5);

        let mut combined = d_mu.data.clone();
        combined.extend(&d_log_var.data);

        Tensor::from_vec(vec![self.latent_dim * 2, d_output.shape[1]], combined)
    }
    
    fn get_params(&self) -> Vec<Tensor> {
        vec![]
    }
    
    fn get_grads(&self) -> Vec<Tensor> {
        vec![]
    }
    
    fn save(&self, writer: &mut dyn Write) -> io::Result<()> {
        write_u8(writer, TAG_SAMPLER)?;
        write_u32(writer, self.latent_dim as u32)
    }
    
    fn set_params(&mut self, _params: Vec<Tensor>) {
        todo!()
    }

    
}

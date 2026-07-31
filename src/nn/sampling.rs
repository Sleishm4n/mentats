use crate::{
    nn::Layer,
    tensor::Tensor,
    utils::model_io::{read_u32, write_u32, write_u8, TAG_SAMPLER},
};
use rand::Rng;
use std::{
    io::{self, Read, Write},
    vec,
};

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

    pub fn sample_standard_normal(size: usize) -> Tensor {
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

    fn split_input(&self, input: &Tensor) -> (Tensor, Tensor, usize, bool) {
        match input.shape.len() {
            2 => {
                assert_eq!(input.shape[0], self.latent_dim * 2, "input must be [latent_dim*2, 1]");
                let mu = Tensor::from_vec(vec![self.latent_dim, 1], input.data[0..self.latent_dim].to_vec());
                let log_var = Tensor::from_vec(vec![self.latent_dim, 1], input.data[self.latent_dim..].to_vec());
                (mu, log_var, 1, false)
            }
            3 => {
                let batch_size = input.shape[0];
                assert_eq!(input.shape[1], self.latent_dim * 2, "input must be [batch, latent_dim*2, 1]");

                let mut mu_data = Vec::with_capacity(batch_size * self.latent_dim);
                let mut log_var_data = Vec::with_capacity(batch_size * self.latent_dim);

                for b in 0..batch_size {
                    let start = b * self.latent_dim * 2;
                    mu_data.extend_from_slice(&input.data[start..start + self.latent_dim]);
                    log_var_data.extend_from_slice(&input.data[start + self.latent_dim..start + self.latent_dim * 2]);
                }

                let mu = Tensor::from_vec(vec![batch_size, self.latent_dim, 1], mu_data);
                let log_var = Tensor::from_vec(vec![batch_size, self.latent_dim, 1], log_var_data);
                (mu, log_var, batch_size, true)
            }
            _ => panic!("GaussianSampler only supports 2D [latent_dim*2,1] or 3D [batch,latent_dim*2,1] inputs"),
        }
    }
}

impl Layer for GaussianSampler {
    fn forward_pass(&mut self, input: &Tensor) -> Tensor {
        let (mu, log_var, batch_size, batched) = self.split_input(input);

        let eps_flat = Self::sample_standard_normal(self.latent_dim * batch_size);
        let eps = if batched {
            Tensor::from_vec(vec![batch_size, self.latent_dim, 1], eps_flat.data)
        } else {
            Tensor::from_vec(vec![self.latent_dim, 1], eps_flat.data)
        };

        let sigma = log_var.map(|x| (0.5 * x).exp());
        let z = mu.add(&sigma.zip_map(&eps, |s, e| s * e));

        self.mu = Some(mu);
        self.log_var = Some(log_var);
        self.eps = Some(eps);

        z
    }

    fn backward_pass(&mut self, d_output: &Tensor) -> Tensor {
        let log_var = self.log_var.as_ref().unwrap();
        let eps = self.eps.as_ref().unwrap();

        let batched = d_output.shape.len() == 3;
        let batch_size = if batched { d_output.shape[0] } else { 1 };

        let d_mu = d_output.clone();
        let sigma = log_var.map(|x| (0.5 * x).exp());
        let d_sigma = d_output.zip_map(eps, |dz, e| dz * e);
        let d_log_var = d_sigma.zip_map(&sigma, |ds, s| ds * s * 0.5);

        if batched {
            let mut combined = Vec::with_capacity(batch_size * self.latent_dim * 2);
            for b in 0..batch_size {
                let start = b * self.latent_dim;
                combined.extend_from_slice(&d_mu.data[start..start + self.latent_dim]);
                combined.extend_from_slice(&d_log_var.data[start..start + self.latent_dim]);
            }
            Tensor::from_vec(vec![batch_size, self.latent_dim * 2, 1], combined)
        } else {
            let mut combined = d_mu.data.clone();
            combined.extend(&d_log_var.data);
            Tensor::from_vec(vec![self.latent_dim * 2, 1], combined)
        }
    }

    fn get_params(&self) -> Vec<Tensor> {
        vec![]
    }
    fn get_grads(&self) -> Vec<Tensor> {
        vec![]
    }
    fn set_params(&mut self, _params: Vec<Tensor>) {}

    fn save(&self, writer: &mut dyn Write) -> io::Result<()> {
        write_u8(writer, TAG_SAMPLER)?;
        write_u32(writer, self.latent_dim as u32)
    }
}

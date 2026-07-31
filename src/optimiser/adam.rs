use crate::{optimiser::Optimiser, tensor::Tensor};

pub struct Adam {
    pub alpha: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub t: u32,
    pub m: Vec<Tensor>,
    pub v: Vec<Tensor>,
}

impl Adam {
    pub fn new(alpha: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            alpha,
            beta1,
            beta2,
            epsilon,
            t: 0,
            m: vec![],
            v: vec![],
        }
    }
}

impl Optimiser for Adam {
    fn step(&mut self, params: &mut Vec<Tensor>, grads: &[Tensor]) {
        if self.m.is_empty() {
            self.m = params
                .iter()
                .map(|p| Tensor::new(p.shape.clone()))
                .collect();
            self.v = params
                .iter()
                .map(|p| Tensor::new(p.shape.clone()))
                .collect();
        }
        self.t += 1;
        for i in 0..params.len() {
            self.m[i] = self.m[i]
                .scale(self.beta1)
                .add(&grads[i].scale(1.0 - self.beta1));
            self.v[i] = self.v[i]
                .scale(self.beta2)
                .add(&grads[i].elementwise_square().scale(1.0 - self.beta2));
            let m_hat = self.m[i].scale(1.0 / (1.0 - self.beta1.powi(self.t as i32)));
            let v_hat = self.v[i].scale(1.0 / (1.0 - self.beta2.powi(self.t as i32)));
            params[i] = params[i].sub(
                &m_hat.zip_map(&v_hat.map(|x: f32| x.powf(0.5) + self.epsilon), |m, v| {
                    self.alpha * m / v
                }),
            );
        }
    }
}

//! The Adam optimiser
use crate::{optimiser::Optimiser, tensor::Tensor};

/// Adaptive Moment Estimation (Adam)
///
/// Keeps a running first moment (`m`, the mean of past gradients) and second
/// moment (`v`, the mean of the past squared gradients) per parameter, giving
/// each parameter its own effective learning rate. Both are bias-corrected using the
/// step counter `t`, which matters most in the first few steps while the
/// moments are warming up from zero
///
/// The moment vectors are allocated lazily on the first `[Optimiser::step]`
/// matching the shapes of the parameters passed in
///
/// # Example
///
/// ```
/// use mentats::optimiser::adam::Adam;
///
/// // Common defaults: lr 1e-3, beta1 0.9, beta2 0.999, eps 1e-8
/// let mut optimiser = Adam::new(0.001, 0.9, 0.999, 1e-3);
/// ```
pub struct Adam {
    /// Learning ratge (step size)
    pub alpha: f32,
    /// Exponential decay rate for the first moment estimate
    pub beta1: f32,
    /// Exponential decay rate for the second moment estimate
    pub beta2: f32,
    /// Small constant added to the denominator for numerical stability
    pub epsilon: f32,
    /// Number of steps taken so far, used for bias correction
    pub t: u32,
    /// First moment estimate, one tensor per parameter
    pub m: Vec<Tensor>,
    /// Second moment estimate, one tensor per parameter
    pub v: Vec<Tensor>,
}

impl Adam {
    /// Creates an optimiser with the given hyperparameters
    ///
    /// The moment vectors start empty and are sized on the first step
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
    /// Applies one Adam update in place
    ///
    /// # Panics
    ///
    /// Panics if `grads` is shorter than `params`, or if a gradients
    /// shape doesn't match its parameter
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

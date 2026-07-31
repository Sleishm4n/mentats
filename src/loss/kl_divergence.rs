use crate::tensor::Tensor;

pub fn kl_divergence(mu: &Tensor, log_var: &Tensor) -> f32 {
    assert_eq!(
        mu.shape, log_var.shape,
        "mu and log_var must have the same shape"
    );

    let batch_size = if mu.shape.len() == 3 { mu.shape[0] } else { 1 };
    let mut total_kl: f32 = 0.0;

    for i in 0..mu.data.len() {
        let m = mu.data[i];
        let lv = log_var.data[i];
        let var = lv.exp();
        total_kl += -0.5 * (1.0 + lv - m * m - var);
    }

    total_kl / batch_size as f32
}

pub fn d_kl_divergence_mu(mu: &Tensor) -> Tensor {
    let batch_size = if mu.shape.len() == 3 { mu.shape[0] } else { 1 };
    mu.map(|m| m / batch_size as f32)
}

pub fn d_kl_divergence_log_var(log_var: &Tensor) -> Tensor {
    let batch_size = if log_var.shape.len() == 3 {
        log_var.shape[0]
    } else {
        1
    };
    log_var.map(|lv| 0.5 * (lv.exp() - 1.0) / batch_size as f32)
}

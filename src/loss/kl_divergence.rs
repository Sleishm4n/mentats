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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "mu and log_var must have the same shape")]
    fn test_kl_panics_on_shape_mismatch() {
        let mu = Tensor::new(vec![3, 2]);
        let log_var = Tensor::new(vec![2, 3]);

        let _ = kl_divergence(&mu, &log_var);
    }

    #[test]
    fn test_kl_correct_value() {
        // mu=1.0, log_var=0.0 (var=1.0):
        // KL = -0.5 * (1 + 0 - 1 - 1) = -0.5 * (-1) = 0.5
        let mu = Tensor::from_vec(vec![1, 1], vec![1.0]);
        let log_var = Tensor::from_vec(vec![1, 1], vec![0.0]);

        let kl = kl_divergence(&mu, &log_var);

        assert!((kl - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_d_kl_divergence_mu_matches_numerical_gradient() {
        let mu = Tensor::from_vec(vec![2, 2, 1], vec![0.3, -0.7, 1.1, 0.2]);
        let log_var = Tensor::from_vec(vec![2, 2, 1], vec![0.1, -0.2, 0.05, 0.3]);

        let analytical = d_kl_divergence_mu(&mu);

        let epsilon = 1e-4;
        for i in 0..mu.data.len() {
            let mut plus = mu.clone();
            plus.data[i] += epsilon;
            let mut minus = mu.clone();
            minus.data[i] -= epsilon;

            let kl_plus = kl_divergence(&plus, &log_var);
            let kl_minus = kl_divergence(&minus, &log_var);

            let numerical = (kl_plus - kl_minus) / (2.0 * epsilon);

            assert!(
                (analytical.data[i] - numerical).abs() < 1e-3,
                "index {i}: analytical {} vs numerical {}",
                analytical.data[i],
                numerical
            );
        }
    }

    #[test]
    fn test_d_kl_divergence_log_var_matches_numerical_gradient() {
        let mu = Tensor::from_vec(vec![2, 2, 1], vec![0.3, -0.7, 1.1, 0.2]);
        let log_var = Tensor::from_vec(vec![2, 2, 1], vec![0.1, -0.2, 0.05, 0.3]);

        let analytical = d_kl_divergence_log_var(&log_var);

        let epsilon = 1e-4;
        for i in 0..log_var.data.len() {
            let mut plus = log_var.clone();
            plus.data[i] += epsilon;
            let mut minus = log_var.clone();
            minus.data[i] -= epsilon;

            let kl_plus = kl_divergence(&mu, &plus);
            let kl_minus = kl_divergence(&mu, &minus);

            let numerical = (kl_plus - kl_minus) / (2.0 * epsilon);

            assert!(
                (analytical.data[i] - numerical).abs() < 1e-3,
                "index {i}: analytical {} vs numerical {}",
                analytical.data[i],
                numerical
            );
        }
    }
}

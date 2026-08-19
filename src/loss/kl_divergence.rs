//! KL divergence between the encoder's latent distribution and a standard
//! normal prior, with a free-bits floor
//!
//! This is the regularisation half of the VAE objective. All three public
//! functions accept both unbatched (`[latent_dim, 1]`) and batched
//! (`[batch, latent_dim, 1]`) `mu` / `log_var` tensors, and average
//! over the batch
//!
//! The free-bits scheme ([`FREE_BITS_TAU`]) is what prevents posterior
//! collapse: without it the cheapest way to cut the KL term is to drive
//! every latent dim to the prior and let decoder ignore `z`
use crate::tensor::Tensor;

/// Free bits threshold, in nats per latent dimension. Dimensions whose
/// KL contribution falls below this are not penalized, giving the
/// encoder slack to use them without paying a KL cost — this removes
/// the incentive for the decoder to find a solution that ignores z entirely.
pub const FREE_BITS_TAU: f32 = 0.5;

/// Computes raw (unclamped) KL per latent dimension, averaged over the batch.
/// Returns a Vec<f32> of length latent_dim.
fn per_dim_kl(mu: &Tensor, log_var: &Tensor) -> Vec<f32> {
    let (batch_size, latent_dim) = if mu.shape.len() == 3 {
        (mu.shape[0], mu.shape[1])
    } else {
        (1, mu.shape[0])
    };

    let mut per_dim = vec![0.0f32; latent_dim];

    for b in 0..batch_size {
        for (d, item) in per_dim.iter_mut().enumerate().take(latent_dim) {
            let idx = b * latent_dim + d;
            let m = mu.data[idx];
            let lv = log_var.data[idx];
            let var = lv.exp();
            *item += -0.5 * (1.0 + lv - m * m - var);
        }
    }

    for v in per_dim.iter_mut() {
        *v /= batch_size as f32;
    }

    per_dim
}

/// 1.0 for dimensions currently above the free bits threshold (gradient flows),
/// 0.0 for dimensions at or below it (gradient clamped to zero).
fn free_bits_mask(mu: &Tensor, log_var: &Tensor) -> Vec<f32> {
    per_dim_kl(mu, log_var)
        .iter()
        .map(|&k| if k > FREE_BITS_TAU { 1.0 } else { 0.0 })
        .collect()
}

/// KL divergence from `N(mu, exp(log_var))` to `N(0, 1)`, summed over latent
/// dimensions and averaged over the batch
///
/// Each dimension's contribution has [`FREE_BITS_TAU`] subtracted and is
/// floored at zero, so dimensions already close to the prior contribute
/// nothing
///
/// # Panics
///
/// Panics if `mu` and `log_var` have different shapes
pub fn kl_divergence(mu: &Tensor, log_var: &Tensor) -> f32 {
    assert_eq!(
        mu.shape, log_var.shape,
        "mu and log_var must have the same shape"
    );

    per_dim_kl(mu, log_var)
        .iter()
        .map(|&k| (k - FREE_BITS_TAU).max(0.0))
        .sum()
}

/// Gradient of [`kl_divergence`] with respect to `mu`
///
/// Note: now takes log_var as well as mu, since the free-bits mask depends
/// on both — this is a signature change from the pre-free-bits version.
pub fn d_kl_divergence_mu(mu: &Tensor, log_var: &Tensor) -> Tensor {
    let (batch_size, latent_dim) = if mu.shape.len() == 3 {
        (mu.shape[0], mu.shape[1])
    } else {
        (1, mu.shape[0])
    };
    let mask = free_bits_mask(mu, log_var);

    let mut data = Vec::with_capacity(mu.data.len());
    for b in 0..batch_size {
        for (d, &mask_value) in mask.iter().enumerate().take(latent_dim) {
            let idx = b * latent_dim + d;
            data.push(mu.data[idx] * mask_value / batch_size as f32);
        }
    }

    Tensor::from_vec(mu.shape.clone(), data)
}

/// Gradient of [`kl_divergence`] with respect to `log_var`
///
/// Note: signature change — mu is now required alongside log_var for the mask.
pub fn d_kl_divergence_log_var(mu: &Tensor, log_var: &Tensor) -> Tensor {
    let (batch_size, latent_dim) = if log_var.shape.len() == 3 {
        (log_var.shape[0], log_var.shape[1])
    } else {
        (1, log_var.shape[0])
    };
    let mask = free_bits_mask(mu, log_var);

    let mut data = Vec::with_capacity(log_var.data.len());
    for b in 0..batch_size {
        for (d, &mask_value) in mask.iter().enumerate().take(latent_dim) {
            let idx = b * latent_dim + d;
            let lv = log_var.data[idx];
            data.push(0.5 * (lv.exp() - 1.0) * mask_value / batch_size as f32);
        }
    }

    Tensor::from_vec(log_var.shape.clone(), data)
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
        // raw KL = 0.5, but free bits threshold is 0.5, so the clamped result is 0.0.
        let mu = Tensor::from_vec(vec![1, 1], vec![1.0]);
        let log_var = Tensor::from_vec(vec![1, 1], vec![0.0]);

        let kl = kl_divergence(&mu, &log_var);

        assert!((kl - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_d_kl_divergence_mu_matches_numerical_gradient() {
        let mu = Tensor::from_vec(vec![2, 2, 1], vec![0.3, -0.7, 1.1, 0.2]);
        let log_var = Tensor::from_vec(vec![2, 2, 1], vec![0.1, -0.2, 0.05, 0.3]);

        let analytical = d_kl_divergence_mu(&mu, &log_var);

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

        let analytical = d_kl_divergence_log_var(&mu, &log_var);

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

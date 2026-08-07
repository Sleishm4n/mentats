use crate::tensor::Tensor;

/// Splits a 3D tensor of shape `[batch, latent_dim * 2, 1]` into `mu` and `log_var`,
/// both of shape `[batch, latent_dim, 1]`.
pub fn split_mu_log_var(mu_log_var: &Tensor, latent_dim: usize) -> (Tensor, Tensor) {
    let batch_size = mu_log_var.shape[0];
    let mut mu_data = Vec::with_capacity(batch_size * latent_dim);
    let mut log_var_data = Vec::with_capacity(batch_size * latent_dim);

    let stride = latent_dim * 2;
    for i in 0..batch_size {
        let sample_start = i * stride;
        let mu_end = sample_start + latent_dim;
        let log_var_end = sample_start + stride;

        mu_data.extend_from_slice(&mu_log_var.data[sample_start..mu_end]);
        log_var_data.extend_from_slice(&mu_log_var.data[mu_end..log_var_end]);
    }

    let mu = Tensor::from_vec(vec![batch_size, latent_dim, 1], mu_data);
    let log_var = Tensor::from_vec(vec![batch_size, latent_dim, 1], log_var_data);

    (mu, log_var)
}

/// Combines `d_mu` and `d_log_var` (each `[batch, latent_dim, 1]`) back into
/// a contiguous `[batch, latent_dim * 2, 1]` gradient tensor.
pub fn combine_kl_grads(d_mu: &Tensor, d_log_var: &Tensor, latent_dim: usize) -> Tensor {
    let batch_size = d_mu.shape[0];
    let mut combined_data = Vec::with_capacity(batch_size * latent_dim * 2);

    for i in 0..batch_size {
        let mu_start = i * latent_dim;
        let mu_end = mu_start + latent_dim;

        let log_var_start = i * latent_dim;
        let log_var_end = log_var_start + latent_dim;

        combined_data.extend_from_slice(&d_mu.data[mu_start..mu_end]);
        combined_data.extend_from_slice(&d_log_var.data[log_var_start..log_var_end]);
    }

    Tensor::from_vec(vec![batch_size, latent_dim * 2, 1], combined_data)
}

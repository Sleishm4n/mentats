use mentats::data::mnist::{load_images, load_labels};
use mentats::loss::kl_divergence::{d_kl_divergence_log_var, d_kl_divergence_mu};
use mentats::loss::{cross_entropy::binary_cross_entropy, kl_divergence::kl_divergence};
use mentats::nn::activation::{ActivationKind::Relu, ActivationLayer};
use mentats::nn::linear::LinearLayer;
use mentats::nn::network::Network;
use mentats::nn::sampling::GaussianSampler;
use mentats::nn::Layer;
use mentats::optimiser::adam::Adam;
use mentats::tensor::Tensor;
use mentats::utils::batch::BatchIterator;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::Path;

/// Splits a 3D tensor of shape `[batch, latent_dim * 2, 1]` into `mu` and `log_var`,
/// both of shape `[batch, latent_dim, 1]`.
fn split_mu_log_var(mu_log_var: &Tensor, latent_dim: usize) -> (Tensor, Tensor) {
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
fn combine_kl_grads(d_mu: &Tensor, d_log_var: &Tensor, latent_dim: usize) -> Tensor {
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

/// Stacks a slice of image tensors into a single 3D batch tensor of shape
/// `[batch_size, 784, 1]`.
fn stack_batch(images: &[Tensor], indices: &[usize]) -> Tensor {
    let batch_size = indices.len();
    let feature_dim = images[0].data.len();

    let mut batch_data = Vec::with_capacity(batch_size * feature_dim);

    for &idx in indices {
        batch_data.extend_from_slice(&images[idx].data);
    }

    Tensor::from_vec(vec![batch_size, feature_dim, 1], batch_data)
}

/// Converts a normalized pixel value into an 8-bit grayscale value.
fn to_u8_gray(v: f32, apply_sigmoid: bool) -> u8 {
    let p = if apply_sigmoid {
        1.0 / (1.0 + (-v).exp())
    } else {
        v
    }
    .clamp(0.0, 1.0);

    (p * 255.0).round() as u8
}

/// Saves a single MNIST tensor as a PGM image.
fn save_mnist_tensor_pgm(
    tensor: &Tensor,
    path: &Path,
    apply_sigmoid: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if tensor.data.len() != 28 * 28 {
        return Err(format!(
            "Expected 784 values for MNIST images, got {}",
            tensor.data.len()
        )
        .into());
    }

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "P2")?;
    writeln!(writer, "28 28")?;
    writeln!(writer, "255")?;

    for (i, v) in tensor.data.iter().enumerate() {
        let px = to_u8_gray(*v, apply_sigmoid);

        if i % 28 == 27 {
            writeln!(writer, "{}", px)?;
        } else {
            write!(writer, "{} ", px)?;
        }
    }

    Ok(())
}

/// Saves multiple MNIST tensors as a square grid.
///
/// For example, 16 images with `grid_size = 4` produces a 4x4 grid
/// with an output resolution of 112x112 pixels.
fn save_mnist_grid_pgm(
    tensors: &[Tensor],
    path: &Path,
    grid_size: usize,
    apply_sigmoid: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if tensors.len() != grid_size * grid_size {
        return Err(format!(
            "Expected {} tensors for a {}x{} grid, got {}",
            grid_size * grid_size,
            grid_size,
            grid_size,
            tensors.len()
        )
        .into());
    }

    for tensor in tensors {
        if tensor.data.len() != 28 * 28 {
            return Err(format!(
                "Expected 784 values per MNIST image, got {}",
                tensor.data.len()
            )
            .into());
        }
    }

    let image_size = 28;
    let output_size = grid_size * image_size;

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "P2")?;
    writeln!(writer, "{} {}", output_size, output_size)?;
    writeln!(writer, "255")?;

    for grid_y in 0..grid_size {
        for y in 0..image_size {
            for grid_x in 0..grid_size {
                let tensor_idx = grid_y * grid_size + grid_x;
                let tensor = &tensors[tensor_idx];

                for x in 0..image_size {
                    let pixel_idx = y * image_size + x;
                    let px = to_u8_gray(tensor.data[pixel_idx], apply_sigmoid);

                    write!(writer, "{} ", px)?;
                }
            }

            writeln!(writer)?;
        }
    }

    Ok(())
}

fn main() {
    let latent_dim = 32;
    let batch_size = 64;
    let epochs = 61;

    // --------------------------------------------------
    // Encoder
    // --------------------------------------------------

    let mut encoder = Network::new(vec![
        Box::new(LinearLayer::new_rand(784, 512)),
        Box::new(ActivationLayer::new(Relu)),
        Box::new(LinearLayer::new_rand(512, 256)),
        Box::new(ActivationLayer::new(Relu)),
        Box::new(LinearLayer::new_rand(256, latent_dim * 2)),
    ]);

    // --------------------------------------------------
    // Gaussian sampler
    // --------------------------------------------------

    let mut sampler = GaussianSampler::new(latent_dim);

    // --------------------------------------------------
    // Decoder
    // --------------------------------------------------

    let mut decoder = Network::new(vec![
        Box::new(LinearLayer::new_rand(latent_dim, 256)),
        Box::new(ActivationLayer::new(Relu)),
        Box::new(LinearLayer::new_rand(256, 512)),
        Box::new(ActivationLayer::new(Relu)),
        Box::new(LinearLayer::new_rand(512, 784)),
    ]);

    // --------------------------------------------------
    // Optimisers
    // --------------------------------------------------

    let mut encoder_opt = Adam::new(0.001, 0.9, 0.999, 1e-8);
    let mut decoder_opt = Adam::new(0.001, 0.9, 0.999, 1e-8);

    // --------------------------------------------------
    // Dataset
    // --------------------------------------------------

    let images = load_images("data/mnist/train-images.idx3-ubyte");

    // --------------------------------------------------
    // Output directory
    // --------------------------------------------------

    let output_dir = Path::new("outputs/vae_mnist_mk2");

    create_dir_all(output_dir).expect("failed to create outputs/vae_mnist_mk2 directory");

    // Save the fixed original image that we will reconstruct
    // throughout training.
    save_mnist_tensor_pgm(&images[0], &output_dir.join("original_0.pgm"), false)
        .expect("failed to save original sample image");

    println!("VAE Training on MNIST");
    println!("========================================\n");

    // --------------------------------------------------
    // Training
    // --------------------------------------------------

    for epoch in 0..epochs {
        // KL annealing:
        //
        // Epoch 0 -> beta = 0.00000
        // Epoch 1 -> beta = 0.00020
        // ...
        // Epoch 5 -> beta = 0.00100
        // Epoch 5+ -> beta = 0.00100
        let beta_max = 1.0;
        let total_warmup_epochs = 20.0;
        let beta = (epoch as f32 / total_warmup_epochs).min(1.0) * beta_max;

        let mut total_loss = 0.0;
        let mut total_recon_loss = 0.0;
        let mut total_kl_loss = 0.0;
        let mut batch_count = 0;

        let mut batch_iter = BatchIterator::new(images.len(), batch_size, true);

        while let Some(batch_indices) = batch_iter.next_batch() {
            // --------------------------------------------------
            // 1. Stack images
            // --------------------------------------------------

            let x_batch = stack_batch(&images, &batch_indices);

            // --------------------------------------------------
            // 2. Encoder forward pass
            //
            // [batch, 784, 1]
            //       ↓
            // [batch, latent_dim * 2, 1]
            //
            // First half = mu
            // Second half = log_var
            // --------------------------------------------------

            let mu_log_var = encoder.forward(&x_batch);

            let (mu, log_var) = split_mu_log_var(&mu_log_var, latent_dim);

            // --------------------------------------------------
            // 3. Reparameterisation / sampling
            // --------------------------------------------------

            let z = sampler.forward_pass(&mu_log_var);

            // --------------------------------------------------
            // 4. Decoder forward pass
            // --------------------------------------------------

            let x_recon = decoder.forward(&z);

            // --------------------------------------------------
            // 5. Losses
            // --------------------------------------------------

            let recon_loss = binary_cross_entropy(&x_recon, &x_batch);

            let kl_loss = kl_divergence(&mu, &log_var);

            // --------------------------------------------------
            // 6. Decoder gradient
            // --------------------------------------------------

            let n_features = 784.0;
            let current_batch_size = batch_indices.len() as f32;

            let d_recon = x_recon
                .zip_map(&x_batch, |logit, target| {
                    (1.0 / (1.0 + (-logit).exp())) - target
                })
                .scale(1.0 / (n_features * current_batch_size));

            let d_z = decoder.backward(&d_recon);

            // --------------------------------------------------
            // 7. Sampler gradient
            // --------------------------------------------------

            let d_mu_log_var_recon = sampler.backward_pass(&d_z);

            // --------------------------------------------------
            // 8. KL gradients
            // --------------------------------------------------

            // Scale by 1 / latent_dim so that the KL gradients
            // match the normalized KL loss scale.

            let d_mu_kl = d_kl_divergence_mu(&mu, &log_var).scale(beta);
            let d_log_var_kl = d_kl_divergence_log_var(&mu, &log_var).scale(beta);
            let d_mu_log_var_kl = combine_kl_grads(&d_mu_kl, &d_log_var_kl, latent_dim);

            // --------------------------------------------------
            // 9. Combine encoder gradients
            // --------------------------------------------------

            let d_mu_log_var_total = d_mu_log_var_recon.add(&d_mu_log_var_kl);

            // --------------------------------------------------
            // 10. Encoder backward pass
            // --------------------------------------------------

            let _ = encoder.backward(&d_mu_log_var_total);

            // --------------------------------------------------
            // 11. Update parameters
            // --------------------------------------------------

            encoder.update(&mut encoder_opt);
            decoder.update(&mut decoder_opt);

            // --------------------------------------------------
            // Track losses
            // --------------------------------------------------

            let batch_total = recon_loss + beta * kl_loss;

            total_loss += batch_total;
            total_recon_loss += recon_loss;
            total_kl_loss += kl_loss;

            batch_count += 1;
        }

        // --------------------------------------------------
        // Epoch statistics
        // --------------------------------------------------

        let avg_loss = total_loss / batch_count as f32;

        let avg_recon_loss = total_recon_loss / batch_count as f32;

        let avg_kl_loss = total_kl_loss / batch_count as f32;

        println!(
            "Epoch {:02} | Loss: {:.4} | Recon: {:.4} | KL: {:.4} | Beta: {:.5}",
            epoch, avg_loss, avg_recon_loss, avg_kl_loss, beta,
        );

        // --------------------------------------------------
        // Epoch preview evaluation
        // --------------------------------------------------

        if epoch % 5 == 0 {
            // --------------------------------------------------
            // Reconstruction
            // --------------------------------------------------

            // Always reconstruct the same image so that
            // improvements can be compared directly between
            // epochs.

            let x_single = &images[0];

            let mu_log_var_single = encoder.forward(x_single);

            let z_single = sampler.forward_pass(&mu_log_var_single);

            let x_recon_single = decoder.forward(&z_single);

            save_mnist_tensor_pgm(
                &x_recon_single,
                &output_dir.join(format!("recon_epoch_{:02}.pgm", epoch)),
                true,
            )
            .expect("failed to save reconstruction image");

            // --------------------------------------------------
            // Random generation
            // --------------------------------------------------

            // Generate 16 samples from the standard normal prior.

            let mut generated_samples = Vec::with_capacity(16);

            for _ in 0..16 {
                let random_z = GaussianSampler::sample_standard_normal(latent_dim);

                let generated = decoder.forward(&random_z);

                generated_samples.push(generated);
            }

            // --------------------------------------------------
            // Save generated samples as a 4x4 grid
            // --------------------------------------------------

            save_mnist_grid_pgm(
                &generated_samples,
                &output_dir.join(format!("samples_epoch_{:02}.pgm", epoch)),
                4,
                true,
            )
            .expect("failed to save generated sample grid");

            println!("  Saved reconstruction + 16 generated samples");
        }
    }

    // --------------------------------------------------
    // Training complete
    // --------------------------------------------------

    println!("\nVAE training complete!");
    println!("Saved outputs in outputs/vae_mnist/");

    // --------------------------------------------------
    // Export latent space
    // --------------------------------------------------

    let labels = load_labels("data/mnist/train-labels.idx1-ubyte");

    export_latent_space_csv(
        &mut encoder,
        &images,
        &labels,
        2000,
        latent_dim,
        "outputs/vae_mnist/latent_space.csv",
    )
    .expect("failed to export latent space CSV");

    println!("Saved outputs and latent CSV in outputs/vae_mnist/");
}

/// Exports the encoder's latent means to CSV.
///
/// Each row contains:
///
/// z0, z1, ..., z31, label
///
/// The label is only included so that the resulting CSV can be
/// coloured/grouped when visualising the latent space. It is NOT
/// used during VAE training.
fn export_latent_space_csv(
    encoder: &mut Network,
    images: &[Tensor],
    labels: &[u8],
    num_samples: usize,
    latent_dim: usize,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);

    // Header
    for i in 0..latent_dim {
        write!(writer, "z{},", i)?;
    }

    writeln!(writer, "label")?;

    let count = num_samples.min(images.len());

    for idx in 0..count {
        let mu_log_var = encoder.forward(&images[idx]);

        // Extract only the latent mean vector.
        let mu = &mu_log_var.data[0..latent_dim];

        for &val in mu.iter() {
            write!(writer, "{:.6},", val)?;
        }

        writeln!(writer, "{}", labels[idx])?;
    }

    println!("Exported {} latent samples to {}", count, output_path);

    Ok(())
}

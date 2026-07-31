use rust_ml::data::mnist::{load_images, load_labels};
use rust_ml::loss::{cross_entropy::binary_cross_entropy, kl_divergence::kl_divergence};
use rust_ml::nn::activation::{ActivationKind::Relu, ActivationLayer};
use rust_ml::nn::linear::LinearLayer;
use rust_ml::nn::network::Network;
use rust_ml::nn::sampling::GaussianSampler;
use rust_ml::nn::Layer;
use rust_ml::optimiser::adam::Adam;
use rust_ml::tensor::Tensor;
use rust_ml::utils::batch::BatchIterator;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::Path;

fn predict_from_logits(logits: &Tensor) -> (u8, f32) {
    let max_logit = logits.tensor_max();
    let exps = logits.map(|x| (x - max_logit).exp());
    let sum: f32 = exps.data.iter().sum();

    let mut best_idx = 0usize;
    let mut best_prob = 0.0_f32;
    for (i, e) in exps.data.iter().enumerate() {
        let p = e / sum;
        if p > best_prob {
            best_prob = p;
            best_idx = i;
        }
    }

    (best_idx as u8, best_prob)
}

fn load_classifier() -> Network {
    let checkpoint_path = "checkpoints/mnist_classifier.rmlc";

    match Network::load(checkpoint_path) {
        Ok(network) => network,
        Err(err) => panic!(
            "Failed to load the mnist classifier checkpoint from {} ({}) \
            Run the mnist example first to train and save.",
            checkpoint_path, err
        ),
    }
}

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

/// Stacks a slice of 2D/1D image tensors into a single 3D batch tensor of shape `[batch_size, 784, 1]`.
fn stack_batch(images: &[Tensor], indices: &[usize]) -> Tensor {
    let batch_size = indices.len();
    let feature_dim = images[0].data.len();
    let mut batch_data = Vec::with_capacity(batch_size * feature_dim);

    for &idx in indices {
        batch_data.extend_from_slice(&images[idx].data);
    }

    Tensor::from_vec(vec![batch_size, feature_dim, 1], batch_data)
}

fn to_u8_gray(v: f32, apply_sigmoid: bool) -> u8 {
    let p = if apply_sigmoid {
        1.0 / (1.0 + (-v).exp())
    } else {
        v
    }
    .clamp(0.0, 1.0);

    (p * 255.0).round() as u8
}

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

fn main() {
    let latent_dim = 32;
    let batch_size = 64;
    let epochs = 30;
    let total_warmup_epochs = 5.0;

    let mut encoder = Network::new(vec![
        Box::new(LinearLayer::new_rand(784, 512)),
        Box::new(ActivationLayer::new(Relu)),
        Box::new(LinearLayer::new_rand(512, 256)),
        Box::new(ActivationLayer::new(Relu)),
        Box::new(LinearLayer::new_rand(256, latent_dim * 2)),
    ]);

    let mut sampler = GaussianSampler::new(latent_dim);

    let mut decoder = Network::new(vec![
        Box::new(LinearLayer::new_rand(latent_dim, 256)),
        Box::new(ActivationLayer::new(Relu)),
        Box::new(LinearLayer::new_rand(256, 512)),
        Box::new(ActivationLayer::new(Relu)),
        Box::new(LinearLayer::new_rand(512, 784)),
    ]);

    let mut encoder_opt = Adam::new(0.001, 0.9, 0.999, 1e-8);
    let mut decoder_opt = Adam::new(0.001, 0.9, 0.999, 1e-8);

    let images = load_images("data/mnist/train-images.idx3-ubyte");
    let mut classifier = load_classifier();

    let output_dir = Path::new("outputs/vae_mnist");
    create_dir_all(output_dir).expect("failed to create outputs/vae_mnist directory");

    save_mnist_tensor_pgm(&images[0], &output_dir.join("original_0.pgm"), false)
        .expect("failed to save original sample image");

    println!("VAE Training on MNIST");
    println!("========================================\n");

    for epoch in 0..epochs {
        // Dynamic KL Annealing Warmup factor (0.0001 -> 0.001)
        let beta = (epoch as f32 / total_warmup_epochs).min(1.0) * 0.001;

        let mut total_loss = 0.0;
        let mut total_recon_loss = 0.0;
        let mut total_kl_loss = 0.0;
        let mut batch_count = 0;

        let mut batch_iter = BatchIterator::new(images.len(), batch_size, true);

        while let Some(batch_indices) = batch_iter.next_batch() {
            // 1. Stack raw images into a [batch, 784, 1] tensor
            let x_batch = stack_batch(&images, &batch_indices);

            // 2. Encoder Forward Pass -> [batch, latent_dim * 2, 1]
            let mu_log_var = encoder.forward(&x_batch);
            let (mu, log_var) = split_mu_log_var(&mu_log_var, latent_dim);

            // 3. Sampling -> [batch, latent_dim, 1]
            let z = sampler.forward_pass(&mu_log_var);

            // 4. Decoder Forward Pass -> [batch, 784, 1]
            let x_recon = decoder.forward(&z);

            // 5. Compute Batched Losses
            let recon_loss = binary_cross_entropy(&x_recon, &x_batch);
            let kl_loss = kl_divergence(&mu, &log_var);

            // 6. Decoder Gradient
            let n_features = 784.0;
            let current_batch_size = batch_indices.len() as f32;
            let d_recon = x_recon
                .zip_map(&x_batch, |logit, t| (1.0 / (1.0 + (-logit).exp())) - t)
                .scale(1.0 / (n_features * current_batch_size));

            let d_z = decoder.backward(&d_recon);

            // 7. Sampler Gradient
            let d_mu_log_var_recon = sampler.backward_pass(&d_z);

            // 8. KL Gradients
            // Scale by (1 / latent_dim) so KL gradients match normalized KL loss scale
            let d_mu_kl = mu.scale(1.0 / latent_dim as f32);
            let d_log_var_kl = log_var.map(|lv| 0.5 * (lv.exp() - 1.0) / (latent_dim as f32));
            let d_mu_log_var_kl = combine_kl_grads(&d_mu_kl, &d_log_var_kl, latent_dim)
                .scale(beta / current_batch_size);

            // 9. Total Encoder Input Gradient
            let d_mu_log_var_total = d_mu_log_var_recon.add(&d_mu_log_var_kl);

            // 10. Encoder Backward Pass
            let _ = encoder.backward(&d_mu_log_var_total);

            // 11. Parameter Updates
            encoder.update(&mut encoder_opt);
            decoder.update(&mut decoder_opt);

            let batch_total = recon_loss + beta * kl_loss;
            total_loss += batch_total;
            total_recon_loss += recon_loss;
            total_kl_loss += kl_loss;
            batch_count += 1;
        }

        println!(
            "Epoch {}: Loss = {:.4} (Recon: {:.4}, KL: {:.4})",
            epoch,
            total_loss / batch_count as f32,
            total_recon_loss / batch_count as f32,
            total_kl_loss / batch_count as f32,
        );

        // Epoch preview evaluation (Single Sample 2D compatibility path)
        let x_single = &images[0];
        let mu_log_var_single = encoder.forward(x_single);
        let z_single = sampler.forward_pass(&mu_log_var_single);
        let x_recon_single = decoder.forward(&z_single);

        if epoch % 5 == 0 {
            save_mnist_tensor_pgm(
                &x_recon_single,
                &output_dir.join(format!("recon_epoch_{:02}.pgm", epoch)),
                true,
            )
            .expect("failed to save reconstruction image");
        }

        let random_z = GaussianSampler::sample_standard_normal(latent_dim);
        let generated = decoder.forward(&random_z);

        let generated_prob = generated.map(|v| 1.0 / (1.0 + (-v).exp()));
        let generated_logits = classifier.forward(&generated_prob);
        let (pred_label, pred_confidence) = predict_from_logits(&generated_logits);

        if epoch % 5 == 0 {
            save_mnist_tensor_pgm(
                &generated,
                &output_dir.join(format!("sample_epoch_{:02}.pgm", epoch)),
                true,
            )
            .expect("failed to save generated image");
        }

        println!(
            "Generated sample guess: {} ({:.1}% confidence)",
            pred_label,
            pred_confidence * 100.0
        );
    }

    println!("\nVAE training complete!");
    println!("Saved outputs in outputs/vae_mnist/");

    let labels = load_labels("data/mnist/train-labels.idx1-ubyte"); // Load labels if not already loaded

    export_latent_space_csv(
        &mut encoder,
        &images,
        &labels,
        2000, // Export 2,000 points
        latent_dim,
        "outputs/vae_mnist/latent_space.csv",
    )
    .expect("failed to export latent space CSV");

    println!("Saved outputs and latent CSV in outputs/vae_mnist/");
}

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

    // Header: z0, z1, ..., z31, label
    for i in 0..latent_dim {
        write!(writer, "z{},", i)?;
    }
    writeln!(writer, "label")?;

    let count = num_samples.min(images.len());
    for idx in 0..count {
        let mu_log_var = encoder.forward(&images[idx]);

        // Extract just the 32D mean vector (mu)
        let mu = &mu_log_var.data[0..latent_dim];

        for &val in mu.iter() {
            write!(writer, "{:.6},", val)?;
        }
        writeln!(writer, "{}", labels[idx])?;
    }

    println!("Exported {} latent samples to {}", count, output_path);
    Ok(())
}

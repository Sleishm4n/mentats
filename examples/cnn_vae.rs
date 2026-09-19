use mentats::data::mnist::load_images;
use mentats::loss::cross_entropy::binary_cross_entropy;
use mentats::loss::kl_divergence::{d_kl_divergence_log_var, d_kl_divergence_mu, kl_divergence};
use mentats::nn::Layer;
use mentats::nn::{
    activation::{ActivationKind::Relu, ActivationLayer},
    conv::Conv2DLayer,
    flatten::FlattenLayer,
    linear::LinearLayer,
    network::Network,
    padding::Pad2DLayer,
    pooling::MaxPool2DLayer,
    reshape::ReshapeLayer,
    sampling::GaussianSampler,
    upsample::Upsample2DLayer,
};
use mentats::optimiser::adam::Adam;
use mentats::utils::image::{save_mnist_tensor_pgm, save_pgm_grid_4x4};
use mentats::Tensor;
use std::fs::create_dir_all;
use std::path::Path;
use std::time::Instant;

fn main() {
    let epochs: usize = 15;
    let lr = 0.001;

    let output_dir = Path::new("outputs/cnn_vae");
    create_dir_all(output_dir).expect("failed to create output directory");
    let latent_dim: usize = 10;

    let mut encoder = Network::new(vec![
        Box::new(ReshapeLayer::new(vec![1, 28, 28])),
        Box::new(Conv2DLayer::new_kaiming(1, 16, (3, 3))),
        Box::new(ActivationLayer::new(Relu)),
        Box::new(MaxPool2DLayer::new_stand()),
        Box::new(FlattenLayer::new()),
        Box::new(LinearLayer::new_rand(16 * 13 * 13, latent_dim * 2)),
    ]);

    let mut sampler = GaussianSampler::new(latent_dim);

    let mut decoder = Network::new(vec![
        Box::new(LinearLayer::new_rand(latent_dim, 16 * 7 * 7)),
        Box::new(ReshapeLayer::new(vec![16, 7, 7])),
        Box::new(ActivationLayer::new(Relu)),
        Box::new(Upsample2DLayer::new_2x()),
        Box::new(Pad2DLayer::new_1()),
        Box::new(Conv2DLayer::new_kaiming(16, 16, (3, 3))),
        Box::new(ActivationLayer::new(Relu)),
        Box::new(Upsample2DLayer::new_2x()),
        Box::new(Pad2DLayer::new_1()),
        Box::new(Conv2DLayer::new_kaiming(16, 8, (3, 3))),
        Box::new(ActivationLayer::new(Relu)),
        Box::new(Pad2DLayer::new_1()),
        Box::new(Conv2DLayer::new_kaiming(8, 1, (3, 3))),
        Box::new(FlattenLayer::new()),
    ]);

    let mut encoder_opt = Adam::new(lr, 0.9, 0.999, 1e-8);
    let mut decoder_opt = Adam::new(lr, 0.9, 0.999, 1e-8);

    let all_images = load_images("data/mnist/train-images.idx3-ubyte");

    let images = &all_images[0..60000];

    save_mnist_tensor_pgm(&images[0], &output_dir.join("original_0.pgm"), false)
        .expect("failed to save original sample image");

    println!("CNN VAE Training on MNIST");
    println!("========================================\n");

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut total_recon = 0.0;
        let mut total_kl = 0.0;
        let start = Instant::now();

        for (step, image) in images.iter().enumerate() {
            // current step's KL weight (ramps from 0.0 -> 1.0)
            let global_step = epoch * images.len() + step;
            let beta_max = 0.1; // or 0.05
            let warmup_steps = (4.0 * images.len() as f32) as usize; // warm up slower
            let beta = (global_step as f32 / warmup_steps as f32).min(1.0) * beta_max;

            // Encoder Forward Pass: [784, 1] -> [32, 1]
            let mu_log_var = encoder.forward(image);
            let mu = Tensor::from_vec(vec![latent_dim, 1], mu_log_var.data[0..latent_dim].to_vec());
            let log_var =
                Tensor::from_vec(vec![latent_dim, 1], mu_log_var.data[latent_dim..].to_vec());

            // Reparameterisation Trick: z = mu + sigma * eps -> [16, 1]
            let z = sampler.forward_pass(&mu_log_var);

            // Decoder Forward Pass (reconstructed logits): -> [784, 1]
            let x_recon = decoder.forward(&z);

            // Losses
            let recon_loss = binary_cross_entropy(&x_recon, image);
            let kl_loss = kl_divergence(&mu, &log_var);
            let loss = recon_loss + beta * kl_loss;
            total_loss += loss;
            total_recon += recon_loss;
            total_kl += kl_loss;

            // Decoder Gradient: d(BCE)/d(logits) = sigmoid(logits) - target
            let d_recon = x_recon.zip_map(image, |logit, t| (1.0 / (1.0 + (-logit).exp())) - t);
            let d_z = decoder.backward(&d_recon);

            // Sampler Gradient
            let d_mu_log_var_recon = sampler.backward_pass(&d_z);

            // KL Divergence Gradients (scaled by beta)
            let d_mu_kl = d_kl_divergence_mu(&mu, &log_var).scale(beta);
            let d_log_var_kl = d_kl_divergence_log_var(&mu, &log_var).scale(beta);
            let mut combined_kl = d_mu_kl.data.clone();
            combined_kl.extend(&d_log_var_kl.data);
            let d_mu_log_var_kl = Tensor::from_vec(vec![latent_dim * 2, 1], combined_kl);

            // Encoder Backward Pass
            let d_encoder_total = d_mu_log_var_recon.add(&d_mu_log_var_kl);
            encoder.backward(&d_encoder_total);

            // Optimiser Updates
            encoder.update(&mut encoder_opt);
            decoder.update(&mut decoder_opt);

            // Progress log every 1,000 images
            if step > 0 && step % 1000 == 0 {
                println!(
                    "Step {}/{}: loss = {:.4} (recon = {:.4}, kl = {:.4}, beta = {:.2})",
                    step,
                    images.len(),
                    total_loss / (step + 1) as f32,
                    total_recon / (step + 1) as f32,
                    total_kl / (step + 1) as f32,
                    beta
                );
            }
        }

        println!(
            "\n>>> Epoch {} finished in {:?}: Loss = {:.4} (Recon = {:.4}, KL = {:.4})",
            epoch,
            start.elapsed(),
            total_loss / images.len() as f32,
            total_recon / images.len() as f32,
            total_kl / images.len() as f32,
        );

        let mu_log_var = encoder.forward(&images[0]);
        let z = sampler.forward_pass(&mu_log_var);
        let recon = decoder.forward(&z);

        save_mnist_tensor_pgm(
            &recon,
            &output_dir.join(format!("recon_epoch_{:02}.pgm", epoch)),
            true,
        )
        .expect("failed to save reconstruction");

        let random_z = GaussianSampler::sample_standard_normal(latent_dim);
        let generated = decoder.forward(&random_z);

        save_mnist_tensor_pgm(
            &generated,
            &output_dir.join(format!("sample_epoch_{:02}.pgm", epoch)),
            true,
        )
        .expect("failed to save generated sample");

        println!("Saved reconstruction and sample for Epoch {}\n", epoch);
    }

    create_dir_all("checkpoints").expect("failed to create checkpoints dir");
    decoder
        .save("checkpoints/cnn_vae_decoder.rmlc")
        .expect("failed to save decoder");
    encoder
        .save("checkpoints/cnn_vae_encoder.rmlc")
        .expect("failed to save encoder");
    println!("Saved checkpoints to checkpoints/cnn_vae_*.rmlc");

    let mut grid_samples = Vec::new();
    for _ in 0..16 {
        let z = GaussianSampler::sample_standard_normal(latent_dim);
        grid_samples.push(decoder.forward(&z));
    }
    save_pgm_grid_4x4(&grid_samples, &output_dir.join("final_grid_4x4.pgm")).unwrap();
}

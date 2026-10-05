use mentats::data::svhn::load_images;
use mentats::loss::cross_entropy::{binary_cross_entropy, d_binary_cross_entropy};
use mentats::loss::kl_divergence::{d_kl_divergence_log_var, d_kl_divergence_mu, kl_divergence};
use mentats::nn::Layer;
use mentats::nn::{network::Network, sampling::GaussianSampler};
use mentats::optimiser::adam::Adam;
use mentats::utils::batch::{stack_tensor_refs, BatchIterator};
use mentats::utils::image::{save_ppm, save_ppm_grid_4x4};
use mentats::utils::vae::{combine_kl_grads, split_mu_log_var};
use mentats::Tensor;
use std::fs::create_dir_all;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let epochs: usize = if args.len() > 1 {
        args[1].parse().unwrap_or(100)
    } else {
        100
    };
    let lr = 0.001;
    let batch_size = 64;
    let total_warmup_epochs = 20.0;
    let beta_max: f32 = 0.1;

    let output_dir = Path::new("outputs/svhn_cnn_vae");
    create_dir_all(output_dir).expect("failed to create output directory");
    let latent_dim: usize = 100;

    let mut encoder = Network::builder()
        .conv2d_kaiming(3, 16, (3, 3))
        .relu()
        .max_pool2d()
        .flatten()
        .input(16 * 15 * 15) // <-- Sets builder feature dimension
        .linear_kaiming(latent_dim * 2) // <-- Uses 3,600 as in_features
        .build();

    let mut sampler = GaussianSampler::new(latent_dim);

    let mut decoder = Network::builder()
        .input(latent_dim)
        .linear(16 * 8 * 8)
        .reshape(vec![16, 8, 8])
        .relu()
        .upsample2d_2x()
        .pad2d_1()
        .conv2d_kaiming(16, 16, (3, 3))
        .relu()
        .upsample2d_2x()
        .pad2d_1()
        .conv2d_kaiming(16, 8, (3, 3))
        .relu()
        .pad2d_1()
        .conv2d_kaiming(8, 3, (3, 3))
        .build();

    let mut encoder_opt = Adam::new(lr, 0.9, 0.999, 1e-8);
    let mut decoder_opt = Adam::new(lr, 0.9, 0.999, 1e-8);

    let all_images = load_images("data/svhn/train_images.bin");

    let images = &all_images[0..60000];

    save_ppm(&images[0], &output_dir.join("original_0.ppm"))
        .expect("failed to save original sample image");

    println!("CNN VAE Training on SVHN");
    println!("========================================\n");

    let batches_per_epoch = images.len().div_ceil(batch_size);
    let total_warmup_steps = (total_warmup_epochs * batches_per_epoch as f32) as usize;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut total_recon = 0.0;
        let mut total_kl = 0.0;
        let mut batch_count = 0;

        let mut batch_iter = BatchIterator::new(images.len(), batch_size, true);

        let start = Instant::now();
        while let Some(batch_indices) = batch_iter.next_batch() {
            let global_step = epoch * batches_per_epoch + batch_count;
            let beta = (global_step as f32 / total_warmup_steps as f32).min(1.0) * beta_max;

            let batch_refs: Vec<&Tensor> = batch_indices.iter().map(|&idx| &images[idx]).collect();
            let x_batch = stack_tensor_refs(&batch_refs);

            let mu_log_var = encoder.forward(&x_batch);
            let (mu, log_var) = split_mu_log_var(&mu_log_var, latent_dim);

            let z = sampler.forward_pass(&mu_log_var);

            let x_recon = decoder.forward(&z);

            let recon_loss = binary_cross_entropy(&x_recon, &x_batch);
            let kl_loss = kl_divergence(&mu, &log_var);
            let loss = recon_loss + beta * kl_loss;
            total_loss += loss;
            total_recon += recon_loss;
            total_kl += kl_loss;

            let d_recon = d_binary_cross_entropy(&x_recon, &x_batch);
            let d_z = decoder.backward(&d_recon);

            let d_mu_log_var_recon = sampler.backward_pass(&d_z);

            let d_mu_kl = d_kl_divergence_mu(&mu, &log_var).scale(beta);
            let d_log_var_kl = d_kl_divergence_log_var(&mu, &log_var).scale(beta);
            let d_mu_log_var_kl = combine_kl_grads(&d_mu_kl, &d_log_var_kl, latent_dim);

            let d_encoder_total = d_mu_log_var_recon.add(&d_mu_log_var_kl);
            encoder.backward(&d_encoder_total);

            encoder.update(&mut encoder_opt);
            decoder.update(&mut decoder_opt);

            batch_count += 1;

            if batch_count == 1 || batch_count % 100 == 0 || batch_count == batches_per_epoch {
                println!(
                    "Epoch {:02}/{} | Batch {:04}/{} | Beta: {:.3} | Loss: {:.4} (Recon: {:.4}, KL: {:.4})",
                    epoch + 1,
                    epochs,
                    batch_count,
                    batches_per_epoch,
                    beta,
                    total_loss / batch_count as f32,
                    total_recon / batch_count as f32,
                    total_kl / batch_count as f32,
                );
            }
        }

        let avg_loss = total_loss / batch_count.max(1) as f32;
        let avg_recon = total_recon / batch_count.max(1) as f32;
        let avg_kl = total_kl / batch_count.max(1) as f32;

        println!(
            "\n>>> Epoch {} finished in {:?}: Loss = {:.4} (Recon = {:.4}, KL = {:.4})",
            epoch + 1,
            start.elapsed(),
            avg_loss,
            avg_recon,
            avg_kl,
        );

        let mu_log_var = encoder.forward(&images[0]);
        let z = sampler.forward_pass(&mu_log_var);
        let recon = decoder.forward(&z);

        save_ppm(
            &recon,
            &output_dir.join(format!("recon_epoch_{:02}.ppm", epoch)),
        )
        .expect("failed to save reconstruction");

        let random_z = GaussianSampler::sample_standard_normal(latent_dim);
        let generated = decoder.forward(&random_z);

        save_ppm(
            &generated,
            &output_dir.join(format!("sample_epoch_{:02}.ppm", epoch)),
        )
        .expect("failed to save generated sample");

        println!("Saved reconstruction and sample for Epoch {}\n", epoch);
    }

    create_dir_all("checkpoints").expect("failed to create checkpoints dir");
    decoder
        .save("checkpoints/svhn_cnn_vae_decoder.rmlc")
        .expect("failed to save decoder");
    encoder
        .save("checkpoints/svhn_cnn_vae_encoder.rmlc")
        .expect("failed to save encoder");
    println!("Saved checkpoints to checkpoints/svhn_cnn_vae_*.rmlc");

    let mut grid_samples = Vec::new();
    for _ in 0..16 {
        let z = GaussianSampler::sample_standard_normal(latent_dim);
        grid_samples.push(decoder.forward(&z));
    }
    save_ppm_grid_4x4(&grid_samples, &output_dir.join("final_grid_4x4.ppm")).unwrap();
    println!("Saved 4x4 sample grid to outputs/svhn_cnn_vae/final_grid_4x4.ppm");
}

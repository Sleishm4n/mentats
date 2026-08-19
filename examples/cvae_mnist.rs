use mentats::data::mnist::{load_images, load_labels, one_hot};
use mentats::loss::kl_divergence::{d_kl_divergence_log_var, d_kl_divergence_mu};
use mentats::loss::{cross_entropy::binary_cross_entropy, kl_divergence::kl_divergence};
use mentats::nn::activation::{ActivationKind::Relu, ActivationLayer};
use mentats::nn::linear::LinearLayer;
use mentats::nn::network::Network;
use mentats::nn::sampling::GaussianSampler;
use mentats::nn::Layer;
use mentats::optimiser::adam::Adam;
use mentats::tensor::Tensor;
use mentats::utils::batch::{stack_tensors, BatchIterator};
use mentats::utils::checkpoint::{load_network_or_panic, save_network};
use mentats::utils::image::save_mnist_tensor_pgm;
use mentats::utils::vae::{combine_kl_grads, split_mu_log_var};
use std::env;
use std::fs::create_dir_all;
use std::path::Path;

const CVAE_ENCODER_CHECKPOINT: &str = "checkpoints/cvae_encoder.rmlc";
const CVAE_DECODER_CHECKPOINT: &str = "checkpoints/cvae_decoder.rmlc";

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
    load_network_or_panic(
        "checkpoints/mnist_classifier.rmlc",
        "Run the mnist example first to train and save",
    )
}

fn concat_features_2d(left: &Tensor, right: &Tensor) -> Tensor {
    left.concat_features_2d(right)
}

fn concat_features_batch(left: &Tensor, right: &Tensor) -> Tensor {
    left.concat_features_batch(right)
}

fn take_first_features_batch(tensor: &Tensor, keep_features: usize) -> Tensor {
    tensor.take_first_features_batch(keep_features)
}

fn save_cvae(encoder: &Network, decoder: &Network) {
    save_network(encoder, CVAE_ENCODER_CHECKPOINT);
    save_network(decoder, CVAE_DECODER_CHECKPOINT);
}

fn generate_from_checkpoint(class_id: u8, count: usize, latent_dim: usize, output_dir: &Path) {
    let mut decoder = load_network_or_panic(
        CVAE_DECODER_CHECKPOINT,
        "Run the cvae_mnist example first to train and save.",
    );

    create_dir_all(output_dir).expect("failed to create output directory");

    for i in 0..count {
        let random_z = GaussianSampler::sample_standard_normal(latent_dim);
        let y_cond = one_hot(class_id);
        let dec_in = concat_features_2d(&random_z, &y_cond);
        let generated = decoder.forward(&dec_in);

        save_mnist_tensor_pgm(
            &generated,
            &output_dir.join(format!("generated_class_{}_{}.pgm", class_id, i)),
            true,
        )
        .expect("failed to save generated image");
    }

    println!(
        "Saved {} generated samples for class {} to {}",
        count,
        class_id,
        output_dir.display()
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let latent_dim = 32;
    let label_dim = 10;
    let batch_size = 64;
    let epochs = 30;

    let beta_max = 1.0;
    let total_warmup_epochs = 5.0;

    if args.len() > 1 && args[1] == "generate" {
        if args.len() < 3 {
            panic!("Usage: cargo run --example cvae_mnist -- generate <class 0-9> [count]");
        }

        let class_id: u8 = args[2]
            .parse::<u8>()
            .expect("class must be a number from 0 to 9");
        assert!(class_id <= 9, "class must be in range 0..=9");

        let count = if args.len() >= 4 {
            args[3]
                .parse::<usize>()
                .expect("count must be a positive integer")
        } else {
            8
        };
        assert!(count > 0, "count must be > 0");

        let output_dir = Path::new("outputs/cvae_mnist/generated");
        generate_from_checkpoint(class_id, count, latent_dim, output_dir);
        return;
    }

    let mut encoder = Network::new(vec![
        Box::new(LinearLayer::new_rand(784 + label_dim, 512)),
        Box::new(ActivationLayer::new(Relu)),
        Box::new(LinearLayer::new_rand(512, 256)),
        Box::new(ActivationLayer::new(Relu)),
        Box::new(LinearLayer::new_rand(256, latent_dim * 2)),
    ]);

    let mut sampler = GaussianSampler::new(latent_dim);

    let mut decoder = Network::new(vec![
        Box::new(LinearLayer::new_rand(latent_dim + label_dim, 256)),
        Box::new(ActivationLayer::new(Relu)),
        Box::new(LinearLayer::new_rand(256, 512)),
        Box::new(ActivationLayer::new(Relu)),
        Box::new(LinearLayer::new_rand(512, 784)),
    ]);

    let mut encoder_opt = Adam::new(0.001, 0.9, 0.999, 1e-8);
    let mut decoder_opt = Adam::new(0.001, 0.9, 0.999, 1e-8);

    let images = load_images("data/mnist/train-images.idx3-ubyte");
    let labels = load_labels("data/mnist/train-labels.idx1-ubyte");
    let mut classifier = load_classifier();

    let output_dir = Path::new("outputs/cvae_mnist");
    create_dir_all(output_dir).expect("failed to create outputs/cvae_mnist directory");

    save_mnist_tensor_pgm(&images[0], &output_dir.join("original_0.pgm"), false)
        .expect("failed to save original sample image");

    println!("CVAE Training on MNIST");
    println!("========================================\n");

    // .div_ceil is same as (images.len() + batch_size - 1) / batch_size
    let batches_per_epoch = images.len().div_ceil(batch_size);
    let total_warmup_steps = (total_warmup_epochs * batches_per_epoch as f32) as usize;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut total_recon_loss = 0.0;
        let mut total_kl_loss = 0.0;
        let mut batch_count = 0;

        let mut batch_iter = BatchIterator::new(images.len(), batch_size, true);

        while let Some(batch_indices) = batch_iter.next_batch() {
            let global_step = epoch * batches_per_epoch + batch_count;
            let beta = (global_step as f32 / total_warmup_steps as f32).min(1.0) * beta_max;

            let mut batch_images = Vec::with_capacity(batch_indices.len());
            let mut batch_labels = Vec::with_capacity(batch_indices.len());

            for &idx in &batch_indices {
                batch_images.push(images[idx].clone());
                batch_labels.push(one_hot(labels[idx]));
            }

            let x_batch = stack_tensors(&batch_images);
            let y_batch = stack_tensors(&batch_labels);

            let enc_input = concat_features_batch(&x_batch, &y_batch);

            let mu_log_var = encoder.forward(&enc_input);
            let (mu, log_var) = split_mu_log_var(&mu_log_var, latent_dim);

            let z = sampler.forward_pass(&mu_log_var);
            let dec_input = concat_features_batch(&z, &y_batch);
            let x_recon = decoder.forward(&dec_input);

            let recon_loss = binary_cross_entropy(&x_recon, &x_batch);
            let kl_loss = kl_divergence(&mu, &log_var);

            let n_features = 784.0;
            let current_batch_size = batch_indices.len() as f32;
            let d_recon = x_recon
                .zip_map(&x_batch, |logit, t| (1.0 / (1.0 + (-logit).exp())) - t)
                .scale(1.0 / (n_features * current_batch_size));

            let d_dec_input = decoder.backward(&d_recon);
            let d_z = take_first_features_batch(&d_dec_input, latent_dim);

            let d_mu_log_var_recon = sampler.backward_pass(&d_z);

            // Free-bits-aware KL gradients (see kl_divergence.rs).
            // Note the signature now requires both mu and log_var.
            let d_mu_kl = d_kl_divergence_mu(&mu, &log_var).scale(beta);
            let d_log_var_kl = d_kl_divergence_log_var(&mu, &log_var).scale(beta);
            let d_mu_log_var_kl = combine_kl_grads(&d_mu_kl, &d_log_var_kl, latent_dim);

            let d_mu_log_var_total = d_mu_log_var_recon.add(&d_mu_log_var_kl);
            let _ = encoder.backward(&d_mu_log_var_total);

            encoder.update(&mut encoder_opt);
            decoder.update(&mut decoder_opt);

            let batch_total = recon_loss + beta * kl_loss;
            total_loss += batch_total;
            total_recon_loss += recon_loss;
            total_kl_loss += kl_loss;
            batch_count += 1;
        }

        let epoch_end_step = (epoch + 1) * batches_per_epoch;
        let beta_logged = (epoch_end_step as f32 / total_warmup_steps as f32).min(1.0) * beta_max;

        println!(
            "Epoch {}: Loss = {:.4} (Recon: {:.4}, KL: {:.4}, Beta: {:.5})",
            epoch,
            total_loss / batch_count as f32,
            total_recon_loss / batch_count as f32,
            total_kl_loss / batch_count as f32,
            beta_logged,
        );

        let x_single = &images[0];
        let y_single = one_hot(labels[0]);
        let enc_in_single = concat_features_2d(x_single, &y_single);
        let mu_log_var_single = encoder.forward(&enc_in_single);
        let z_single = sampler.forward_pass(&mu_log_var_single);
        let dec_in_single = concat_features_2d(&z_single, &y_single);
        let x_recon_single = decoder.forward(&dec_in_single);

        if epoch % 5 == 0 {
            save_mnist_tensor_pgm(
                &x_recon_single,
                &output_dir.join(format!("recon_epoch_{:02}.pgm", epoch)),
                true,
            )
            .expect("failed to save reconstruction image");

            for class_id in 0..10u8 {
                let random_z = GaussianSampler::sample_standard_normal(latent_dim);
                let y_cond = one_hot(class_id);
                let dec_in = concat_features_2d(&random_z, &y_cond);
                let generated = decoder.forward(&dec_in);

                save_mnist_tensor_pgm(
                    &generated,
                    &output_dir.join(format!("sample_epoch_{:02}_class_{}.pgm", epoch, class_id)),
                    true,
                )
                .expect("failed to save generated image");

                let generated_prob = generated.map(|v| 1.0 / (1.0 + (-v).exp()));
                let generated_logits = classifier.forward(&generated_prob);
                let (pred_label, pred_confidence) = predict_from_logits(&generated_logits);

                println!(
                    "Class condition {} -> predicted {} ({:.1}% confidence)",
                    class_id,
                    pred_label,
                    pred_confidence * 100.0
                );
            }
        }
    }

    println!("\nCVAE training complete!");
    save_cvae(&encoder, &decoder);
    println!(
        "Saved checkpoints: {} and {}",
        CVAE_ENCODER_CHECKPOINT, CVAE_DECODER_CHECKPOINT
    );
    println!("Saved outputs in outputs/cvae_mnist/");
}

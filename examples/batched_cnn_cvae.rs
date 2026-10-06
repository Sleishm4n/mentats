use mentats::data::mnist::{load_images, load_labels, one_hot};
use mentats::loss::cross_entropy::{binary_cross_entropy, d_binary_cross_entropy};
use mentats::loss::kl_divergence::{d_kl_divergence_log_var, d_kl_divergence_mu, kl_divergence};
use mentats::nn::Layer;
use mentats::nn::{network::Network, sampling::GaussianSampler};
use mentats::optimiser::adam::Adam;
use mentats::utils::batch::{slice_batch, stack_tensors, BatchIterator};
use mentats::utils::checkpoint::{load_network_or_panic, save_network};
use mentats::utils::image::{save_mnist_tensor_pgm, save_pgm_grid_4x4};
use mentats::utils::vae::{combine_kl_grads, split_mu_log_var};
use mentats::Tensor;
use std::env;
use std::fs::create_dir_all;
use std::path::Path;
use std::time::Instant;

const CVAE_DECODER_CHECKPOINT: &str = "checkpoints/batched_cnn_cvae_decoder.rmlc";
const CVAE_DENSE_ENC_CHECKPOINT: &str = "checkpoints/batched_cnn_cvae_dense_encoder.rmlc";
const CVAE_CONV_ENC_CHECKPOINT: &str = "checkpoints/batched_cnn_cvae_conv_encoder.rmlc";

/// Stacks a slice of Tensor references along a new batch dimension without cloning each tensor first.
fn stack_tensor_refs(tensors: &[&Tensor]) -> Tensor {
    assert!(!tensors.is_empty(), "cannot stack empty tensor list");
    let first_shape = &tensors[0].shape;
    for t in tensors {
        assert_eq!(
            &t.shape, first_shape,
            "all tensors must have the same shape to stack"
        );
    }

    let batch_size = tensors.len();
    let mut stacked_shape = vec![batch_size];
    stacked_shape.extend(first_shape);

    let sample_elements: usize = first_shape.iter().product();
    let mut data = Vec::with_capacity(batch_size * sample_elements);
    for t in tensors {
        data.extend_from_slice(&t.data);
    }

    Tensor::from_vec(stacked_shape, data)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let latent_dim = 10;
    let label_dim = 10;
    let batch_size = 128;
    let epochs = 15;
    let lr = 0.002;

    let beta_max = 0.1;
    let total_warmup_epochs = 5.0;

    if args.len() > 1 && args[1] == "generate" {
        if args.len() < 3 {
            eprintln!(
                "Usage: cargo run --example batched_cnn_cvae --release -- generate <digit 0-9> [count]"
            );
            return;
        }

        let digit: u8 = args[2]
            .parse::<u8>()
            .expect("digit must be an integer between 0 and 9");
        assert!(digit <= 9, "digit must be in range 0..=9");

        let count: usize = if args.len() >= 4 {
            args[3].parse::<usize>().unwrap_or(16)
        } else {
            16
        };

        let mut decoder = load_network_or_panic(
            CVAE_DECODER_CHECKPOINT,
            "Run training first to generate the checkpoint!",
        );

        let output_dir = Path::new("outputs/batched_cnn_cvae/generated");
        create_dir_all(output_dir).expect("failed to create output directory");

        let y = one_hot(digit);
        let mut z_vec = Vec::with_capacity(count);
        let mut y_vec = Vec::with_capacity(count);
        for _ in 0..count {
            z_vec.push(GaussianSampler::sample_standard_normal(latent_dim));
            y_vec.push(y.clone());
        }

        let z_batch = stack_tensors(&z_vec);
        let y_batch = stack_tensors(&y_vec);
        let dec_in = z_batch.concat_features_batch(&y_batch);
        let img_batch = decoder.forward(&dec_in);

        let indices: Vec<usize> = (0..count).collect();
        let samples = slice_batch(&img_batch, &indices);

        for (i, img) in samples.iter().enumerate() {
            save_mnist_tensor_pgm(
                img,
                &output_dir.join(format!("digit_{}_{:02}.pgm", digit, i)),
                true,
            )
            .expect("failed to save generated sample");
        }

        if count == 16 {
            let grid_path = output_dir.join(format!("grid_digit_{}.pgm", digit));
            save_pgm_grid_4x4(&samples, &grid_path).expect("failed to save 4x4 grid");
            println!(
                "Saved 4x4 grid of synthesized {}s to {}",
                digit,
                grid_path.display()
            );
        }

        println!(
            "Successfully generated {} samples of digit {} to {}",
            count,
            digit,
            output_dir.display()
        );
        return;
    }

    let output_dir = Path::new("outputs/batched_cnn_cvae");
    create_dir_all(output_dir).expect("failed to create output directory");

    let mut conv_encoder = Network::builder()
        .reshape(vec![1, 28, 28])
        .conv2d_kaiming(1, 16, (3, 3))
        .relu()
        .max_pool2d()
        .conv2d_kaiming(16, 32, (3, 3))
        .relu()
        .max_pool2d()
        .flatten()
        .build();

    let mut dense_encoder = Network::builder()
        .input(32 * 5 * 5 + label_dim)
        .linear(latent_dim * 2)
        .build();

    let mut sampler = GaussianSampler::new(latent_dim);

    let mut decoder = Network::builder()
        .input(latent_dim + label_dim)
        .linear(16 * 7 * 7)
        .reshape(vec![16, 7, 7])
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
        .conv2d_kaiming(8, 1, (3, 3))
        .flatten()
        .build();

    let mut conv_opt = Adam::new(lr, 0.9, 0.999, 1e-8);
    let mut dense_opt = Adam::new(lr, 0.9, 0.999, 1e-8);
    let mut decoder_opt = Adam::new(lr, 0.9, 0.999, 1e-8);

    println!("Loading MNIST images and labels..");
    let images = load_images("data/mnist/train-images.idx3-ubyte");
    let labels = load_labels("data/mnist/train-labels.idx1-ubyte");
    assert_eq!(
        images.len(),
        labels.len(),
        "images and labels count mismatch"
    );

    println!("CNN Conditional VAE Training on MNIST (60,000 samples)");
    println!("==========================================================\n");

    let mut logger = mentats::utils::metrics::MetricsLogger::new(
        "batched_cnn_cvae",
        "Batched CNN CVAE",
        "MNIST",
        "generative",
        "ConvEncoder -> DenseLatentHead(10) -> ConvDecoder",
        "Adam (lr 0.001)",
    );

    let batches_per_epoch = images.len().div_ceil(batch_size);
    let total_warmup_steps = (total_warmup_epochs * batches_per_epoch as f32) as usize;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut total_recon = 0.0;
        let mut total_kl = 0.0;
        let mut batch_count = 0;

        let mut batch_iter = BatchIterator::new(images.len(), batch_size, true);
        let mut _step = 0;

        let start = Instant::now();

        while let Some(batch_indices) = batch_iter.next_batch() {
            // if step >= 100 { break; }
            let global_step = epoch * batches_per_epoch + batch_count;
            let beta = (global_step as f32 / total_warmup_steps as f32).min(1.0) * beta_max;

            let batch_refs: Vec<&Tensor> = batch_indices.iter().map(|&idx| &images[idx]).collect();
            let batch_labels: Vec<Tensor> = batch_indices
                .iter()
                .map(|&idx| one_hot(labels[idx]))
                .collect();

            let x_batch = stack_tensor_refs(&batch_refs);
            let y_batch = stack_tensors(&batch_labels);

            let features = conv_encoder.forward(&x_batch);

            let enc_in = features.concat_features_batch(&y_batch);

            let mu_log_var = dense_encoder.forward(&enc_in);
            let (mu, log_var) = split_mu_log_var(&mu_log_var, latent_dim);

            let z = sampler.forward_pass(&mu_log_var);
            let dec_in = z.concat_features_batch(&y_batch);
            let x_recon = decoder.forward(&dec_in);

            let recon_loss = binary_cross_entropy(&x_recon, &x_batch);
            let kl_loss = kl_divergence(&mu, &log_var);

            let d_recon = d_binary_cross_entropy(&x_recon, &x_batch);
            let d_dec_input = decoder.backward(&d_recon);
            let d_z = d_dec_input.take_first_features_batch(latent_dim);

            let d_mu_log_var_recon = sampler.backward_pass(&d_z);

            let d_mu_kl = d_kl_divergence_mu(&mu, &log_var).scale(beta);
            let d_log_var_kl = d_kl_divergence_log_var(&mu, &log_var).scale(beta);
            let d_mu_log_var_kl = combine_kl_grads(&d_mu_kl, &d_log_var_kl, latent_dim);

            let d_mu_log_var_total = d_mu_log_var_recon.add(&d_mu_log_var_kl);
            let d_enc_in = dense_encoder.backward(&d_mu_log_var_total);

            let d_features = d_enc_in.take_first_features_batch(32 * 5 * 5);
            conv_encoder.backward(&d_features);

            conv_encoder.update(&mut conv_opt);
            dense_encoder.update(&mut dense_opt);
            decoder.update(&mut decoder_opt);

            let batch_total = recon_loss + beta * kl_loss;
            total_loss += batch_total;
            total_recon += recon_loss;
            total_kl += kl_loss;
            batch_count += 1;

            if batch_count == 1 || batch_count % 20 == 0 {
                println!(
                    " Batch {:3}/{}: loss = {:.4} (recon = {:.4}, kl = {:.4}, beta = {:.4}) [{:.1?}]",
                    batch_count,
                    batches_per_epoch,
                    total_loss / batch_count as f32,
                    total_recon / batch_count as f32,
                    total_kl / batch_count as f32,
                    beta,
                    start.elapsed()
                );
            }

            if batch_count % 100 == 0 {
                println!(
                    " Batch {}/{}: loss = {:.4} (recon = {:.4}, kl = {:.4}, beta = {:.4})",
                    batch_count,
                    batches_per_epoch,
                    total_loss / batch_count as f32,
                    total_recon / batch_count as f32,
                    total_kl / batch_count as f32,
                    beta
                );
            }
            // step += 1;
        }

        let epoch_end_step = (epoch + 1) * batches_per_epoch;
        let beta_logged = (epoch_end_step as f32 / total_warmup_steps as f32).min(1.0) * beta_max;

        let avg_loss = total_loss / batch_count as f32;
        let avg_recon = total_recon / batch_count as f32;
        let avg_kl = total_kl / batch_count as f32;
        let elapsed = start.elapsed();

        logger.log_epoch_with_extra(
            epoch,
            avg_loss,
            None,
            elapsed.as_secs_f32(),
            vec![
                ("recon_loss", avg_recon),
                ("kl_loss", avg_kl),
                ("beta", beta_logged),
            ],
        );

        println!(
            "\n>>> Epoch {} finished in {:?}: Loss = {:.4} (Recon = {:.4}, KL = {:.4}, Beta = {:.4})\n",
            epoch,
            elapsed,
            avg_loss,
            avg_recon,
            avg_kl,
            beta_logged,
        );

        // Epoch reconstruction sample (batch-of-1)
        let x0_batch = stack_tensor_refs(&[&images[0]]);
        let y0_batch = stack_tensors(&[one_hot(labels[0])]);
        let feat0 = conv_encoder.forward(&x0_batch);
        let enc0 = feat0.concat_features_batch(&y0_batch);
        let mu_lv0 = dense_encoder.forward(&enc0);
        let z0 = sampler.forward_pass(&mu_lv0);
        let dec0 = z0.concat_features_batch(&y0_batch);
        let recon0_batch = decoder.forward(&dec0);
        let recon0 = &slice_batch(&recon0_batch, &[0])[0];
        save_mnist_tensor_pgm(
            recon0,
            &output_dir.join(format!("recon_epoch_{:02}.pgm", epoch)),
            true,
        )
        .unwrap();

        // Epoch conditional sample fixed at digit 7 (batch-of-1)
        let sample_z = GaussianSampler::sample_standard_normal(latent_dim);
        let sample_z_batch = stack_tensors(&[sample_z]);
        let sample_y_batch = stack_tensors(&[one_hot(7)]);
        let sample_dec_in = sample_z_batch.concat_features_batch(&sample_y_batch);
        let sample_7_batch = decoder.forward(&sample_dec_in);
        let sample_7 = &slice_batch(&sample_7_batch, &[0])[0];
        save_mnist_tensor_pgm(
            sample_7,
            &output_dir.join(format!("sample_digit7_epoch_{:02}.pgm", epoch)),
            true,
        )
        .unwrap();

        println!(
            "Saved reconstruction and digit-7 sample for Epoch {}",
            epoch
        );
    }

    save_network(&decoder, CVAE_DECODER_CHECKPOINT);
    save_network(&dense_encoder, CVAE_DENSE_ENC_CHECKPOINT);
    save_network(&conv_encoder, CVAE_CONV_ENC_CHECKPOINT);
    println!("Saved all checkpoints to checkpoints/");

    // Batched 4x4 showcase grid (16 samples in 1 forward pass)
    let showcase_digits: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 4, 7, 8, 9];
    let mut z_vec = Vec::with_capacity(16);
    let mut y_vec = Vec::with_capacity(16);
    for &digit in &showcase_digits {
        z_vec.push(GaussianSampler::sample_standard_normal(latent_dim));
        y_vec.push(one_hot(digit));
    }
    let z_batch = stack_tensors(&z_vec);
    let y_batch = stack_tensors(&y_vec);
    let dec_in = z_batch.concat_features_batch(&y_batch);
    let grid_batch = decoder.forward(&dec_in);
    let grid_indices: Vec<usize> = (0..16).collect();
    let grid_samples = slice_batch(&grid_batch, &grid_indices);

    let showcase_path = output_dir.join("final_showcase_grid_4x4.pgm");
    save_pgm_grid_4x4(&grid_samples, &showcase_path).unwrap();
    println!(
        "Saved full 4x4 multi-class showcase grid to {}",
        showcase_path.display()
    );

    if let Err(e) = logger.save_to_json(output_dir.join("metrics.json")) {
        eprintln!("Warning: failed to save metrics: {e}");
    }
}

use mentats::data::mnist::{load_images, load_labels, one_hot};
use mentats::loss::cross_entropy::binary_cross_entropy;
use mentats::loss::kl_divergence::{d_kl_divergence_log_var, d_kl_divergence_mu, kl_divergence};
use mentats::nn::Layer;
use mentats::nn::{network::Network, sampling::GaussianSampler};
use mentats::optimiser::adam::Adam;
use mentats::utils::checkpoint::{load_network_or_panic, save_network};
use mentats::utils::image::{save_mnist_tensor_pgm, save_pgm_grid_4x4};
use mentats::Tensor;
use std::env;
use std::fs::create_dir_all;
use std::path::Path;
use std::time::Instant;

const CVAE_DECODER_CHECKPOINT: &str = "checkpoints/cnn_cvae_decoder.rmlc";
const CVAE_DENSE_ENC_CHECKPOINT: &str = "checkpoints/cnn_cvae_dense_encoder.rmlc";
const CVAE_CONV_ENC_CHECKPOINT: &str = "checkpoints/cnn_cvae_conv_encoder.rmlc";

fn main() {
    let args: Vec<String> = env::args().collect();
    let latent_dim: usize = 10;
    let label_dim: usize = 10;
    let epochs: usize = 15;
    let lr = 0.001;

    // CLI Generation Mode: cargo run --example cnn_cvae --release -- generate <digit 0-9> [count]
    if args.len() > 1 && args[1] == "generate" {
        if args.len() < 3 {
            eprintln!(
                "Usage: cargo run --example cnn_cvae --release -- generate <digit 0-9> [count]"
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

        let output_dir = Path::new("outputs/cnn_cvae/generated");
        create_dir_all(output_dir).expect("failed to create output directory");

        let y = one_hot(digit);
        let mut samples = Vec::new();

        for i in 0..count {
            let z = GaussianSampler::sample_standard_normal(latent_dim);
            let dec_in = z.concat_features_2d(&y);
            let img = decoder.forward(&dec_in);

            save_mnist_tensor_pgm(
                &img,
                &output_dir.join(format!("digit_{}_{:02}.pgm", digit, i)),
                true,
            )
            .expect("failed to save generated sample");

            samples.push(img);
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

    // Training Mode
    let output_dir = Path::new("outputs/cnn_cvae");
    create_dir_all(output_dir).expect("failed to create output directory");

    // Old version pre [`nn::NetworkBuilder`]
    //
    // let mut conv_encoder = Network::new(vec![
    //     Box::new(ReshapeLayer::new(vec![1, 28, 28])),
    //     Box::new(Conv2DLayer::new_kaiming(1, 16, (3, 3))),
    //     Box::new(ActivationLayer::new(Relu)),
    //     Box::new(MaxPool2DLayer::new_stand()),
    //     Box::new(FlattenLayer::new()),
    // ]);

    let mut conv_encoder = Network::builder()
        .reshape(vec![1, 28, 28])
        .conv2d_kaiming(1, 16, (3, 3))
        .relu()
        .max_pool2d()
        .flatten()
        .build();

    // Old version pre [`nn::NetworkBuilder`]
    //
    // let mut dense_encoder = Network::new(vec![Box::new(LinearLayer::new_rand(
    //     16 * 13 * 13 + label_dim,
    //     latent_dim * 2,
    // ))]);

    let mut dense_encoder = Network::builder()
        .input(16 * 13 * 13 + label_dim)
        .linear(latent_dim * 2)
        .build();

    let mut sampler = GaussianSampler::new(latent_dim);

    // Old version pre [`nn::NetworkBuilder`]
    //
    // let mut decoder = Network::new(vec![
    //     Box::new(LinearLayer::new_rand(latent_dim + label_dim, 16 * 7 * 7)),
    //     Box::new(ReshapeLayer::new(vec![16, 7, 7])),
    //     Box::new(ActivationLayer::new(Relu)),
    //     Box::new(Upsample2DLayer::new_2x()),
    //     Box::new(Pad2DLayer::new_1()),
    //     Box::new(Conv2DLayer::new_kaiming(16, 16, (3, 3))),
    //     Box::new(ActivationLayer::new(Relu)),
    //     Box::new(Upsample2DLayer::new_2x()),
    //     Box::new(Pad2DLayer::new_1()),
    //     Box::new(Conv2DLayer::new_kaiming(16, 8, (3, 3))),
    //     Box::new(ActivationLayer::new(Relu)),
    //     Box::new(Pad2DLayer::new_1()),
    //     Box::new(Conv2DLayer::new_kaiming(8, 1, (3, 3))),
    //     Box::new(FlattenLayer::new()),
    // ]);

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

    println!("Loading MNIST images and labels...");
    let images = load_images("data/mnist/train-images.idx3-ubyte");
    let labels = load_labels("data/mnist/train-labels.idx1-ubyte");
    assert_eq!(
        images.len(),
        labels.len(),
        "images and labels count mismatch"
    );

    println!("CNN Conditional VAE Training on MNIST (60,000 samples)");
    println!("==========================================================\n");

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut total_recon = 0.0;
        let mut total_kl = 0.0;
        let start = Instant::now();

        for (step, image) in images.iter().enumerate() {
            let global_step = epoch * images.len() + step;
            let beta_max = 0.1;
            let warmup_steps = (4.0 * images.len() as f32) as usize;
            let beta = (global_step as f32 / warmup_steps as f32).min(1.0) * beta_max;

            let y = one_hot(labels[step]);

            let features = conv_encoder.forward(image);
            let enc_in = features.concat_features_2d(&y);
            let mu_log_var = dense_encoder.forward(&enc_in);

            let mu = Tensor::from_vec(vec![latent_dim, 1], mu_log_var.data[0..latent_dim].to_vec());
            let log_var =
                Tensor::from_vec(vec![latent_dim, 1], mu_log_var.data[latent_dim..].to_vec());

            let z = sampler.forward_pass(&mu_log_var);
            let dec_in = z.concat_features_2d(&y);
            let x_recon = decoder.forward(&dec_in);

            let recon_loss = binary_cross_entropy(&x_recon, image);
            let kl_loss = kl_divergence(&mu, &log_var);
            let loss = recon_loss + beta * kl_loss;

            total_loss += loss;
            total_recon += recon_loss;
            total_kl += kl_loss;

            let d_recon = x_recon.zip_map(image, |logit, t| (1.0 / (1.0 + (-logit).exp())) - t);
            let d_dec_in = decoder.backward(&d_recon);

            let d_z = Tensor::from_vec(vec![latent_dim, 1], d_dec_in.data[0..latent_dim].to_vec());
            let d_mu_log_var_recon = sampler.backward_pass(&d_z);

            let d_mu_kl = d_kl_divergence_mu(&mu, &log_var).scale(beta);
            let d_log_var_kl = d_kl_divergence_log_var(&mu, &log_var).scale(beta);
            let mut combined_kl = d_mu_kl.data.clone();
            combined_kl.extend(&d_log_var_kl.data);
            let d_mu_log_var_kl = Tensor::from_vec(vec![latent_dim * 2, 1], combined_kl);

            let d_encoder_total = d_mu_log_var_recon.add(&d_mu_log_var_kl);

            let d_enc_in = dense_encoder.backward(&d_encoder_total);
            let d_features = Tensor::from_vec(
                vec![16 * 13 * 13, 1],
                d_enc_in.data[0..16 * 13 * 13].to_vec(),
            );
            conv_encoder.backward(&d_features);

            conv_encoder.update(&mut conv_opt);
            dense_encoder.update(&mut dense_opt);
            decoder.update(&mut decoder_opt);

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

        // Epoch reconstruction sample
        let y0 = one_hot(labels[0]);
        let feat0 = conv_encoder.forward(&images[0]);
        let enc0 = feat0.concat_features_2d(&y0);
        let mu_lv0 = dense_encoder.forward(&enc0);
        let z0 = sampler.forward_pass(&mu_lv0);
        let dec0 = z0.concat_features_2d(&y0);
        let recon0 = decoder.forward(&dec0);
        save_mnist_tensor_pgm(
            &recon0,
            &output_dir.join(format!("recon_epoch_{:02}.pgm", epoch)),
            true,
        )
        .unwrap();

        // Epoch conditional sample fixed at digit 7 for consitency
        let sample_z = GaussianSampler::sample_standard_normal(latent_dim);
        let sample_y = one_hot(7);
        let sample_dec_in = sample_z.concat_features_2d(&sample_y);
        let sample_7 = decoder.forward(&sample_dec_in);
        save_mnist_tensor_pgm(
            &sample_7,
            &output_dir.join(format!("sample_digit7_epoch_{:02}.pgm", epoch)),
            true,
        )
        .unwrap();

        println!(
            "Saved reconstruction and digit-7 sample for Epoch {}\n",
            epoch
        );
    }

    save_network(&decoder, CVAE_DECODER_CHECKPOINT);
    save_network(&dense_encoder, CVAE_DENSE_ENC_CHECKPOINT);
    save_network(&conv_encoder, CVAE_CONV_ENC_CHECKPOINT);
    println!("Saved checkpoints to {}", CVAE_DECODER_CHECKPOINT);

    // Generate a 4x4 showcase grid containing digits 0 through 9 + 6 extra
    let showcase_digits: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 4, 7, 8, 9];
    let mut grid_samples = Vec::new();
    for &digit in &showcase_digits {
        let z = GaussianSampler::sample_standard_normal(latent_dim);
        let y = one_hot(digit);
        let dec_in = z.concat_features_2d(&y);
        grid_samples.push(decoder.forward(&dec_in));
    }
    save_pgm_grid_4x4(
        &grid_samples,
        &output_dir.join("final_showcase_grid_4x4.pgm"),
    )
    .unwrap();
    println!(
        "Saved full 4x4 multi-class showcase grid to outputs/cnn_cvae/final_showcase_grid_4x4.pgm"
    );
}

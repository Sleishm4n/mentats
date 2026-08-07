use mentats::data::mnist::{load_images, load_labels, one_hot};
use mentats::loss::cross_entropy::{cross_entropy, d_cross_entropy};
use mentats::nn::activation::{ActivationKind::Relu, ActivationLayer};
use mentats::nn::linear::LinearLayer;
use mentats::nn::network::Network;
use mentats::optimiser::adam::Adam;
use mentats::utils::batch::{slice_batch, stack_tensors, BatchIterator};
use std::fs::create_dir_all;
use std::time::Instant;

fn main() {
    let mut network = Network::new(vec![
        Box::new(LinearLayer::new_rand(784, 128)),
        Box::new(ActivationLayer::new(Relu)),
        Box::new(LinearLayer::new_rand(128, 10)),
    ]);
    let mut optimiser = Adam::new(0.001, 0.9, 0.999, 1e-8);
    let epochs = 5;

    let images = load_images("data/mnist/train-images.idx3-ubyte");
    let labels = load_labels("data/mnist/train-labels.idx1-ubyte");
    let test_images = load_images("data/mnist/t10k-images.idx3-ubyte");
    let test_labels = load_labels("data/mnist/t10k-labels.idx1-ubyte");

    let y_mat = one_hot(labels[0]);
    let y_hat = network.forward(&images[0]);
    let loss = cross_entropy(&y_hat, &y_mat);

    println!("Initial loss: {}", loss);

    for epoch in 0..epochs {
        let mut total_loss: f32 = 0.0;
        let mut correct = 0;

        let mut batch_iterator = BatchIterator::new(images.len(), 32, true);

        let start = Instant::now();
        let mut forward_time = std::time::Duration::ZERO;
        let mut backward_time = std::time::Duration::ZERO;
        while let Some(batch_indicies) = batch_iterator.next_batch() {
            let mut batch_images = Vec::new();
            let mut batch_labels = Vec::new();
            for &idx in &batch_indicies {
                batch_images.push(images[idx].clone());
                batch_labels.push(labels[idx])
            }

            let batched_input = stack_tensors(&batch_images);

            let t0 = Instant::now();
            let batched_output = network.forward(&batched_input);
            forward_time += t0.elapsed();

            let mut batch_loss = 0.0;
            let batched_output_sliced = slice_batch(
                &batched_output,
                &(0..batch_indicies.len()).collect::<Vec<_>>(),
            );

            for (out, &label) in batched_output_sliced.iter().zip(batch_labels.iter()) {
                let y_mat = one_hot(label);
                batch_loss += cross_entropy(out, &y_mat);

                let predicted = out
                    .data
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                if predicted == label as usize {
                    correct += 1;
                }
            }

            let _avg_loss = batch_loss / batch_indicies.len() as f32;
            total_loss += batch_loss;

            let mut d_outs = Vec::new();
            for (out, &label) in batched_output_sliced.iter().zip(batch_labels.iter()) {
                let y_mat = one_hot(label);
                let d_out = d_cross_entropy(out, &y_mat);
                d_outs.push(d_out.scale(1.0 / batch_indicies.len() as f32));
            }

            let t1 = Instant::now();
            let d_out_batched = stack_tensors(&d_outs);
            network.backward(&d_out_batched);
            network.update(&mut optimiser);
            backward_time += t1.elapsed();
        }

        println!(
            "forward: {:?}, backward+update: {:?}",
            forward_time, backward_time
        );

        println!(
            "Epoch {epoch}: loss = {:.4}, accuracy = {:.2}, elapsed time = {:?}",
            total_loss / images.len() as f32,
            correct as f32 / images.len() as f32 * 100.0,
            start.elapsed()
        );
    }
    let mut correct = 0;

    for (x, y) in test_images.iter().zip(test_labels.iter()) {
        let y_hat = network.forward(x);

        let predicted = y_hat
            .data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        if predicted == *y as usize {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / test_images.len() as f32 * 100.0;

    println!("Test accuracy: {:.2}%", accuracy);

    let checkpoint_path = "checkpoints/mnist_classifier.rmlc";
    create_dir_all("checkpoints").expect("failed to create checkpoints directory");
    network
        .save(checkpoint_path)
        .expect("failed to save classifer checkpoint");
    println!("Saved MNIST classifer checkpoint to {}", checkpoint_path);
}

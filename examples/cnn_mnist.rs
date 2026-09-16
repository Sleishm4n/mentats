use mentats::data::mnist::{load_images, load_labels, one_hot};
use mentats::loss::cross_entropy::{cross_entropy, d_cross_entropy};
use mentats::nn::{
    activation::{ActivationKind::Relu, ActivationLayer},
    conv::Conv2DLayer,
    flatten::FlattenLayer,
    linear::LinearLayer,
    network::Network,
    pooling::MaxPool2DLayer,
    reshape::ReshapeLayer,
};
use mentats::optimiser::adam::Adam;
use std::fs::create_dir_all;
use std::time::Instant;

fn main() {
    let mut cnn = Network::new(vec![
        Box::new(ReshapeLayer::new(vec![1, 28, 28])),
        Box::new(Conv2DLayer::new_kaiming(1, 8, (3, 3))),
        Box::new(ActivationLayer::new(Relu)),
        Box::new(MaxPool2DLayer::new_stand()),
        Box::new(FlattenLayer::new()),
        Box::new(LinearLayer::new_rand(8 * 13 * 13, 10)),
    ]);
    let mut optimiser = Adam::new(0.001, 0.9, 0.999, 1e-8);
    let epochs = 5;

    let images = load_images("data/mnist/train-images.idx3-ubyte");
    let labels = load_labels("data/mnist/train-labels.idx1-ubyte");
    let test_images = load_images("data/mnist/t10k-images.idx3-ubyte");
    let test_labels = load_labels("data/mnist/t10k-labels.idx1-ubyte");

    let y_mat = one_hot(labels[0]);
    let y_hat = cnn.forward(&images[0]);
    let loss = cross_entropy(&y_hat, &y_mat);

    println!("Initial loss: {}", loss);

    for epoch in 0..epochs {
        let mut total_loss: f32 = 0.0;
        let mut correct = 0;
        let start = Instant::now();

        let indices: Vec<usize> = (0..images.len()).collect();

        for (step, &idx) in indices.iter().enumerate() {
            let image = &images[idx];
            let label = labels[idx];
            let y_mat = one_hot(label);

            let y_hat = cnn.forward(image);

            let loss = cross_entropy(&y_hat, &y_mat);
            total_loss += loss;

            let predicted = y_hat
                .data
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            if predicted == label as usize {
                correct += 1;
            }

            let d_out = d_cross_entropy(&y_hat, &y_mat);
            cnn.backward(&d_out);
            cnn.update(&mut optimiser);

            if step > 0 && step % 1000 == 0 {
                println!(
                    "Step {step}/{}: running loss = {:.4}, accuracy = {:.2}%",
                    images.len(),
                    total_loss / (step + 1) as f32,
                    correct as f32 / (step + 1) as f32 * 100.0
                );
            }
        }

        println!(
            "Epoch {epoch}: loss = {:.4}, accuracy = {:.2}%, elapsed = {:?}",
            total_loss / images.len() as f32,
            correct as f32 / images.len() as f32 * 100.0,
            start.elapsed()
        );
    }
    let mut correct = 0;

    for (x, y) in test_images.iter().zip(test_labels.iter()) {
        let y_hat = cnn.forward(x);

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

    let checkpoint_path = "checkpoints/cnn_mnist_classifier.rmlc";
    create_dir_all("checkpoints").expect("failed to create checkpoints directory");
    cnn.save(checkpoint_path)
        .expect("failed to save classifer checkpoint");
    println!("Saved MNIST classifer checkpoint to {}", checkpoint_path);
}

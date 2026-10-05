use std::time::Instant;

use mentats::{
    data::svhn::{load_images, load_labels},
    loss::cross_entropy::{cross_entropy, d_cross_entropy},
    nn::Network,
    optimiser::adam::Adam,
    utils::batch::{stack_targets, stack_tensor_refs, BatchIterator},
    Tensor,
};

fn main() {
    let train_img_path = "data/svhn/train_images.bin";
    let train_lbl_path = "data/svhn/train_labels.bin";
    let test_img_path = "data/svhn/test_images.bin";
    let test_lbl_path = "data/svhn/test_labels.bin";

    println!("Loading SVHN training dataset...");
    let train_images = load_images(train_img_path);
    let train_labels = load_labels(train_lbl_path);

    assert_eq!(
        train_images.len(),
        train_labels.len(),
        "Train images and labels count mismatch"
    );

    println!("Loaded {} training samples.", train_images.len());

    println!("Loading SVHN test dataset...");
    let test_images = load_images(test_img_path);
    let test_labels = load_labels(test_lbl_path);

    assert_eq!(
        test_images.len(),
        test_labels.len(),
        "Test images and labels count mismatch"
    );

    println!("Loaded {} test samples.\n", test_images.len());

    let mut model = Network::builder()
        .conv2d_kaiming(3, 16, (3, 3))
        .relu()
        .max_pool2d()
        .conv2d_kaiming(16, 32, (3, 3))
        .relu()
        .max_pool2d()
        .flatten()
        .input(32 * 6 * 6)
        .linear(128)
        .relu()
        .linear(10)
        .build();

    let lr = 0.001;
    let mut optimiser = Adam::new(lr, 0.9, 0.999, 1e-8);
    let batch_size = 64;
    let epochs = 5;

    println!("SVHN Convolutional Classifier Training");
    println!(
        "Architecture: Conv(3->16) -> Pool -> Conv(16->32) -> Pool -> FC(1152->128) -> FC(128->10)"
    );
    println!(
        "Batch size: {}, Learning rate: {}, Epochs: {}\n",
        batch_size, lr, epochs
    );
    for epoch in 0..epochs {
        let mut total_loss = 0.0f32;
        let mut train_correct = 0;
        let mut processed_samples = 0;
        let mut batch_iter = BatchIterator::new(train_images.len(), batch_size, true);
        let start = Instant::now();

        let mut step = 0;

        while let Some(batch_indices) = batch_iter.next_batch() {
            let cur_batch_size = batch_indices.len();
            let batch_refs: Vec<&Tensor> = batch_indices
                .iter()
                .map(|&idx| &train_images[idx])
                .collect();
            let batch_lbls: Vec<u8> = batch_indices.iter().map(|&idx| train_labels[idx]).collect();

            let x_batch = stack_tensor_refs(&batch_refs);
            let y_batch = stack_targets(&batch_lbls);

            let y_hat = model.forward(&x_batch);
            let loss = cross_entropy(&y_hat, &y_batch);
            total_loss += loss * cur_batch_size as f32;

            // Accuracy calculation
            for (b, &lbl) in batch_lbls.iter().enumerate() {
                let start_idx = b * 10;
                let logits = &y_hat.data[start_idx..start_idx + 10];
                let pred = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(cls, _)| cls)
                    .unwrap();
                if pred == lbl as usize {
                    train_correct += 1;
                }
            }
            processed_samples += cur_batch_size;

            let d_loss = d_cross_entropy(&y_hat, &y_batch);
            model.backward(&d_loss);
            model.update(&mut optimiser);

            step += 1;
            if step % 200 == 0 {
                let current_acc = train_correct as f32 / processed_samples as f32 * 100.0;
                let current_loss = total_loss / processed_samples as f32;
                println!(
                    "Epoch {} | Step {}/{} | Loss: {:.4} | Train Acc: {:.2}%",
                    epoch + 1,
                    step,
                    train_images.len().div_ceil(batch_size),
                    current_loss,
                    current_acc
                );
            }
        }

        let epoch_loss = total_loss / processed_samples as f32;
        let epoch_acc = train_correct as f32 / processed_samples as f32 * 100.0;
        println!(
            "\n>>> Epoch {} Complete | Loss: {:.4} | Train Acc: {:.2}% | Elapsed: {:?}",
            epoch + 1,
            epoch_loss,
            epoch_acc,
            start.elapsed()
        );

        // Evaluate on test set (in batches of 128)
        let mut test_correct = 0;
        let test_batch_size = 128;
        let mut test_iter = BatchIterator::new(test_images.len(), test_batch_size, false);

        while let Some(batch_indices) = test_iter.next_batch() {
            let batch_refs: Vec<&Tensor> =
                batch_indices.iter().map(|&idx| &test_images[idx]).collect();
            let batch_lbls: Vec<u8> = batch_indices.iter().map(|&idx| test_labels[idx]).collect();

            let x_batch = stack_tensor_refs(&batch_refs);
            let y_hat = model.forward(&x_batch);

            for (b, &lbl) in batch_lbls.iter().enumerate() {
                let start_idx = b * 10;
                let logits = &y_hat.data[start_idx..start_idx + 10];
                let pred = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(cls, _)| cls)
                    .unwrap();
                if pred == lbl as usize {
                    test_correct += 1;
                }
            }
        }

        let test_acc = test_correct as f32 / test_images.len() as f32 * 100.0;
        println!(">>> Test Accuracy: {:.2}%\n", test_acc);
    }
}

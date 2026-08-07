use mentats::loss::mse::{d_mse, mse};
use mentats::nn::activation::{ActivationKind::Sigmoid, ActivationLayer};
use mentats::nn::linear::LinearLayer;
use mentats::nn::network::Network;
use mentats::optimiser::adam::Adam;
use mentats::tensor::Tensor;

fn main() {
    let mut network = Network::new(vec![
        Box::new(LinearLayer::new_rand(2, 2)),
        Box::new(ActivationLayer::new(Sigmoid)),
        Box::new(LinearLayer::new_rand(2, 1)),
    ]);
    let mut optimiser = Adam::new(0.001, 0.9, 0.999, 1e-8);
    let epochs = 10000;

    // XOR inputs and targets
    let inputs = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let targets = [[0.0], [1.0], [1.0], [0.0]];

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (x, y) in inputs.iter().zip(targets.iter()) {
            let x_mat = Tensor::from_vec(vec![2, 1], x.to_vec());
            let y_mat = Tensor::from_vec(vec![1], y.to_vec());
            let y_hat = network.forward(&x_mat);
            let loss = mse(&y_hat, &y_mat);
            let d_out = d_mse(&y_hat, &y_mat);
            network.backward(&d_out);
            network.update(&mut optimiser);
            total_loss += loss;
        }

        if epoch % 100 == 0 {
            println!("Epoch {epoch}: loss = {:.4}", total_loss / 4.0);
        }
    }

    println!("\nPredictions:");
    for (x, y) in inputs.iter().zip(targets.iter()) {
        let x_mat = Tensor::from_vec(vec![2, 1], x.to_vec());
        let y_hat = network.forward(&x_mat);
        println!("{:?} → {:.4} (target: {})", x, y_hat.data[0], y[0]);
    }
}

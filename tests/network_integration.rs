//! Integration tests for the Network API.
//!
//! These tests exercise the crate from the outside (no access to private
//! internals) and cover full training loops, forward-pass shape correctness,
//! and gradient descent convergence on a toy dataset.

use mentats::{
    loss::mse::{d_mse, mse},
    nn::{
        activation::{ActivationKind, ActivationLayer},
        linear::LinearLayer,
        network::Network,
    },
    optimiser::adam::Adam,
    tensor::Tensor,
};

// Helpers

fn scalar_tensor(v: f32) -> Tensor {
    let mut t = Tensor::new(vec![1, 1]);
    t.data[0] = v;
    t
}

fn make_xor_data() -> Vec<(Tensor, Tensor)> {
    // XOR truth table – inputs are [a, b] column vectors, target is scalar.
    let cases = [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
    ];
    cases
        .iter()
        .map(|&(a, b, y)| {
            let mut x = Tensor::new(vec![2, 1]);
            x.data[0] = a;
            x.data[1] = b;
            (x, scalar_tensor(y))
        })
        .collect()
}

fn make_network() -> Network {
    Network::new(vec![
        Box::new(LinearLayer::new_rand(2, 4)),
        Box::new(ActivationLayer::new(ActivationKind::Sigmoid)),
        Box::new(LinearLayer::new_rand(4, 1)),
        Box::new(ActivationLayer::new(ActivationKind::Sigmoid)),
    ])
}

// Tests

/// The output of a forward pass through the XOR network should always be a
/// [1, 1] tensor regardless of the input.
#[test]
fn forward_pass_output_shape() {
    let mut net = make_network();
    let data = make_xor_data();
    for (x, _) in &data {
        let out = net.forward(x);
        assert_eq!(out.shape, vec![1, 1], "output shape should be [1, 1]");
    }
}

/// Output values should be in (0, 1) because the final activation is sigmoid.
#[test]
fn forward_pass_output_in_range() {
    let mut net = make_network();
    let data = make_xor_data();
    for (x, _) in &data {
        let out = net.forward(x);
        let v = out.data[0];
        assert!(v > 0.0 && v < 1.0, "sigmoid output {v} should be in (0, 1)");
    }
}

/// After enough training steps the mean squared error on the XOR dataset
/// should decrease substantially from its initial value.
#[test]
fn xor_loss_decreases_after_training() {
    let mut net = make_network();
    let mut opt = Adam::new(0.05, 0.9, 0.999, 1e-8);
    let data = make_xor_data();

    // Record initial loss.
    let initial_loss: f32 = data
        .iter()
        .map(|(x, y)| {
            let out = net.forward(x);
            mse(&out, y)
        })
        .sum::<f32>()
        / data.len() as f32;

    // Train for 1 000 epochs.
    for _ in 0..1_000 {
        for (x, y) in &data {
            let out = net.forward(x);
            let grad = d_mse(&out, y);
            net.backward(&grad);
            net.update(&mut opt);
        }
    }

    // Measure final loss.
    let final_loss: f32 = data
        .iter()
        .map(|(x, y)| {
            let out = net.forward(x);
            mse(&out, y)
        })
        .sum::<f32>()
        / data.len() as f32;

    assert!(
        final_loss < initial_loss * 0.5,
        "expected loss to halve after training; initial={initial_loss:.4}, final={final_loss:.4}"
    );
}

/// A single gradient descent step must reduce the loss for a simple 1-sample
/// regression problem (sanity check that backward + update are wired up).
#[test]
fn single_step_reduces_loss() {
    let mut net = Network::new(vec![
        Box::new(LinearLayer::new_rand(2, 2)),
        Box::new(ActivationLayer::new(ActivationKind::Sigmoid)),
        Box::new(LinearLayer::new_rand(2, 1)),
    ]);
    let mut opt = Adam::new(0.1, 0.9, 0.999, 1e-8);

    let mut x = Tensor::new(vec![2, 1]);
    x.data[0] = 1.0;
    x.data[1] = 0.0;
    let target = scalar_tensor(1.0);

    let loss_before = mse(&net.forward(&x), &target);
    let grad = d_mse(&net.forward(&x), &target);
    net.backward(&grad);
    net.update(&mut opt);
    let loss_after = mse(&net.forward(&x), &target);

    assert!(
        loss_after < loss_before,
        "loss should decrease after one Adam step; before={loss_before:.4}, after={loss_after:.4}"
    );
}

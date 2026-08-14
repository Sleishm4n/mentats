//! Integration tests for individual layers: Softmax, activation functions,
//! gradient checker, and Adam optimizer.

use mentats::{
    loss::mse::{d_mse, mse},
    nn::{
        activation::{ActivationKind, ActivationLayer},
        linear::LinearLayer,
        network::Network,
        softmax::SoftmaxLayer,
        Layer,
    },
    optimiser::adam::Adam,
    tensor::Tensor,
    utils::grad_check::gradient_check,
};

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

// Softmax layer

/// Softmax outputs must always sum to 1.
#[test]
fn softmax_output_sums_to_one() {
    let mut layer = SoftmaxLayer::new();
    let input = Tensor::from_vec(vec![4, 1], vec![1.0, 2.0, 3.0, 4.0]);
    let out = layer.forward_pass(&input);
    let sum: f32 = out.data.iter().sum();
    assert!(approx_eq(sum, 1.0, 1e-5), "softmax sum={sum}");
}

/// All softmax outputs must be non-negative and finite.
#[test]
fn softmax_output_all_positive() {
    let mut layer = SoftmaxLayer::new();
    let input = Tensor::from_vec(vec![3, 1], vec![-100.0, 0.0, 100.0]);
    let out = layer.forward_pass(&input);
    for &v in &out.data {
        assert!(
            v >= 0.0 && v.is_finite(),
            "expected non-negative finite probability, got {v}"
        );
    }
}

/// Softmax should be numerically stable for very large inputs.
#[test]
fn softmax_numerically_stable_large_values() {
    let mut layer = SoftmaxLayer::new();
    let input = Tensor::from_vec(vec![3, 1], vec![1000.0, 1001.0, 1002.0]);
    let out = layer.forward_pass(&input);
    for v in &out.data {
        assert!(v.is_finite(), "softmax produced non-finite value");
    }
    let sum: f32 = out.data.iter().sum();
    assert!(approx_eq(sum, 1.0, 1e-4));
}

/// The backward gradient from softmax must have the same shape as the input.
#[test]
fn softmax_backward_shape_matches_input() {
    let mut layer = SoftmaxLayer::new();
    let input = Tensor::from_vec(vec![5, 1], vec![0.1, 0.2, 0.3, 0.4, 0.5]);
    layer.forward_pass(&input);
    let d_out = Tensor::from_vec(vec![5, 1], vec![1.0, 0.0, 0.0, 0.0, -1.0]);
    let grad = layer.backward_pass(&d_out);
    assert_eq!(grad.shape, input.shape);
}

// Activation layers

#[test]
fn relu_forward_zeroes_negatives() {
    let mut layer = ActivationLayer::new(ActivationKind::Relu);
    let input = Tensor::from_vec(vec![4, 1], vec![-2.0, -0.5, 0.0, 3.0]);
    let out = layer.forward_pass(&input);
    assert!(approx_eq(out.data[0], 0.0, 1e-6));
    assert!(approx_eq(out.data[1], 0.0, 1e-6));
    assert!(approx_eq(out.data[2], 0.0, 1e-6));
    assert!(approx_eq(out.data[3], 3.0, 1e-6));
}

#[test]
fn sigmoid_output_in_range() {
    let mut layer = ActivationLayer::new(ActivationKind::Sigmoid);
    let input = Tensor::from_vec(vec![4, 1], vec![-100.0, -1.0, 1.0, 100.0]);
    let out = layer.forward_pass(&input);
    for &v in &out.data {
        assert!(
            v >= 0.0 && v <= 1.0 && v.is_finite(),
            "sigmoid out of range: {v}"
        );
    }
}

/// Sigmoid(0) == 0.5 exactly.
#[test]
fn sigmoid_at_zero_is_half() {
    let mut layer = ActivationLayer::new(ActivationKind::Sigmoid);
    let input = Tensor::from_vec(vec![1, 1], vec![0.0]);
    let out = layer.forward_pass(&input);
    assert!(approx_eq(out.data[0], 0.5, 1e-6));
}

#[test]
fn tanh_at_zero_is_zero() {
    let mut layer = ActivationLayer::new(ActivationKind::Tanh);
    let input = Tensor::from_vec(vec![1, 1], vec![0.0]);
    let out = layer.forward_pass(&input);
    assert!(approx_eq(out.data[0], 0.0, 1e-6));
}

/// Activation backward pass shape must match the input shape.
#[test]
fn activation_backward_shape_matches_input() {
    for kind in [
        ActivationKind::Relu,
        ActivationKind::Sigmoid,
        ActivationKind::Tanh,
    ] {
        let mut layer = ActivationLayer::new(kind);
        let input = Tensor::from_vec(vec![3, 1], vec![1.0, -1.0, 0.5]);
        layer.forward_pass(&input);
        let d_out = Tensor::from_vec(vec![3, 1], vec![1.0, 1.0, 1.0]);
        let grad = layer.backward_pass(&d_out);
        assert_eq!(grad.shape, input.shape);
    }
}

// Gradient checker

/// The finite-difference gradient checker must pass for a randomly initialised
/// linear layer. This catches any bug in the analytical backward pass.
#[test]
fn gradient_check_passes_for_linear_layer() {
    let mut layer = LinearLayer::new(3, 2);
    layer.weight = Tensor::from_vec(vec![2, 3], vec![0.1, -0.2, 0.3, -0.1, 0.4, -0.3]);
    layer.bias = Tensor::from_vec(vec![2, 1], vec![0.05, -0.05]);
    let input = Tensor::from_vec(vec![3, 1], vec![0.5, -0.3, 0.8]);
    let target = Tensor::from_vec(vec![2, 1], vec![1.0, 0.0]);
    // eps=1e-4 is near-optimal for f32 (sqrt of machine epsilon ~1.2e-7)
    gradient_check(&layer, &input, &target, 1e-3);
}

// Adam optimizer

/// Adam must converge a single linear layer to fit a constant target.
#[test]
fn adam_converges_single_linear_layer() {
    let mut net = Network::new(vec![Box::new(LinearLayer::new_rand(2, 1))]);
    let mut opt = Adam::new(0.01, 0.9, 0.999, 1e-8);

    let mut x = Tensor::new(vec![2, 1]);
    x.data[0] = 1.0;
    x.data[1] = 1.0;
    let mut target = Tensor::new(vec![1, 1]);
    target.data[0] = 0.5;

    let initial_loss = mse(&net.forward(&x), &target);

    for _ in 0..500 {
        let out = net.forward(&x);
        let grad = d_mse(&out, &target);
        net.backward(&grad);
        net.update(&mut opt);
    }

    let final_loss = mse(&net.forward(&x), &target);
    assert!(
        final_loss < initial_loss * 0.01,
        "Adam failed to converge; initial={initial_loss:.4} final={final_loss:.4}"
    );
}

/// Adam with beta1=0, beta2=0 degenerates to vanilla gradient descent
/// scaled by alpha — loss must still decrease.
#[test]
fn adam_degenerate_still_decreases_loss() {
    let mut net = Network::new(vec![Box::new(LinearLayer::new_rand(1, 1))]);
    let mut opt = Adam::new(0.01, 0.0, 0.0, 1e-8);

    let mut x = Tensor::new(vec![1, 1]);
    x.data[0] = 1.0;
    let mut tgt = Tensor::new(vec![1, 1]);
    tgt.data[0] = 2.0;

    let l0 = mse(&net.forward(&x), &tgt);
    let g = d_mse(&net.forward(&x), &tgt);
    net.backward(&g);
    net.update(&mut opt);
    let l1 = mse(&net.forward(&x), &tgt);

    assert!(
        l1 < l0,
        "loss should decrease after Adam step; l0={l0:.4} l1={l1:.4}"
    );
}

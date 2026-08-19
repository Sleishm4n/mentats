//! Integration tests for loss functions.
//!
//! Each loss is verified against a known analytical result, and its gradient
//! is checked for shape consistency and correct direction (increasing the
//! prediction toward the target must decrease the loss).

use mentats::{
    loss::{
        cross_entropy::{cross_entropy, d_cross_entropy},
        kl_divergence::{d_kl_divergence_log_var, d_kl_divergence_mu, kl_divergence},
        mse::{d_mse, mse},
    },
    tensor::Tensor,
};

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

// MSE

#[test]
fn mse_zero_when_output_equals_target() {
    let t = Tensor::from_vec(vec![3, 1], vec![1.0, 2.0, 3.0]);
    assert!(approx_eq(mse(&t, &t), 0.0, 1e-6));
}

#[test]
fn mse_known_value() {
    // output=[0], target=[1] → MSE = (0-1)² / 1 = 1.0
    let out = Tensor::from_vec(vec![1, 1], vec![0.0]);
    let tgt = Tensor::from_vec(vec![1, 1], vec![1.0]);
    assert!(approx_eq(mse(&out, &tgt), 1.0, 1e-6));
}

#[test]
fn d_mse_shape_matches_output() {
    let out = Tensor::from_vec(vec![4, 1], vec![0.1, 0.2, 0.3, 0.4]);
    let tgt = Tensor::from_vec(vec![4, 1], vec![1.0, 1.0, 1.0, 1.0]);
    let grad = d_mse(&out, &tgt);
    assert_eq!(grad.shape, out.shape);
}

/// Moving the output one small step in the direction *opposite* to d_mse
/// should decrease the loss.
#[test]
fn d_mse_gradient_direction_is_correct() {
    let out = Tensor::from_vec(vec![2, 1], vec![0.0, 0.0]);
    let tgt = Tensor::from_vec(vec![2, 1], vec![1.0, 1.0]);
    let grad = d_mse(&out, &tgt);
    // gradient points away from target — negating it should improve loss
    let updated = out.sub(&grad.scale(0.1));
    assert!(mse(&updated, &tgt) < mse(&out, &tgt));
}

// Cross-entropy

/// For a uniform prediction over 2 classes with a one-hot target the
/// cross-entropy should equal ln(2) ≈ 0.693.
#[test]
fn cross_entropy_uniform_binary() {
    let out = Tensor::from_vec(vec![2, 1], vec![0.0, 0.0]); // logits → uniform after softmax
    let tgt = Tensor::from_vec(vec![2, 1], vec![1.0, 0.0]);
    let ce = cross_entropy(&out, &tgt);
    assert!(approx_eq(ce, 2_f32.ln(), 1e-4), "got {ce}");
}

/// A perfect prediction (very high logit for the correct class) should give
/// near-zero loss.
#[test]
fn cross_entropy_near_zero_for_confident_correct_prediction() {
    let out = Tensor::from_vec(vec![3, 1], vec![100.0, 0.0, 0.0]);
    let tgt = Tensor::from_vec(vec![3, 1], vec![1.0, 0.0, 0.0]);
    let ce = cross_entropy(&out, &tgt);
    assert!(ce < 1e-3, "expected near-zero loss, got {ce}");
}

#[test]
fn d_cross_entropy_shape_matches_output() {
    let out = Tensor::from_vec(vec![4, 1], vec![1.0, 2.0, 3.0, 4.0]);
    let tgt = Tensor::from_vec(vec![4, 1], vec![0.0, 0.0, 1.0, 0.0]);
    let grad = d_cross_entropy(&out, &tgt);
    assert_eq!(grad.shape, out.shape);
}

/// The gradient for the correct class should be negative (we want to push
/// its logit up), and gradients should sum to zero.
#[test]
fn d_cross_entropy_gradients_sum_to_zero() {
    let out = Tensor::from_vec(vec![3, 1], vec![1.0, 2.0, 3.0]);
    let tgt = Tensor::from_vec(vec![3, 1], vec![0.0, 1.0, 0.0]);
    let grad = d_cross_entropy(&out, &tgt);
    let sum: f32 = grad.data.iter().sum();
    assert!(
        approx_eq(sum, 0.0, 1e-5),
        "gradients should sum to 0, got {sum}"
    );
}

// KL divergence

/// KL(N(0,1) || N(0,1)) = 0.  mu=0, log_var=0 (var=1).
#[test]
fn kl_divergence_zero_for_standard_normal() {
    let mu = Tensor::from_vec(vec![2, 1], vec![0.0, 0.0]);
    let log_var = Tensor::from_vec(vec![2, 1], vec![0.0, 0.0]);
    let kl = kl_divergence(&mu, &log_var);
    assert!(approx_eq(kl, 0.0, 1e-6), "got {kl}");
}

/// KL is non-negative.
#[test]
fn kl_divergence_is_non_negative() {
    let mu = Tensor::from_vec(vec![3, 1], vec![0.5, -1.0, 2.0]);
    let log_var = Tensor::from_vec(vec![3, 1], vec![-0.5, 0.3, -1.0]);
    assert!(kl_divergence(&mu, &log_var) >= 0.0);
}

#[test]
fn d_kl_mu_shape_matches_input() {
    let mu = Tensor::from_vec(vec![4, 1], vec![0.1, 0.2, 0.3, 0.4]);
    let log_var = Tensor::from_vec(vec![4, 1], vec![-0.5, 0.3, -1.0, 0.2]);
    assert_eq!(d_kl_divergence_mu(&mu, &log_var).shape, mu.shape);
}

#[test]
fn d_kl_log_var_shape_matches_input() {
    let mu = Tensor::from_vec(vec![4, 1], vec![0.1, 0.2, 0.3, 0.4]);
    let log_var = Tensor::from_vec(vec![4, 1], vec![0.1, 0.2, 0.3, 0.4]);
    assert_eq!(d_kl_divergence_log_var(&mu, &log_var).shape, log_var.shape);
}

/// d/d(log_var) KL at log_var=0 should be 0 (minimum of KL w.r.t. variance).
#[test]
fn d_kl_log_var_zero_at_log_var_zero() {
    let mu = Tensor::from_vec(vec![2, 1], vec![0.1, 0.2]);
    let log_var = Tensor::from_vec(vec![2, 1], vec![0.0, 0.0]);
    let grad = d_kl_divergence_log_var(&mu, &log_var);
    for &v in &grad.data {
        assert!(approx_eq(v, 0.0, 1e-6), "expected 0 gradient, got {v}");
    }
}

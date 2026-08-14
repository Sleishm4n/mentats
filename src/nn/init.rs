//! Weight initialisation
//!
//! Both schemes scale the intial weights so that activation variance is
//! roughly preserved layer to layer, which stops deep stacks from saturating
//! or exploding before training starts. Pick based on the activation that follows:
//! Xavier for sigmoid/tanh, Kaiming for ReLU
use crate::tensor::Tensor;
use rand::Rng;

/// Xavier/Glorto uniform initialisation
///
/// Samples uniformly from `[-limit, limit]` where
/// `limit = sqrt(6 / (in_features + out_features))`. Returns a tensor of
/// shape `[out_features, in_features]`
///
/// # Panics
///
/// Panics if either dimension is zero
pub fn xavier_uniform(in_features: usize, out_features: usize) -> Tensor {
    assert!(in_features > 0, "in_features must be > 0");
    assert!(out_features > 0, "out_features must be > 0");

    let denom = (in_features + out_features) as f32;
    let limit = (6.0 / denom).sqrt();
    Tensor::rand_range(vec![out_features, in_features], -limit, limit)
}

/// Kaiming/He uniform initialisation
///
/// Samples from a normal distribution with standard deviation
/// `sqrt(2 / in_features)`. The factor of 2 compensates for ReLU zeroing half
/// its inputs. Returns a tensor of shape `[out_features, in_features]`
///
/// # Panics
///
/// Panics if either dimension is zero
pub fn kaiming_normal(in_features: usize, out_features: usize) -> Tensor {
    assert!(in_features > 0, "in_features must be > 0");
    assert!(out_features > 0, "out_features must be > 0");

    let std = (2.0 / in_features as f32).sqrt();
    let size = in_features * out_features;
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(size);

    for _ in 0..size {
        data.push(sample_standard_normal(&mut rng) * std);
    }

    Tensor::from_vec(vec![out_features, in_features], data)
}

fn sample_standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    let u1 = loop {
        let cand: f32 = rng.gen_range(0.0..1.0);
        if cand > 0.0 {
            break cand;
        }
    };
    let u2: f32 = rng.gen_range(0.0..1.0);

    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    r * theta.cos()
}

//! Weight initialisation
//!
//! Both schemes scale the intial weights so that activation variance is
//! roughly preserved layer to layer, which stops deep stacks from saturating
//! or exploding before training starts. Pick based on the activation that follows:
//! Xavier for sigmoid/tanh, Kaiming for ReLU
use crate::tensor::Tensor;
use rand::Rng;

/// Xavier/Glorto uniform initialisation with a custom RNG
///
/// Samples uniformly from `[-limit, limit]` where
/// `limit = sqrt(6 / (in_features + out_features))`. Returns a tensor of
/// shape `[out_features, in_features]`
///
/// # Panics
///
/// Panics if either dimension is zero
pub fn xavier_uniform_with_rng<R: Rng + ?Sized>(
    in_features: usize,
    out_features: usize,
    rng: &mut R,
) -> Tensor {
    assert!(in_features > 0, "in_features must be > 0");
    assert!(out_features > 0, "out_features must be > 0");

    let denom = (in_features + out_features) as f32;
    let limit = (6.0 / denom).sqrt();
    Tensor::rand_range_with_rng(vec![out_features, in_features], -limit, limit, rng)
}

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
    let mut rng = rand::thread_rng();
    xavier_uniform_with_rng(in_features, out_features, &mut rng)
}

/// Kaiming/He uniform initialisation with a custom RNG
///
/// Samples from a normal distribution with standard deviation
/// `sqrt(2 / in_features)`. The factor of 2 compensates for ReLU zeroing half
/// its inputs. Returns a tensor of shape `[out_features, in_features]`
///
/// # Panics
///
/// Panics if either dimension is zero
pub fn kaiming_normal_with_rng<R: Rng + ?Sized>(
    in_features: usize,
    out_features: usize,
    rng: &mut R,
) -> Tensor {
    assert!(in_features > 0, "in_features must be > 0");
    assert!(out_features > 0, "out_features must be > 0");

    let std = (2.0 / in_features as f32).sqrt();
    let size = in_features * out_features;
    let mut data = Vec::with_capacity(size);

    for _ in 0..size {
        data.push(sample_standard_normal(rng) * std);
    }

    Tensor::from_vec(vec![out_features, in_features], data)
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
    let mut rng = rand::thread_rng();
    kaiming_normal_with_rng(in_features, out_features, &mut rng)
}

/// Kaiming/He normal initialisation for 2D convolution filters.
///
/// Samples from a normal distribution with standard deviation
/// `sqrt(2 / (in_channels * kh * kw))`. Returns a 4D tensor of shape
/// `[out_channels, in_channels, kh, kw]`.
///
/// # Panics
///
/// Panics if any dimension is zero.
pub fn kaiming_normal_conv_with_rng<R: Rng + ?Sized>(
    in_channels: usize,
    out_channels: usize,
    kh: usize,
    kw: usize,
    rng: &mut R,
) -> Tensor {
    assert!(in_channels > 0, "in_channels must be > 0");
    assert!(out_channels > 0, "out_channels must be > 0");
    assert!(kh > 0, "kh must be > 0");
    assert!(kw > 0, "kw must be > 0");

    let fan_in = in_channels * kh * kw;
    let std = (2.0 / fan_in as f32).sqrt();
    let size = out_channels * in_channels * kh * kw;
    let mut data = Vec::with_capacity(size);

    for _ in 0..size {
        data.push(sample_standard_normal(rng) * std);
    }

    Tensor::from_vec(vec![out_channels, in_channels, kh, kw], data)
}

/// Convenience wrapper for [`kaiming_normal_conv_with_rng`] using [`rand::thread_rng`].
pub fn kaiming_normal_conv(
    in_channels: usize,
    out_channels: usize,
    kh: usize,
    kw: usize,
) -> Tensor {
    let mut rng = rand::thread_rng();
    kaiming_normal_conv_with_rng(in_channels, out_channels, kh, kw, &mut rng)
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_xavier_uniform_shape_and_bounds() {
        let mut rng = StdRng::seed_from_u64(42);
        let in_f = 100;
        let out_f = 50;
        let t = xavier_uniform_with_rng(in_f, out_f, &mut rng);

        assert_eq!(t.shape, vec![out_f, in_f]);
        let limit = (6.0 / (in_f + out_f) as f32).sqrt();

        for &val in &t.data {
            assert!(
                val >= -limit && val < limit,
                "val={val} outside limit={limit}"
            );
        }
    }

    #[test]
    fn test_kaiming_normal_shape_and_statistics() {
        let mut rng = StdRng::seed_from_u64(42);
        let in_f = 1000;
        let out_f = 1000;
        let t = kaiming_normal_with_rng(in_f, out_f, &mut rng);

        assert_eq!(t.shape, vec![out_f, in_f]);

        let mean: f32 = t.data.iter().sum::<f32>() / t.data.len() as f32;
        assert!(mean.abs() < 0.01, "mean was {mean}");

        let expected_var = 2.0 / in_f as f32;
        let variance: f32 =
            t.data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / t.data.len() as f32;

        assert!(
            (variance - expected_var).abs() < 0.001,
            "variance was {variance}, expected {expected_var}"
        );
    }

    #[test]
    fn test_kaiming_conv_shape_and_stats() {
        let mut rng = StdRng::seed_from_u64(42);
        let in_c = 16;
        let out_c = 16;
        let kh = 3;
        let kw = 3;
        let tensor = kaiming_normal_conv_with_rng(in_c, out_c, kh, kw, &mut rng);

        assert_eq!(tensor.shape, vec![out_c, in_c, kh, kw]);
        assert_eq!(tensor.data.len(), out_c * in_c * kh * kw);

        // Check empirical statistics over 2304 samples
        let n = tensor.data.len() as f32;
        let mean: f32 = tensor.data.iter().sum::<f32>() / n;
        let variance: f32 = tensor.data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
        let empirical_std = variance.sqrt();

        let fan_in = (in_c * kh * kw) as f32;
        let expected_std = (2.0 / fan_in).sqrt();

        assert!(mean.abs() < 0.05, "Mean was {mean}, expected near 0.0");
        let rel_err = (empirical_std - expected_std).abs() / expected_std;
        assert!(
            rel_err < 0.15,
            "Expected std {expected_std}, got {empirical_std} (err {rel_err})"
        );
    }

    #[test]
    #[should_panic(expected = "in_channels must be > 0")]
    fn test_kaiming_conv_panics_on_zero_in_channels() {
        kaiming_normal_conv(0, 4, 3, 3);
    }

    #[test]
    #[should_panic(expected = "kh must be > 0")]
    fn test_kaiming_conv_panics_on_zero_kh() {
        kaiming_normal_conv(4, 4, 0, 3);
    }
}

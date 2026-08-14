//! Random initialisation constructors for `[Tensor]`
//!
//! Layer specific checmes (Xavier, Kaiming) live in `[crate::nn::init]` and
//! are built on top of these primitives
use crate::tensor::Tensor;
use rand::prelude::*;

impl Tensor {
    /// Creates a tensor of the given `shape` filled with values drawn
    /// uniformly from `[min, max)`
    ///
    /// # Panics
    ///
    /// Panics if `min >= max` or if `shape` is empty
    pub fn rand_range(shape: Vec<usize>, min: f32, max: f32) -> Tensor {
        assert!(min < max, "min cannot be larger than max");
        assert!(!shape.is_empty(), "shape must be at least 1D");

        let size = shape.iter().product();

        let mut rng = rand::thread_rng();
        let mut vec = Vec::with_capacity(size);

        for _ in 0..size {
            let val = rng.gen_range(min..max);
            vec.push(val);
        }
        Tensor::from_vec(shape, vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rand_range_io_shape_match() {
        let tensor = Tensor::rand_range(vec![3, 2], -1.0, 1.0);

        assert_eq!(tensor.shape, vec![3, 2]);
    }

    #[test]
    fn test_rand_range_values_within_minmax() {
        let tensor = Tensor::rand_range(vec![3, 2], -10.0, 10.0);

        assert!(tensor.tensor_min() >= -10.0);
        assert!(tensor.tensor_max() <= 10.0);
    }

    #[test]
    #[should_panic(expected = "min cannot be larger than max")]
    fn test_rand_range_panics_on_min_larger_max() {
        let _tensor = Tensor::rand_range(vec![3, 2], 2.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "at least 1D")]
    fn test_rand_range_panics_on_empty() {
        let _tensor = Tensor::rand_range(vec![], 1.0, 2.0);
    }
}

//! Elementwise, shape and lin-alg operations on [`Tensor`]
//!
//! Everything is implemented on the flat data buffer
//! Ops that change layout ([`Tensor::permute`]) rewrite the
//! strides rather than the data, meaning they are cheap
use std::assert_eq;

use crate::tensor::Tensor;

impl Tensor {
    /// Elementwise add. Both tensors need the same shape
    ///
    /// # Panics
    ///
    /// Panics if the shapes differ
    pub fn add(&self, other: &Tensor) -> Tensor {
        self.zip_map(other, |a, b| a + b)
    }

    /// Elementwise subtraction (`self - other`). Both tensors must have the same shape
    ///
    /// # Panics
    ///
    /// Panics if shapes differ
    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.zip_map(other, |a, b| a - b)
    }

    /// Multiples every element by scalar `n`
    pub fn scale(&self, n: f32) -> Tensor {
        self.map(|x| x * n)
    }

    /// Returns the largest element in the tensor
    ///
    /// Returns `-f32::INFINITY` for an emnpty tensor
    pub fn tensor_max(&self) -> f32 {
        let mut max = -f32::INFINITY;

        for val in &self.data {
            if val > &max {
                max = *val;
            }
        }
        max
    }

    /// Returns the smallest element in the tensor
    ///
    /// Returns `-f32::INFINITY` for an emnpty tensor
    pub fn tensor_min(&self) -> f32 {
        let mut min = f32::INFINITY;

        for val in &self.data {
            if val < &min {
                min = *val;
            }
        }

        min
    }

    /// 2D matric multiplication: `[m, n] @ [n, p] -> [m, p]`
    ///
    /// For batched inputs use [`Tensor::matmul_batched`] or [`Tensor::matmul_batched_broadcast`]
    /// instead.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not rank 2, or if the inner dims don't match
    /// (`self.shape[1] != other.shape[1]`)
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert!(self.shape.len() == 2);
        assert!(self.shape[1] == other.shape[0]);

        let mut c = Tensor::new(vec![self.shape[0], other.shape[1]]);

        for i in 0..self.shape[0] {
            for k in 0..self.shape[1] {
                let a = self.get(&[i, k]);

                for j in 0..other.shape[1] {
                    let prev = c.get(&[i, j]);
                    c.set(&[i, j], prev + a * other.get(&[k, j]));
                }
            }
        }
        c
    }

    /// Swaps the two axes of a rank 2 tensor. Shorthand for `permute(&[1, 0])`
    ///
    /// # Panics
    ///
    /// Panics if tensor is not rank 2
    pub fn transpose(&self) -> Tensor {
        self.permute(&[1, 0])
    }

    /// Reorders the tensor's axes
    ///
    /// `axes[i]` is the axis of `self` that becomes axis `i` of the result.
    /// Only the shape and strides are rewritten, the underlying data is copied as is,
    /// so returned tensor may be non=contiguous
    ///
    /// # Panics
    ///
    /// Panics if `axes` is not the same length as the tensor's rank
    pub fn permute(&self, axes: &[usize]) -> Tensor {
        assert_eq!(
            axes.len(),
            self.shape.len(),
            "axes length must match tensor rank"
        );

        let mut seen = vec![false; self.shape.len()];
        let mut new_shape: Vec<usize> = Vec::new();
        let mut new_strides: Vec<usize> = Vec::new();

        for &axis in axes {
            assert!(axis < self.shape.len(), "axis out of bound");
            assert!(!seen[axis], "axes must form a permutation");
            seen[axis] = true;
            new_shape.push(self.shape[axis]);
            new_strides.push(self.strides[axis]);
        }

        Tensor {
            shape: new_shape,
            strides: new_strides,
            data: self.data.clone(),
        }
    }

    /// Applies `f` to every element, returning a new tensor of the same shape
    pub fn map<F: Fn(f32) -> f32>(&self, f: F) -> Tensor {
        let mut result = Tensor::new(self.shape.clone());
        let mut index = vec![0; self.shape.len()];

        loop {
            let val = self.get(&index);
            result.set(&index, f(val));

            if !Self::next_index(&mut index, &self.shape) {
                break;
            }
        }

        result
    }

    /// Apples `f` to each pair of corresponding elements from `self` and
    /// `other`, returning a new tensor of the same shape
    ///
    /// # Panics
    ///
    /// Panics if shapes differ
    pub fn zip_map<F: Fn(f32, f32) -> f32>(&self, other: &Tensor, f: F) -> Tensor {
        assert_eq!(self.shape, other.shape);

        let mut result = Tensor::new(self.shape.clone());
        let mut index = vec![0; self.shape.len()];

        loop {
            let val = self.get(&index);
            let other_val = other.get(&index);
            result.set(&index, f(val, other_val));

            if !Self::next_index(&mut index, &self.shape) {
                break;
            }
        }

        result
    }

    fn next_index(index: &mut [usize], shape: &[usize]) -> bool {
        for i in (0..index.len()).rev() {
            index[i] += 1;
            if index[i] < shape[i] {
                return true;
            }
            index[i] = 0;
        }
        false
    }

    /// Squares every element `(x * x)`, returns a new tensor
    pub fn elementwise_square(&self) -> Tensor {
        self.map(|x| x * x)
    }

    /// Concatenates two unbatched tensors into a single `[features, 1]` column vector
    ///
    /// Used to glue a latent vector onto a one-hot class label for conditional
    /// models. Accepts any 1D or 2D input shape (`[N]`, `[N, 1]` or `[1, N]`)
    /// and always normalises the result to a column vector
    pub fn concat_features_2d(&self, other: &Tensor) -> Tensor {
        // Accepts any 1D or 2D shape ([N], [N, 1], or [1, N])
        // and returns a normalized [features, 1] column vector.
        let left_len = self.data.len();
        let right_len = other.data.len();

        let mut data = Vec::with_capacity(left_len + right_len);
        data.extend_from_slice(&self.data);
        data.extend_from_slice(&other.data);

        Tensor::from_vec(vec![left_len + right_len, 1], data)
    }

    /// Batched version of [`Tensor::concat_features_2d`]
    ///
    /// Concatenates along the feature axis:
    /// `[batch, a, 1]` and `[batch, b, 1]` -> `[batch, a + b, 1]`
    ///
    /// # Panics
    ///
    /// Panics if either tensor is not rank 3, if the batch sizes are different
    /// or if either final dimension is not 1
    pub fn concat_features_batch(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape.len(),
            3,
            "self must be rank-3 [batch, features, 1]"
        );
        assert_eq!(
            other.shape.len(),
            3,
            "other must be rank-3 [batch, features, 1]"
        );
        assert_eq!(self.shape[0], other.shape[0], "batch sizes must match");
        assert_eq!(self.shape[2], 1, "self trailing dimension must be 1");
        assert_eq!(other.shape[2], 1, "other trailing dimension must be 1");

        let batch = self.shape[0];
        let left_features = self.shape[1];
        let right_features = other.shape[1];
        let mut data = Vec::with_capacity(batch * (left_features + right_features));

        for b in 0..batch {
            let ls = b * left_features;
            data.extend_from_slice(&self.data[ls..ls + left_features]);
            let rs = b * right_features;
            data.extend_from_slice(&other.data[rs..rs + right_features]); // Fixed to other.data
        }

        Tensor::from_vec(vec![batch, left_features + right_features, 1], data)
    }

    /// Keeps the first `keep` features of a `[batch, features, 1]` tensor,
    /// dropping the rest
    ///
    /// The inverse of [`Tensor::concat_features_batch`] on the backwards pass:
    /// gradients flowing back through a concatenated input are trimmed to the
    /// slice that belongs to the upstream layer
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not rank 3, if the trailing dimensions in not 1
    /// or if `keep` exceeds the number of features
    pub fn take_first_features_batch(&self, keep: usize) -> Tensor {
        assert_eq!(
            self.shape.len(),
            3,
            "tensor must be rank-3 [batch, features, 1]"
        );
        assert_eq!(self.shape[2], 1, "trailing dimension must be 1");
        assert!(
            keep <= self.shape[1],
            "cannot keep more features ({}) than tensor has ({})",
            keep,
            self.shape[1]
        );

        let batch = self.shape[0];
        let features = self.shape[1];
        let mut data = Vec::with_capacity(batch * keep);

        for b in 0..batch {
            let start = b * features;
            data.extend_from_slice(&self.data[start..start + keep]);
        }

        Tensor::from_vec(vec![batch, keep, 1], data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul() {
        let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::from_vec(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_transpose() {
        let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose();
        assert_eq!(t.shape, vec![3, 2]);
        assert_eq!(t.get(&[0, 0]), 1.0);
        assert_eq!(t.get(&[0, 1]), 4.0);
        assert_eq!(t.get(&[1, 0]), 2.0);
        assert_eq!(t.get(&[2, 1]), 6.0);
    }

    #[test]
    fn test_matmul_transposed() {
        // A @ A^T should give a symmetric matrix
        let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let at = a.transpose();
        let c = a.matmul(&at);
        assert_eq!(c.shape, vec![2, 2]);
        // c[0][1] should equal c[1][0]
        assert_eq!(c.get(&[0, 1]), c.get(&[1, 0]));
    }

    #[test]
    fn test_map_transposed() {
        let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose();
        let result = t.map(|x| x * 2.0);
        assert_eq!(result.get(&[0, 0]), 2.0);
        assert_eq!(result.get(&[0, 1]), 8.0); // was 4.0 before map
        assert_eq!(result.get(&[1, 0]), 4.0); // was 2.0 before map
    }

    #[test]
    fn test_3d_indexing_matches_rm_layout() {
        let shape = vec![32, 32, 3];
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(shape, data);

        assert_eq!(tensor.get(&[0, 0, 0]), 0.0);
        assert_eq!(tensor.get(&[0, 0, 1]), 1.0);
        assert_eq!(tensor.get(&[0, 1, 0]), 3.0);
        assert_eq!(tensor.get(&[1, 0, 0]), 96.0);
        assert_eq!(tensor.get(&[31, 31, 2]), (size - 1) as f32);
    }

    #[test]
    fn test_3d_permute_preserves_vals() {
        let shape = vec![32, 32, 3];
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(shape, data);
        let channel_first = tensor.permute(&[2, 0, 1]);

        assert_eq!(channel_first.shape, vec![3, 32, 32]);
        assert_eq!(channel_first.get(&[0, 0, 0]), tensor.get(&[0, 0, 0]));
        assert_eq!(channel_first.get(&[1, 0, 0]), tensor.get(&[0, 0, 1]));
        assert_eq!(channel_first.get(&[2, 31, 31]), tensor.get(&[31, 31, 2]));
    }

    #[test]
    fn test_map_over_permuted_3d_tensor_respects_view_strides() {
        let shape = vec![32, 32, 3];
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(shape, data);
        let permuted = tensor.permute(&[2, 0, 1]);
        let mapped = permuted.map(|x| x + 0.5);

        assert_eq!(mapped.shape, vec![3, 32, 32]);
        assert_eq!(mapped.get(&[0, 0, 0]), tensor.get(&[0, 0, 0]) + 0.5);
        assert_eq!(mapped.get(&[1, 0, 0]), tensor.get(&[0, 0, 1]) + 0.5);
        assert_eq!(mapped.get(&[2, 31, 31]), tensor.get(&[31, 31, 2]) + 0.5);
    }

    #[test]
    #[should_panic(expected = "axes length must match tensor rank")]
    fn test_permute_panics_on_rank_mismatch() {
        let tensor = Tensor::new(vec![2, 3, 4]);
        tensor.permute(&[0, 1]);
    }

    #[test]
    #[should_panic(expected = "axes must form a permutation")]
    fn test_permute_panics_on_duplicate_axes() {
        let tensor = Tensor::new(vec![2, 3, 4]);
        tensor.permute(&[0, 0, 2]);
    }
}

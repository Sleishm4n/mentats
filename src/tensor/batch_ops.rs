//! Batched tensor operations
//!
//! Layers run over a whole mini-batch at once, whcih means one rank 3
//! `[batch, features, 1]` tensor rather than a `Vec` of rank 2 samples
//! These helpers provide the batched matmuls and batch-axis reduction that
//! the forward and backwards passes need
use crate::tensor::Tensor;

impl Tensor {
    /// Batched matrix multiply: `[batch, m, n] @ [n, p] -> [batch, m, p]`
    ///
    /// `other` is shared across every batch element, this is the shape used when
    /// multiplying a batch of activations by a weight matrix
    ///
    /// # Panics
    ///
    /// Panics if `self` if not rank 3, `other` is not rank 2, ot the inner dims
    /// don't match
    pub fn matmul_batched(&self, other: &Tensor) -> Tensor {
        assert!(
            self.shape.len() == 3 && other.shape.len() == 2,
            "batched matmul expects [batch, m, n] @ [n, p] -> [batch, m, p]"
        );

        let batch_size = self.shape[0];
        let m = self.shape[1];
        let n = self.shape[2];
        let p = other.shape[1];

        assert_eq!(n, other.shape[0], "dimension mismatch for matmul");

        // `other` is shared across every batch, so flatten it once outside the loop
        // rather than calling other.get() (stride-walked, bounds-checked) on every
        // (b, i, j, k), this is the same caching trick as the plain matmul rewrite
        let other_flat: Vec<f32> = (0..n)
            .flat_map(|k| (0..p).map(move |j| other.get(&[k, j])))
            .collect();

        let mut result = Tensor::new(vec![batch_size, m, p]);

        for b in 0..batch_size {
            for i in 0..m {
                // Pre-extract this row so the inner k-loop below hits plain Vec
                // indexing instead of self.get() (stride math) on every k.
                let self_row: Vec<f32> = (0..n).map(|k| self.get(&[b, i, k])).collect();
                for j in 0..p {
                    let mut sum: f32 = 0.0;

                    for k in 0..n {
                        sum += self_row[k] * other_flat[k * p + j];
                    }
                    result.set(&[b, i, j], sum);
                }
            }
        }

        result
    }

    /// Broadcast matrix multiply: `[m, n] @ [batch, n, p] -> [batch, m, p]`
    ///
    /// The mirror of `[Tensor::matmul_batched]`, here the *left* opreand
    /// is shared across the batch. Used on the backwards pass, where a single
    /// weight matrix is applied to a batch of upstream gradients
    ///
    /// # Panics
    ///
    /// Panics if `self` is not rank 2, `other` is not rank 3 or the inner
    /// dims don't match
    pub fn matmul_batched_broadcast(&self, other: &Tensor) -> Tensor {
        assert!(
            self.shape.len() == 2 && other.shape.len() == 3,
            "broadcast matmul expects [m, n] @ [batch, n, p] -> [batch, m, p]"
        );

        let batch_size = other.shape[0];
        let m = self.shape[0];
        let n = self.shape[1];
        let p = other.shape[2];

        assert_eq!(n, other.shape[1], "dimension mismatch for matmul");

        // Unlike matmul_batched, here it's `self` that's shared across batches
        // (broadcast), so cache all its rows once up front.
        let self_rows: Vec<Vec<f32>> = (0..m)
            .map(|i| (0..n).map(move |k| self.get(&[i, k])).collect())
            .collect();

        let mut result = Tensor::new(vec![batch_size, m, p]);

        for b in 0..batch_size {
            // `other` differs per batch, so this flatten has to stay inside the
            // loop, indexed by b, not shared like self_rows above.
            let other_flat: Vec<f32> = (0..n)
                .flat_map(|k| (0..p).map(move |j| other.get(&[b, k, j])))
                .collect();
            for (i, row) in self_rows.iter().enumerate() {
                for j in 0..p {
                    let mut sum = 0.0;

                    for k in 0..n {
                        sum += row[k] * other_flat[k * p + j];
                    }
                    result.set(&[b, i, j], sum);
                }
            }
        }

        result
    }

    /// Sums over the leading (batch) axis, returning a tensor with shape
    /// `self.shape[1..]`
    ///
    /// Used to accumulate per-sample parameter gradients into a single
    /// gradient for the batch. A rank 1 input has no separate batch azis
    /// and collapes to a `[1]` scalar tensor
    ///
    /// # Panics
    ///
    /// Panics if the tensor has no dimensions
    pub fn sum_batch(&self) -> Tensor {
        assert!(
            !self.shape.is_empty(),
            "tensor must have at least 1 dimension"
        );

        // 1D input has no batch dim to sum over separately from the values
        // themselves, collapse straight to a scalar.
        if self.shape.len() == 1 {
            let sum = self.data.iter().sum();
            return Tensor::from_vec(vec![1], vec![sum]);
        }

        let batch_size = self.shape[0];
        let remaining_shape = self.shape[1..].to_vec();
        let remaining_size: usize = remaining_shape.iter().product();

        let mut result = Tensor::new(remaining_shape);
        let mut result_data = vec![0.0; remaining_size];

        // Relies on self being contiguous/default-strided: indexes the flat
        // buffer directly as `b * remaining_size + i` rather than going through
        // self.get(),
        for b in 0..batch_size {
            let batch_start = b * remaining_size;
            let batch = &self.data[batch_start..batch_start + remaining_size];

            for (result, value) in result_data.iter_mut().zip(batch) {
                *result += value;
            }
        }

        result.data = result_data;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_batched() {
        let batch_size = 2;
        let m = 2;
        let n = 3;
        let p = 2;

        let a = Tensor::from_vec(vec![batch_size, m, n], (1..=12).map(|x| x as f32).collect());
        let b = Tensor::from_vec(vec![n, p], (1..=6).map(|x| x as f32).collect());

        let c = a.matmul_batched(&b);

        assert_eq!(c.shape, vec![batch_size, m, p]);

        let first_batch_2d = Tensor::from_vec(vec![m, n], a.data[0..6].to_vec());
        let expected = first_batch_2d.matmul(&b);

        for i in 0..m {
            for j in 0..p {
                assert!((c.get(&[0, i, j]) - expected.get(&[i, j])).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_sum_batch_2d() {
        // [batch=3, features=2]
        let a = Tensor::from_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let s = a.sum_batch();

        assert_eq!(s.shape, vec![2]);
        assert_eq!(s.data, vec![9.0, 12.0]); // (1+3+5), (2+4+6)
    }

    #[test]
    fn test_sum_batch_1d() {
        let a = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let s = a.sum_batch();

        assert_eq!(s.shape, vec![1]);
        assert_eq!(s.data, vec![10.0]);
    }

    #[test]
    fn test_sum_batch_3d() {
        // [batch=2, m=2, n=2]
        let a = Tensor::from_vec(vec![2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let s = a.sum_batch();

        assert_eq!(s.shape, vec![2, 2]);
        assert_eq!(s.data, vec![6.0, 8.0, 10.0, 12.0]); // elementwise across batch
    }

    #[test]
    #[should_panic(expected = "batched matmul expects")]
    fn test_matmul_batched_panics_on_wrong_rank() {
        let a = Tensor::new(vec![2, 3]); // rank 2, needs rank 3
        let b = Tensor::new(vec![3, 2]);
        let _ = a.matmul_batched(&b);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch for matmul")]
    fn test_matmul_batched_panics_on_inner_dim_mismatch() {
        let a = Tensor::new(vec![2, 2, 3]); // n = 3
        let b = Tensor::new(vec![4, 2]); // expects n = 4
        let _ = a.matmul_batched(&b);
    }

    #[test]
    fn test_matmul_batched_broadcast_correctness() {
        let a = Tensor::from_vec(vec![2, 3], (1..=6).map(|x| x as f32).collect()); // [m=2, n=3]
        let b = Tensor::from_vec(
            vec![2, 3, 2], // [batch=2, n=3, p=2]
            (1..=12).map(|x| x as f32).collect(),
        );

        let c = a.matmul_batched_broadcast(&b);
        assert_eq!(c.shape, vec![2, 2, 2]);

        // verify against unbatched matmul per-slice
        let b0 = Tensor::from_vec(vec![3, 2], b.data[0..6].to_vec());
        let b1 = Tensor::from_vec(vec![3, 2], b.data[6..12].to_vec());
        let expected0 = a.matmul(&b0);
        let expected1 = a.matmul(&b1);

        for i in 0..2 {
            for j in 0..2 {
                assert!((c.get(&[0, i, j]) - expected0.get(&[i, j])).abs() < 1e-5);
                assert!((c.get(&[1, i, j]) - expected1.get(&[i, j])).abs() < 1e-5);
            }
        }

        // batches must actually differ, catching a b-index bug in other_flat
        assert_ne!(c.get(&[0, 0, 0]), c.get(&[1, 0, 0]));
    }

    #[test]
    #[should_panic(expected = "broadcast matmul expects")]
    fn test_matmul_batched_broadcast_panics_on_wrong_rank() {
        let a = Tensor::new(vec![2, 3, 4]); // rank 3, needs rank 2
        let b = Tensor::new(vec![2, 4, 3]);
        let _ = a.matmul_batched_broadcast(&b);
    }
}

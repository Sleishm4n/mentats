use crate::tensor::Tensor;

impl Tensor {
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

        let other_flat: Vec<f32> = (0..n).flat_map(|k| (0..p).map(move |j| other.get(&[k, j]))).collect();

        let mut result = Tensor::new(vec![batch_size, m, p]);

        for b in 0..batch_size {
            for i in 0..m {
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

        let self_rows: Vec<Vec<f32>> = (0..m).map(|i| (0..n).map(move |k| self.get(&[i, k])).collect()).collect();

        let mut result = Tensor::new(vec![batch_size, m, p]);

        for b in 0..batch_size {
            let other_flat: Vec<f32> = (0..n).flat_map(|k| (0..p).map(move |j| other.get(&[b, k, j]))).collect();
            for i in 0..m {
                
                for j in 0..p {
                    let mut sum = 0.0;

                    for k in 0..n {
                        sum += self_rows[i][k] * other_flat[k * p +j];
                    }
                    result.set(&[b, i, j], sum);
                }
            }
        }

        result
    }

    pub fn sum_batch(&self) -> Tensor {
        assert!(self.shape.len() >= 1, "tensor must have at least 1 dimension");

        if self.shape.len() == 1 {
            let sum = self.data.iter().sum();
            return Tensor::from_vec(vec![1], vec![sum]);
        }

        let batch_size = self.shape[0];
        let remaining_shape = self.shape[1..].to_vec();
        let remaining_size: usize = remaining_shape.iter().product();

        let mut result = Tensor::new(remaining_shape);
        let mut result_data = vec![0.0; remaining_size];

        for b in 0..batch_size {
            for i in 0..remaining_size {
                result_data[i] += self.data[b * remaining_size + i];
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
}

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

        let result_shape = vec![batch_size, m, p];
        let mut result = Tensor::new(result_shape);

        for b in 0..batch_size {
            for i in 0..m {
                for k in 0..n {
                    let a = self.get(&[b, i, k]);

                    for j in 0..p {
                        let prev = result.get(&[b, i, j]);
                        result.set(&[b, i, j], prev + a * other.get(&[k, j]));
                    }
                }
            }
        }

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

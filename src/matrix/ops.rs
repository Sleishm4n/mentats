use crate::matrix::Matrix;
use std::{f32::INFINITY, iter::zip};

pub fn square(x: f32) -> f32 {
    x * x
}
pub fn cube(x: f32) -> f32 {
    x * x * x
}
pub fn add_scalars(a: f32, b: f32) -> f32 {
    a + b
}

pub fn exp(x: f32) -> f32 {
    x.exp()
}

impl Matrix {
    pub fn scale_inplace(&mut self, n: f32) {
        for val in &mut self.data {
            *val *= n;
        }
    }

    pub fn scale(&self, n: f32) -> Matrix {
        let mut vec = Vec::with_capacity(self.data.len());
        for val in &self.data {
            let v = val * n;
            vec.push(v);
        }
        Matrix::from_vec(self.rows, self.cols, vec)
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut vec = Vec::with_capacity(self.data.len());
        for row in 0..self.rows {
            for col in 0..self.cols {
                let v = self.get(row, col) + other.get(row, col);
                vec.push(v);
            }
        }
        Matrix::from_vec(self.rows, self.cols, vec)
    }

    pub fn sub(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut vec = Vec::with_capacity(self.data.len());
        for row in 0..self.rows {
            for col in 0..self.cols {
                let v = self.get(row, col) - other.get(row, col);
                vec.push(v);
            }
        }
        Matrix::from_vec(self.rows, self.cols, vec)
    }

    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows);

        let mut c = Matrix::new(self.rows, other.cols);

        for i in 0..self.rows {
            for k in 0..self.cols {
                let a_index = i * self.cols + k;
                let a = self.data[a_index];

                for j in 0..other.cols {
                    let b_index = k * other.cols + j;
                    let c_index = i * c.cols + j;
                    c.data[c_index] += a * other.data[b_index];
                }
            }
        }
        c
    }

    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);
        for row in 0..self.rows {
            for col in 0..self.cols {
                result.set(col, row, self.get(row, col));
            }
        }
        result
    }

    pub fn sum(&self) -> f32 {
        let mut total: f32 = 0.0;

        for row in 0..self.rows {
            for col in 0..self.cols {
                total += self.get(row, col);
            }
        }
        total
    }

    pub fn rowsum(&self, row: usize) -> f32 {
        let mut total: f32 = 0.0;
        for col in 0..self.cols {
            total += self.get(row, col);
        }
        total
    }

    pub fn colsum(&self, col: usize) -> f32 {
        let mut total: f32 = 0.0;
        for row in 0..self.rows {
            total += self.get(row, col);
        }
        total
    }

    pub fn mean(&self) -> f32 {
        let mut total: f32 = 0.0;

        for row in 0..self.rows {
            for col in 0..self.cols {
                total += self.get(row, col);
            }
        }
        total / (self.data.len()) as f32
    }

    pub fn rowmean(&self, row: usize) -> f32 {
        let mut total: f32 = 0.0;

        for col in 0..self.cols {
            total += self.get(row, col);
        }
        total / (self.cols) as f32
    }

    pub fn colmean(&self, col: usize) -> f32 {
        let mut total: f32 = 0.0;

        for row in 0..self.rows {
            total += self.get(row, col);
        }
        total / (self.rows) as f32
    }

    pub fn mat_max(&self) -> f32 {
        let mut max = -INFINITY;

        for val in &self.data {
            if val > &max {
                max = *val;
            }
        }
        max
    }

    pub fn map<F: Fn(f32) -> f32>(&self, f: F) -> Matrix {
        let mut vec = Vec::with_capacity(self.data.len());

        for val in &self.data {
            vec.push(f(*val));
        }

        Matrix::from_vec(self.rows, self.cols, vec)
    }

    pub fn zip_map<F: Fn(f32, f32) -> f32>(&self, other: &Matrix, f: F) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let mut vec = Vec::with_capacity(self.data.len());

        for (a, b) in zip(&self.data, &other.data) {
            vec.push(f(*a, *b));
        }
        Matrix::from_vec(self.rows, self.cols, vec)
    }

    pub fn elementwise_add(&self, other: &Matrix) -> Matrix {
        self.zip_map(other, add_scalars)
    }

    pub fn elementwise_square(&self) -> Matrix {
        self.map(square)
    }

    pub fn elementwise_cube(&self) -> Matrix {
        self.map(cube)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let result = a.add(&b);
        assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_sub() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 4.0, 3.0, 4.0]);
        let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let result = b.sub(&a);
        assert_eq!(result.data, vec![4.0, 2.0, 4.0, 4.0]);
    }

    #[test]
    fn test_scale() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let result = a.scale(2.0);
        assert_eq!(result.data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_scale_inplace() {
        let mut a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        a.scale_inplace(2.0);
        assert_eq!(a.data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_matmul() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let result = a.matmul(&b);
        assert_eq!(result.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_transpose() {
        let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = a.transpose();
        assert_eq!(result.rows, 3);
        assert_eq!(result.cols, 2);
        assert_eq!(result.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_sum() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0]);
        let result = a.sum();
        assert_eq!(result, 4.0);
    }

    #[test]
    fn test_rowsum() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0]);
        let result = a.rowsum(0);
        assert_eq!(result, 2.0);
    }

    #[test]
    fn test_colsum() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0]);
        let result = a.colsum(0);
        assert_eq!(result, 2.0);
    }

    #[test]
    fn test_rowmean() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0]);
        let result = a.rowmean(0);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_colmean() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0]);
        let result = a.colmean(0);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_map() {
        let a = Matrix::from_vec(2, 2, vec![2.0, 2.0, 2.0, 2.0]);
        let result = a.map(cube);
        assert_eq!(result.data, vec![8.0, 8.0, 8.0, 8.0]);
    }

    #[test]
    fn test_zipmap() {
        let a = Matrix::from_vec(2, 2, vec![2.0, 2.0, 2.0, 2.0]);
        let b = Matrix::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0]);
        let result = a.zip_map(&b, add_scalars);
        assert_eq!(result.data, vec![3.0, 3.0, 3.0, 3.0]);
    }
}

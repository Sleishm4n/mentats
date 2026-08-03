#[derive(Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>) -> Tensor {
        let size: usize = shape.iter().product();
        let strides = Tensor::calc_strides(shape.clone());
        Tensor {
            shape,
            strides,
            data: vec![0.0; size],
        }
    }

    fn calc_strides(shape: Vec<usize>) -> Vec<usize> {
        let mut strides = vec![0; shape.len()];
        let mut running_prod = 1;

        for i in (0..shape.len()).rev() {
            strides[i] = running_prod;
            running_prod *= shape[i];
        }
        strides
    }

    fn flat(&self, index: &[usize]) -> usize {
        assert_eq!(
            index.len(),
            self.shape.len(),
            "index rank must match tensor rank"
        );

        let mut result: usize = 0;

        for (i, &idx) in index.iter().enumerate() {
            assert!(idx < self.shape[i], "index out of bounds for axis {i}");
            result += idx * self.strides[i];
        }

        result
    }

    pub fn get(&self, index: &[usize]) -> f32 {
        let flat_index = self.flat(index);
        self.data[flat_index]
    }

    pub fn set(&mut self, index: &[usize], val: f32) {
        let flat_index = self.flat(index);
        self.data[flat_index] = val;
    }

    pub fn from_vec(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
        assert_eq!(
            data.len(),
            shape.iter().product(),
            "data and shape must be same dimensions"
        );
        let strides = Tensor::calc_strides(shape.clone());
        Tensor {
            shape,
            strides,
            data,
        }
    }

    pub fn display(&self) {
        println!(
            "Tensor(shape={:?}, strides={:?}, data={:?})",
            self.shape, self.strides, self.data
        );
    }
}

#[allow(dead_code)]
fn main() {
    println!("tensor crate main");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_zeros_correct_length() {
        let tensor = Tensor::new(vec![3, 2]);
        assert_eq!(tensor.shape, vec![3, 2])
    }

    #[test]
    fn test_get_returns_expected_value_multi_axis() {
        let tensor = Tensor::from_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(tensor.get(&[1, 1]), 4.0)
    }

    #[test]
    fn test_set_mutates_correct_element() {
        let mut tensor = Tensor::from_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(tensor.get(&[1, 1]), 4.0);
        tensor.set(&[1, 1], 7.0);
        assert_eq!(tensor.get(&[1, 1]), 7.0)
    }

    #[test]
    #[should_panic(expected = "data and shape must be same dimensions")]
    fn test_from_vec_panics_on_mismatch() {
        let _tensor = Tensor::from_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_strides_correct_for_2d() {
        let tensor = Tensor::new(vec![3, 2]);
        assert_eq!(tensor.strides, vec![2, 1])
    }

    #[test]
    fn test_strides_correct_for_3d() {
        let tensor = Tensor::new(vec![2, 3, 4]);
        assert_eq!(tensor.strides, vec![12, 4, 1])
    }

    #[test]
    #[should_panic(expected = "index rank must match tensor rank")]
    fn test_get_panics_on_rank_mismatch() {
        let tensor = Tensor::from_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let _ = tensor.get(&[1]);
    }

    #[test]
    #[should_panic(expected = "index out of bounds for axis 1")]
    fn test_get_panics_on_out_of_bounds() {
        let tensor = Tensor::from_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let _ = tensor.get(&[0, 2]);
    }

    #[test]
    fn test_new_data_is_actually_zeroed() {
        let tensor = Tensor::new(vec![3, 2]);
        assert_eq!(tensor.data, vec![0.0; 6]);
    }

    #[test]
    fn test_strides_correct_for_1d() {
        let tensor = Tensor::new(vec![5]);
        assert_eq!(tensor.strides, vec![1]);
    }

    #[test]
    fn test_from_vec_preserves_data_order() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(vec![3, 2], data.clone());
        assert_eq!(tensor.data, data);
    }

    #[test]
    fn test_zero_size_dimension_produces_empty_data() {
        let tensor = Tensor::new(vec![0, 3]);
        assert_eq!(tensor.data, Vec::<f32>::new());
        assert_eq!(tensor.strides, vec![3, 1]);
    }
}

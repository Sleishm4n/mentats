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
        assert_eq!(index.len(), self.shape.len(), "index rank must match tensor rank");

        let mut result: usize = 0;
        for i in 0..index.len() {
            assert!(index[i] < self.shape[i], "index out of bounds for axis {i}");
            result += index[i] * self.strides[i];
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
        assert_eq!(data.len(), shape.iter().product());
        let strides = Tensor::calc_strides(shape.clone());
        Tensor {
            shape, strides, data
        }
    }
    
    pub fn display(&self) {
        println!("Tensor(shape={:?}, strides={:?}, data={:?})", self.shape, self.strides, self.data);
    }
}

#[allow(dead_code)]
fn main() {
    println!("tensor crate main");
}
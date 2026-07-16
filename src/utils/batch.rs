use crate::tensor::Tensor;

pub struct BatchIterator {
    indices: Vec<usize>,
    batch_size: usize,
    current_batch: usize,
}

impl BatchIterator {
    pub fn new(dataset_size: usize, batch_size: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..dataset_size).collect();

        if shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }

        Self {
            indices,
            batch_size,
            current_batch: 0,
        }
    }

    pub fn next_batch(&mut self) -> Option<Vec<usize>> {
        let start = self.current_batch * self.batch_size;
        if start >= self.indices.len() {
            return None;
        }

        let end = (start + self.batch_size).min(self.indices.len());
        self.current_batch += 1;
        Some(self.indices[start..end].to_vec())
    }

    pub fn num_batches(&self) -> usize {
        (self.indices.len() + self.batch_size - 1) / self.batch_size
    }

    pub fn reset(&mut self, shuffle: bool) {
        self.current_batch = 0;
        if shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }
}

pub fn stack_tensors(tensors: &[Tensor]) -> Tensor {
    assert!(!tensors.is_empty(), "cannot stack empty tensor list");

    let first_shape = &tensors[0].shape;
    for t in tensors {
        assert_eq!(
            &t.shape, first_shape,
            "all tensors must have the same shape to stack"
        )
    }

    let batch_size = tensors.len();
    let mut stacked_shape = vec![batch_size];
    stacked_shape.extend(first_shape);

    let mut data = Vec::new();
    for tensor in tensors {
        data.extend(&tensor.data);
    }

    Tensor::from_vec(stacked_shape, data)
}

pub fn slice_batch(tensor: &Tensor, indices: &[usize]) -> Vec<Tensor> {
    assert!(tensor.shape.len() >= 1, "tensor must be at least 1D");

    let batch_dim_size = tensor.shape[0];
    let mut result = Vec::new();

    for &idx in indices {
        assert!(idx < batch_dim_size, "index out of bounds");

        let sample_shape: Vec<usize> = tensor.shape[1..].to_vec();
        let sample_size: usize = sample_shape.iter().product();

        let start = idx * sample_size;
        let end = start + sample_size;

        result.push(Tensor::from_vec(
            sample_shape,
            tensor.data[start..end].to_vec(),
        ));
    }

    result
}

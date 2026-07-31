use crate::tensor::Tensor;
use rand::prelude::*;

impl Tensor {
    pub fn rand_range(shape: Vec<usize>, min: f32, max: f32) -> Tensor {
        assert!(min < max);

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

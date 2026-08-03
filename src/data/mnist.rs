use std::fs::File;
use std::io::Read;

use crate::tensor::Tensor;

pub fn load_images(path: &str) -> Vec<Tensor> {
    let mut file = File::open(path).expect("Could not open file");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Could not read file");

    let num_images = u32::from_be_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]) as usize;
    let num_rows = u32::from_be_bytes([buffer[8], buffer[9], buffer[10], buffer[11]]) as usize;
    let num_cols = u32::from_be_bytes([buffer[12], buffer[13], buffer[14], buffer[15]]) as usize;
    let image_size = num_rows * num_cols;

    let mut images = Vec::new();
    for i in 0..num_images {
        let offset = 16 + i * image_size;
        let pixels: Vec<f32> = buffer[offset..offset + image_size]
            .iter()
            .map(|&p| p as f32 / 255.0)
            .collect();
        images.push(Tensor::from_vec(vec![image_size, 1], pixels));
    }
    images
}

pub fn load_labels(path: &str) -> Vec<u8> {
    let mut file = File::open(path).expect("Could not open file");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Could not read file");

    let num_labels = u32::from_be_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]) as usize;

    buffer[8..8 + num_labels].to_vec()
}

pub fn one_hot(label: u8) -> Tensor {
    let mut ten = Tensor::new(vec![10, 1]);
    ten.set(&[label as usize, 0], 1.0);
    ten
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_loader() {
        let images = load_images("data/mnist/train-images.idx3-ubyte");
        assert_eq!(images.len(), 60000);
        assert_eq!(images[0].shape[0], 784);
        assert_eq!(images[0].shape[1], 1);
        assert!(images[0].data.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }

    #[test]
    fn test_labels() {
        let labels = load_labels("data/mnist/train-labels.idx1-ubyte");
        assert_eq!(labels.len(), 60000);
        assert!(labels.iter().all(|&l| l <= 9));
    }
}

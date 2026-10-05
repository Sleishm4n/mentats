use std::fs::File;
use std::io::Read;

use crate::tensor::Tensor;

/// Loads SVHN images from a float32 binary file (N, 32, 32, 3)
/// and transposes each sample into planar NCHW format ([3, 32, 32])
pub fn load_images(path: &str) -> Vec<Tensor> {
    let mut file = File::open(path).expect("Could not open SVHN images file");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .expect("Could not read SVHN images file");

    let h = 32;
    let w = 32;
    let c = 3;
    let floats_per_image = h * w * c;
    let bytes_per_image = floats_per_image * 4;
    let num_images = buffer.len() / bytes_per_image;

    let mut images = Vec::with_capacity(num_images);

    for img_bytes in buffer.chunks_exact(bytes_per_image) {
        let hwc_data: Vec<f32> = img_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        // Transpose [32, 32, 3] (H, W, C) -> [3, 32, 32] (C, H, W)
        let mut chw_data = vec![0.0f32; floats_per_image];
        for i in 0..h {
            for j in 0..w {
                for ch in 0..c {
                    let hwc_idx = i * (w * c) + j * c + ch;
                    let chw_idx = ch * (h * w) + i * w + j;
                    chw_data[chw_idx] = hwc_data[hwc_idx];
                }
            }
        }

        images.push(Tensor::from_vec(vec![3, 32, 32], chw_data));
    }

    images
}

/// Loads SVHN labels from an int64 binary file (8 bytes per sample)
pub fn load_labels(path: &str) -> Vec<u8> {
    let mut file = File::open(path).expect("Could not open SVHN labels file");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .expect("Could not read SVHN labels file");

    // Each label is an int64 (8 bytes, little-endian)
    buffer
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as u8)
        .collect()
}

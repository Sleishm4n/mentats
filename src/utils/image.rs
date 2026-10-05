use crate::tensor::Tensor;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub fn to_u8_gray(v: f32, apply_sigmoid: bool) -> u8 {
    let p = if apply_sigmoid {
        1.0 / (1.0 + (-v).exp())
    } else {
        v
    }
    .clamp(0.0, 1.0);

    (p * 255.0).round() as u8
}

pub fn save_mnist_tensor_pgm(
    tensor: &Tensor,
    path: &Path,
    apply_sigmoid: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if tensor.data.len() != 28 * 28 {
        return Err(format!(
            "Expected 784 values for MNIST images, got {}",
            tensor.data.len()
        )
        .into());
    }

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "P2")?;
    writeln!(writer, "28 28")?;
    writeln!(writer, "255")?;

    for (i, v) in tensor.data.iter().enumerate() {
        let px = to_u8_gray(*v, apply_sigmoid);
        if i % 28 == 27 {
            writeln!(writer, "{}", px)?;
        } else {
            write!(writer, "{} ", px)?;
        }
    }

    Ok(())
}

pub fn save_pgm_grid_4x4(
    samples: &[Tensor],
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(samples.len(), 16);
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    // 4 digits wide x 28 = 112, 4 digits high x 28 = 112
    writeln!(writer, "P2")?;
    writeln!(writer, "112 112")?;
    writeln!(writer, "255")?;
    for grid_row in 0..4 {
        for pixel_row in 0..28 {
            for grid_col in 0..4 {
                let sample_idx = grid_row * 4 + grid_col;
                for pixel_col in 0..28 {
                    let val = samples[sample_idx].get(&[pixel_row * 28 + pixel_col, 0]);
                    let px = to_u8_gray(val, true);
                    write!(writer, "{} ", px)?;
                }
            }
            writeln!(writer)?;
        }
    }
    Ok(())
}

pub fn save_ppm(tensor: &Tensor, path: &Path) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    // P6 = binary RGB, 32 32 = width height, 255 = max color
    writeln!(file, "P6\n32 32\n255")?;
    let mut bytes = Vec::with_capacity(32 * 32 * 3);
    for row in 0..32 {
        for col in 0..32 {
            for ch in 0..3 {
                let val = tensor.get(&[ch, row, col]).clamp(0.0, 1.0);
                bytes.push((val * 255.0) as u8);
            }
        }
    }
    file.write_all(&bytes)?;
    Ok(())
}

pub fn save_ppm_grid_4x4(samples: &[Tensor], path: &Path) -> std::io::Result<()> {
    assert_eq!(samples.len(), 16);
    let mut file = File::create(path)?;
    // 4 digits wide x 32 = 128, 4 digits high x 32 = 128
    writeln!(file, "P6\n128 128\n255")?;
    let mut bytes = Vec::with_capacity(128 * 128 * 3);
    for grid_row in 0..4 {
        for pixel_row in 0..32 {
            for grid_col in 0..4 {
                let sample_idx = grid_row * 4 + grid_col;
                for pixel_col in 0..32 {
                    for ch in 0..3 {
                        let v = samples[sample_idx].get(&[ch, pixel_row, pixel_col]);
                        let p = (1.0 / (1.0 + (-v).exp())).clamp(0.0, 1.0);
                        bytes.push((p * 255.0).round() as u8);
                    }
                }
            }
        }
    }
    file.write_all(&bytes)?;
    Ok(())
}

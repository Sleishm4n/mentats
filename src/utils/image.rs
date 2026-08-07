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

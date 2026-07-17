use std::io::{self, Read, Write};

use crate::nn::{
    activation::ActivationLayer, flatten::FlattenLayer, linear::LinearLayer,
    reshape::ReshapeLayer, sampling::GaussianSampler, softmax::SoftmaxLayer, Layer,
};
use crate::tensor::Tensor;

// ---- layer type tags -------------------------------------------------
// Written by each Layer::save impl, read by `load_layer` below to know
// which concrete type's constructor to call. Keep in sync as new layers
// are added (ConvLayer / MaxPoolLayer will need tags 5, 6, ...).
pub const TAG_LINEAR: u8 = 0;
pub const TAG_ACTIVATION: u8 = 1;
pub const TAG_SOFTMAX: u8 = 2;
pub const TAG_FLATTEN: u8 = 3;
pub const TAG_RESHAPE: u8 = 4;
pub const TAG_SAMPLER: u8 = 5;

// ---- primitive helpers -------------------------------------------------

pub fn write_u8(writer: &mut dyn Write, val: u8) -> io::Result<()> {
    writer.write_all(&[val])
}

pub fn read_u8(reader: &mut dyn Read) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

pub fn write_u32(writer: &mut dyn Write, val: u32) -> io::Result<()> {
    writer.write_all(&val.to_le_bytes())
}

pub fn read_u32(reader: &mut dyn Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn write_f32(writer: &mut dyn Write, val: f32) -> io::Result<()> {
    writer.write_all(&val.to_le_bytes())
}

fn read_f32(reader: &mut dyn Read) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

pub fn write_shape(writer: &mut dyn Write, shape: &[usize]) -> io::Result<()> {
    write_u32(writer, shape.len() as u32)?;
    for &dim in shape {
        write_u32(writer, dim as u32)?;
    }
    Ok(())
}

pub fn read_shape(reader: &mut dyn Read) -> io::Result<Vec<usize>> {
    let ndim = read_u32(reader)? as usize;
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        shape.push(read_u32(reader)? as usize);
    }
    Ok(shape)
}

// ---- tensor format -------------------------------------------------
// [ndim: u32][shape: ndim x u32][data: (product of shape) x f32]
//
// Only `tensor.data` is written, in the order it's currently stored.
// This is only correct for a *contiguous* tensor (strides matching a
// fresh row-major layout for that shape) - e.g. weights/biases owned
// directly by a layer, never a transposed/permuted view. If you ever
// save a tensor that came out of `.transpose()`/`.permute()` without
// copying it first, this will silently write the data in the wrong
// order. Worth asserting `tensor.strides == calc_strides(tensor.shape)`
// here if that ever becomes a risk.
pub fn write_tensor(writer: &mut dyn Write, tensor: &Tensor) -> io::Result<()> {
    write_shape(writer, &tensor.shape)?;
    for &val in &tensor.data {
        write_f32(writer, val)?;
    }
    Ok(())
}

pub fn read_tensor(reader: &mut dyn Read) -> io::Result<Tensor> {
    let shape = read_shape(reader)?;
    let len: usize = shape.iter().product();
    let mut data = Vec::with_capacity(len);
    for _ in 0..len {
        data.push(read_f32(reader)?);
    }
    Ok(Tensor::from_vec(shape, data))
}

// ---- layer dispatch -------------------------------------------------
// Can't live on the Layer trait: reconstructing a layer means producing
// a concrete Self, which isn't callable through a `dyn Layer` you don't
// have yet. So this is a free function that reads the tag itself wrote
// and calls the matching concrete type's own (non-trait) `load`.
pub fn load_layer(reader: &mut dyn Read) -> io::Result<Box<dyn Layer>> {
    let tag = read_u8(reader)?;
    match tag {
        TAG_LINEAR => Ok(Box::new(LinearLayer::load(reader)?)),
        TAG_ACTIVATION => Ok(Box::new(ActivationLayer::load(reader)?)),
        TAG_SOFTMAX => Ok(Box::new(SoftmaxLayer::load(reader)?)),
        TAG_FLATTEN => Ok(Box::new(FlattenLayer::load(reader)?)),
        TAG_RESHAPE => Ok(Box::new(ReshapeLayer::load(reader)?)),
        TAG_SAMPLER => Ok(Box::new(GaussianSampler::load(reader)?)),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown layer tag: {other}"),
        )),
    }
}
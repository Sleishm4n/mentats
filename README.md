# rust-ml 

A neural network built from scratch in Rust

The goal is to understand how they are implemented and how they fundamentally work
This involves understanding every operation from the maths to the memory layout

At the current stage, the network solves MNIST at 97% accurarcy 

---

## Results

**Initial loss: 2.4407463**
 
| Epoch | Loss   | Train Accuracy | Elapsed Time (s) |
|-------|--------|----------------|------------------|
| 0     | 0.2634 | 92.48%         | 41.86693635      |
| 1     | 0.1167 | 96.55%         | 41.962896705     |
| 2     | 0.0797 | 97.66%         | 41.903314238     |
| 3     | 0.0594 | 98.16%         | 41.835316543     |
| 4     | 0.0466 | 98.57%         | 41.93213144      |
 
**Test accuracy: 97.43%**

---

## Network Architecture
 
```
Input (784)
  └─ LinearLayer (784 → 128)
  └─ ReLU
  └─ LinearLayer (128 → 10)
  └─ Softmax (implicit via cross-entropy loss)
```
Trained with SGD, learning rate `0.01`, categorical cross-entropy loss, 5 epochs over the full 60,000 training examples.

---

## Project Structure
 
```
data/
├── mnist/
│   ├── train-images.idx3-ubyte
│   ├── ...
src/
├── data/
│   ├── mnist.rs # Load images and labels
│   ├── mod.rs
├── loss /
│   ├── cross_entropy.rs
│   ├── mse.rs # mean squared error
│   ├── mod.rs
├── tensor/
│   ├── matrix.rs   # Matrix struct and core operations
│   ├── ops.rs
│   ├── init.rs     # Weight initialisation
│   └── mod.rs
├── nn/
│   ├── activation.rs  # ReLU, sigmoid, tanh
│   ├── linear.rs      # LinearLayer (forward pass, backward pass, sgd_update)
│   ├── network.rs # Collection of network layers
│   ├── softmax.rs # Softmax function
│   └── mod.rs
├── utils /
│   ├── lib.rs
examples/
│   ├── xor.rs
│   ├── mnist.rs
│   ├── basics.rs
└──────
```
---
 
## Getting the Data
 
The MNIST binary files are not included in this repo. Download them from the [Kaggle MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset), and place them in `data/mnist/`:
 
```
data/mnist/train-images.idx3-ubyte
data/mnist/train-labels.idx1-ubyte
data/mnist/t10k-images.idx3-ubyte
data/mnist/t10k-labels.idx1-ubyte
```
 
---

## What's Implemented
 
**Matrix (`src/tensor/matrix.rs`)**
- Flat `Vec<f32>` row-major storage
- `new`, `get`, `set`, `from_vec`, `display`
- `scale`, `scale_inplace`, `add`, `sub`
- `matmul` — cache-friendly i-k-j loop order
- `transpose`
 
**Activation functions (`src/nn/activation.rs`)**
- ReLU, sigmoid, tanh
- Derivatives of all the above
 
**LinearLayer (`src/nn/linear.rs`)**
- Weight shape: `(out_features × in_features)`
- Bias stored as `Matrix`
- `forward(&input)` computes `W·x + b`

**Network (`src/nn/network.rs`)**
- Stores a vector of layers (`Vec<Box<dyn Layer>>`)
- Forward, Backward and Update


 
## Dependencies
 
```toml
[dependencies]
rand = "0.8"
```

`rand` is the only external crate, used for weight initialisation. 
Everything else is standard library.

## Running
 
```bash
cargo run --example basics
cargo run --example xor
cargo run --example mnist
cargo test
```

## Design Notes
 
- Weights are `(out × in)` — consistent with the convention that `forward` computes `W·x`, where `x` is a column vector.
- `matmul` uses i-k-j loop order intentionally for cache performance; don't reorder.
- Tests use `Matrix::from_vec` with known values and epsilon comparison — no random inputs in correctness tests.

use std::vec;

use rust_ml::nn::activation::{relu, sigmoid, tanh};
use rust_ml::nn::linear::LinearLayer;
use rust_ml::tensor::Tensor;

fn main() {
    let w = Tensor::rand_range(vec![3, 3], -10.0, 10.0);
    let mut layer = LinearLayer::new_rand(3, 1);

    println!("Original Matrix:");
    w.display();

    println!("Layer:");
    layer.weight.display();
    println!("Bias");
    layer.bias.display();

    println!("After ReLU:");
    let relu_res = w.map(relu);
    relu_res.display();

    println!("After Sigmoid:");
    let sig_res = w.map(sigmoid);
    sig_res.display();

    println!("After Tanh:");
    let tanh_res = w.map(tanh);
    tanh_res.display();

    println!("Input");
    let input = Tensor::rand_range(vec![3, 1], -1.0, 1.0);
    input.display();

    println!("After Forward:");
    let forward_res = layer.forward(&input);
    forward_res.display();
}

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mentats::tensor::Tensor;

fn _bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_multiplication");
    
    let a = Tensor::from_vec(vec![256, 256],vec![0.5; 256 * 256]);
    let b = Tensor::from_vec(vec![256, 256], vec![0.5; 256 * 256]);

    group.bench_function("matmul_256x256", |bencher| {
        bencher.iter(|| {
            black_box(a.matmul(black_box(&b)))
        });
    });
    group.finish();
}

use mentats::{nn::conv::Conv2DLayer};

fn bench_conv(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d");

    // Realistic SVHN feature map: batch 64, 3 input channels, 32x32 image
    // Conv: 3 -> 16 channels, 3x3 kernel
    let mut conv = Conv2DLayer::new(3, 16, (3, 3));
    let input = Tensor::new(vec![64, 3, 32, 32]);
    let d_output = Tensor::new(vec![64, 16, 30, 30]);

    group.bench_function("conv_forward_batch64", |bencher| {
        bencher.iter(|| {
            black_box(conv.forward(black_box(&input)))
        });
    });

    group.bench_function("conv_backward_batch64", |bencher| {
        // Run one forward so internal cached input exists
        conv.forward(&input);
        bencher.iter(|| {
            black_box(conv.backward(black_box(&d_output)))
        });
    });

    group.finish();
}


criterion_group!(benches, bench_conv);
criterion_main!(benches);
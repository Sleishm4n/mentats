use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mentats::nn::conv::Conv2DLayer;
use mentats::tensor::Tensor;

fn naive_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    assert!(a.shape.len() == 2 && a.shape[1] == b.shape[0]);
    let mut c = Tensor::new(vec![a.shape[0], b.shape[1]]);
    for i in 0..a.shape[0] {
        for k in 0..a.shape[1] {
            let val_a = a.get(&[i, k]);
            for j in 0..b.shape[1] {
                let prev = c.get(&[i, j]);
                c.set(&[i, j], prev + val_a * b.get(&[k, j]));
            }
        }
    }
    c
}

fn bench_matmul_comp(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_comparison");

    let a = Tensor::from_vec(vec![256, 256], vec![0.5; 256 * 256]);
    let b = Tensor::from_vec(vec![256, 256], vec![0.5; 256 * 256]);

    // 1. Optimised version (i-k-j contiguous row auto-vectorized)
    group.bench_function("256x256_optimised", |bencher| {
        bencher.iter(|| black_box(a.matmul(black_box(&b))));
    });

    // 2. Naive version (with get/set in inner loop)
    group.bench_function("256x256_naive", |bencher| {
        bencher.iter(|| black_box(naive_matmul(black_box(&a), black_box(&b))));
    });

    group.finish();
}

fn bench_matmul_var(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_varing_sizes");

    for size in [64, 128, 512] {
        let a = Tensor::from_vec(vec![size, size], vec![0.5; size * size]);
        let b = Tensor::from_vec(vec![size, size], vec![0.5; size * size]);
        group.bench_function(format!("{size}x{size}_optimised"), |bencher| {
            bencher.iter(|| black_box(a.matmul(black_box(&b))));
        });
    }

    group.finish();
}

fn bench_conv(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d");

    // Realistic SVHN feature map: batch 64, 3 input channels, 32x32 image
    // Conv: 3 -> 16 channels, 3x3 kernel
    let mut conv = Conv2DLayer::new(3, 16, (3, 3));
    let input = Tensor::new(vec![64, 3, 32, 32]);
    let d_output = Tensor::new(vec![64, 16, 30, 30]);

    group.bench_function("conv_forward_batch64", |bencher| {
        bencher.iter(|| black_box(conv.forward(black_box(&input))));
    });

    group.bench_function("conv_backward_batch64", |bencher| {
        // Run one forward so internal cached input exists
        conv.forward(&input);
        bencher.iter(|| black_box(conv.backward(black_box(&d_output))));
    });

    group.finish();
}

fn bench_conv_comp(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_comparison");

    let mut conv = Conv2DLayer::new(1, 16, (3, 3));
    let single_input = Tensor::new(vec![1, 28, 28]); // MNIST single image shape

    group.bench_function("naive_generic_single", |b| {
        b.iter(|| black_box(conv._forward_generic(black_box(&single_input))));
    });

    group.bench_function("im2col_gemm_single", |b| {
        b.iter(|| black_box(conv.forward(black_box(&single_input))));
    });

    group.finish();
}

criterion_group!(benches, bench_matmul_comp, bench_matmul_var, bench_conv, bench_conv_comp);
criterion_main!(benches);

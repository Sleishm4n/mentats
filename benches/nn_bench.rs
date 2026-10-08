use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
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
    let mut group = c.benchmark_group("matmul_varying_sizes");

    for size in [64, 128, 512] {
        let flops = 2 * (size as u64).pow(3);
        group.throughput(Throughput::Elements(flops));

        if size == 512 {
            group.sample_size(10);
        } else {
            group.sample_size(100);
        }

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

fn bench_matmul_batched_var(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_batched_sizes");

    // Realistic deep learning batch/matrix configurations:
    // 1. Small / edge of threshold: batch 16, 64x64 (~4.2M ops)
    // 2. Medium training batch: batch 64, 64x64 (~16.8M ops)
    // 3. Large MLP feature size: batch 64, 128x128 (~134M ops)
    for (batch, size) in [(16, 64), (64, 64), (64, 128)] {
        let a = Tensor::from_vec(vec![batch, size, size], vec![0.5; batch * size * size]);
        let b = Tensor::from_vec(vec![size, size], vec![0.5; size * size]);

        group.bench_function(format!("b{batch}_{size}x{size}_batched"), |bencher| {
            bencher.iter(|| black_box(a.matmul_batched(black_box(&b))));
        });

        group.bench_function(format!("b{batch}_{size}x{size}_broadcast"), |bencher| {
            bencher.iter(|| black_box(b.matmul_batched_broadcast(black_box(&a))));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_matmul_comp,
    bench_matmul_var,
    bench_conv,
    bench_conv_comp,
    bench_matmul_batched_var
);
criterion_main!(benches);

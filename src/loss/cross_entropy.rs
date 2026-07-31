use crate::tensor::Tensor;

pub fn cross_entropy(output: &Tensor, target: &Tensor) -> f32 {
    let max = output.tensor_max();
    let exps = output.map(|x| (x - max).exp());
    let sum: f32 = exps.data.iter().sum();

    let mut loss = 0.0;

    for (i, e) in exps.data.iter().enumerate() {
        let prob = e / sum;
        loss += target.data[i] * (prob + 1e-7).ln();
    }

    -loss
}

pub fn d_cross_entropy(output: &Tensor, target: &Tensor) -> Tensor {
    let max = output.tensor_max();
    let exps = output.map(|x| (x - max).exp());
    let sum: f32 = exps.data.iter().sum();

    let probs = exps.map(|x| x / sum);

    probs.zip_map(target, |p, y| p - y)
}

pub fn binary_cross_entropy(logits: &Tensor, target: &Tensor) -> f32 {
    let n = logits.data.len() as f32;
    let mut loss: f32 = 0.0;

    for (logit, t) in logits.data.iter().zip(target.data.iter()) {
        let p = (1.0 / (1.0 + (-logit).exp())).clamp(1e-7, 1.0 - 1e-7);
        loss += -(t * p.ln() + (1.0 - t) * (1.0 - p).ln());
    }
    loss / n
}

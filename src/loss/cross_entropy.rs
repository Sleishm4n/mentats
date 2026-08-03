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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy_correct_value() {
        let output = Tensor::from_vec(vec![3, 1], vec![0.0, 0.0, 0.0]);
        let target = Tensor::from_vec(vec![3, 1], vec![1.0, 0.0, 0.0]);

        let loss = cross_entropy(&output, &target);

        // -ln(1/3) approx 1.098612
        assert!((loss - 1.098612).abs() < 1e-5)
    }

    #[test]
    fn test_cross_entropy_stable_for_extreme_logits() {
        let output = Tensor::from_vec(vec![3, 1], vec![1000.0, 1000.0, 1000.0]);
        let target = Tensor::from_vec(vec![3, 1], vec![1.0, 0.0, 0.0]);

        let loss = cross_entropy(&output, &target);

        assert!(loss.is_finite(), "loss was {loss}, expected a finite value");
        assert!((loss - 1.098612).abs() < 1e-4); 
    }

    #[test]
    fn test_cross_entropy_extreme_unequal_logits() {
        let output = Tensor::from_vec(vec![3, 1], vec![1000.0, 0.0, -1000.0]);
        let target = Tensor::from_vec(vec![3, 1], vec![1.0, 0.0, 0.0]);

        let loss = cross_entropy(&output, &target);

        assert!(loss.is_finite());
        assert!(loss.abs() < 1e-4);
    }

    #[test]
    fn test_d_cross_entropy_matches_closed_form() {
        let output = Tensor::from_vec(vec![3, 1], vec![1.0, 2.0, 3.0]);
        let target = Tensor::from_vec(vec![3, 1], vec![0.0, 1.0, 0.0]);

        let grad = d_cross_entropy(&output, &target);

        let max = 3.0_f32;
        let exps: Vec<f32> = vec![1.0f32, 2.0, 3.0]
            .iter()
            .map(|x| (x - max).exp())
            .collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        for i in 0..3 {
            let expected = probs[i] - target.data[i];
            assert!((grad.data[i] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_d_cross_entropy_matches_numerical_gradient() {
        let output = Tensor::from_vec(vec![3, 1], vec![0.5, -1.2, 2.3]);
        let target = Tensor::from_vec(vec![3, 1], vec![0.0, 0.0, 1.0]);

        let analytical = d_cross_entropy(&output, &target);

        let epsilon = 1e-4;
        for i in 0..3 {
            let mut plus = output.clone();
            plus.data[i] += epsilon;
            let mut minus = output.clone();
            minus.data[i] -= epsilon;

            let loss_plus = cross_entropy(&plus, &target);
            let loss_minus = cross_entropy(&minus, &target);

            let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);

            assert!(
                (analytical.data[i] - numerical).abs() < 1e-3,
                "index {i}: analytical {} vs numerical {}",
                analytical.data[i],
                numerical
            );
        }
    }

    #[test]
    fn test_binary_cross_entropy_correct_value() {
        let logits = Tensor::from_vec(vec![1], vec![0.0]);
        let target = Tensor::from_vec(vec![1], vec![1.0]);

        let loss = binary_cross_entropy(&logits, &target);

        assert!((loss - 0.693147).abs() < 1e-5);
    }

    #[test]
    fn test_binary_cross_entropy_averages_over_batch() {
        let logits = Tensor::from_vec(vec![2], vec![0.0, 0.0]);
        let target = Tensor::from_vec(vec![2], vec![1.0, 1.0]);

        let loss = binary_cross_entropy(&logits, &target);

        assert!((loss - 0.693147).abs() < 1e-5);
    }

    #[test]
    fn test_binary_cross_entropy_stable_for_extreme_logits() {
        let logits = Tensor::from_vec(vec![2], vec![1000.0, -1000.0]);
        let target = Tensor::from_vec(vec![2], vec![1.0, 0.0]);

        let loss = binary_cross_entropy(&logits, &target);

        assert!(loss.is_finite());
        assert!(loss.abs() < 1e-3);
    }

    #[test]
    fn test_binary_cross_entropy_stable_for_confidently_wrong_logits() {
        let logits = Tensor::from_vec(vec![1], vec![1000.0]);
        let target = Tensor::from_vec(vec![1], vec![0.0]);

        let loss = binary_cross_entropy(&logits, &target);

        assert!(loss.is_finite());
        assert!(loss > 10.0); 
    }
}

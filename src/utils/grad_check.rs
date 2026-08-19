use crate::loss::mse::{d_mse, mse};
use crate::nn::Layer;
use crate::tensor::Tensor;

pub fn gradient_check(layer: &mut dyn Layer, input: &Tensor, target: &Tensor, eps: f32) {
    let forward_res = layer.forward_pass(input);
    let d_output = d_mse(&forward_res, target);
    let _ = layer.backward_pass(&d_output);
    let grads = layer.get_grads();
    let params = layer.get_params();

    assert_eq!(
        grads.len(),
        params.len(),
        "Layer gradient count ({}) does not match parameter tensor count ({})",
        grads.len(),
        params.len()
    );

    for (param_idx, param) in params.iter().enumerate() {
        for i in 0..param.data.len() {
            let mut params_plus = params.clone();
            params_plus[param_idx].data[i] += eps;
            layer.set_params(params_plus);
            let loss_plus = mse(&layer.forward_pass(input), target);

            let mut params_minus = params.clone();
            params_minus[param_idx].data[i] -= eps;
            layer.set_params(params_minus);
            let loss_minus = mse(&layer.forward_pass(input), target);

            layer.set_params(params.clone());

            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            let analytical = grads[param_idx].data[i];
            let relative_error = if numerical.abs() + analytical.abs() < 1e-10 {
                0.0
            } else {
                (numerical - analytical).abs() / (numerical.abs() + analytical.abs())
            };
            assert!(
                relative_error < 1e-4,
                "param[{param_idx}][{i}]: numerical={numerical:.6}, analytical={analytical:.6}, err={relative_error:.2e}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::flatten::FlattenLayer;
    use crate::nn::linear::LinearLayer;
    use crate::nn::Layer;

    #[test]
    fn test_gradient_check_linear() {
        let mut layer = LinearLayer::new(2, 2);
        layer.set_params(vec![
            Tensor::from_vec(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4]),
            Tensor::from_vec(vec![2, 1], vec![0.0, 0.0]),
        ]);
        let input = Tensor::from_vec(vec![2, 1], vec![1.0, 2.0]);
        let target = Tensor::from_vec(vec![2, 1], vec![0.5, 0.5]);
        gradient_check(&mut layer, &input, &target, 1e-3);
    }

    #[test]
    fn test_gradient_check_linear_nonzero_bias() {
        let mut layer = LinearLayer::new(3, 2);
        layer.set_params(vec![
            Tensor::from_vec(vec![2, 3], vec![0.5, -0.3, 0.2, 0.1, 0.4, -0.6]),
            Tensor::from_vec(vec![2, 1], vec![0.2, -0.1]),
        ]);
        let input = Tensor::from_vec(vec![3, 1], vec![1.0, -2.0, 0.5]);
        let target = Tensor::from_vec(vec![2, 1], vec![1.0, 0.0]);
        gradient_check(&mut layer, &input, &target, 1e-3);
    }

    #[test]
    fn test_gradient_check_parameterless_layer_is_noop() {
        // FlattenLayer get_params/get_grads both return Vec::new(), so
        // both loops in gradient_check run zero times.
        let mut layer = FlattenLayer::new();
        let input = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let target = Tensor::from_vec(vec![4, 1], vec![0.0, 0.0, 0.0, 0.0]);
        gradient_check(&mut layer, &input, &target, 1e-4);
    }
}

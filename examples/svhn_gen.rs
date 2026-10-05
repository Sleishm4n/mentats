use mentats::nn::network::Network;
use mentats::nn::sampling::GaussianSampler;
use mentats::utils::image::save_ppm_grid_4x4;
use std::path::Path;

fn main() {
    // needs to be updated in line with the vae latent dims
    let latent_dim = 10;
    let mut decoder =
        Network::load("checkpoints/svhn_cnn_vae_decoder.rmlc").expect("failed to load decoder");
    let mut grid_samples = Vec::new();
    for _ in 0..16 {
        let z = GaussianSampler::sample_standard_normal(latent_dim);
        grid_samples.push(decoder.forward(&z));
    }
    save_ppm_grid_4x4(
        &grid_samples,
        Path::new("outputs/svhn_cnn_vae/final_grid_4x4.ppm"),
    )
    .unwrap();
    println!("Saved 4x4 grid to outputs/svhn_cnn_vae/final_grid_4x4.ppm");
}

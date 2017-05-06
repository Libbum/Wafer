use num::Complex;
use ndarray::Array3;

use potential;
use config::*;

pub fn show_complex() {
    let test = Complex::new(52, 20);
    println!("Complex number: {}", test);
    let testfloat: Complex<f64> = Complex::new(25.3, 1e4);
    println!("Complex float: {}", testfloat);
}

pub fn build_array() {
    // I think we are going to dynamically fill everything...
    //    let test = array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
    //   println!("3D array: {}", test);

    let mut test_fill = Array3::<f64>::zeros((3, 4, 5));
    println!("{}", test_fill);
    println!("{:?}", test_fill);

    test_fill[[2, 2, 2]] += 0.5;
    println!("{}", test_fill);
}

pub fn load_potential_arrays(config: &Config) {
    let origin = Index3 { x: 0, y: 0, z: 0 };
    let minima: f64 = potential::potential(config, &origin);
    let v: Array3<f64> = potential::generate(config);
    let b = 1. / (1. + config.grid.dt * v / 2.);
}

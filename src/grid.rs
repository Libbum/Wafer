//use num::Complex;
use ndarray::Array3;
//use ndarray_parallel::prelude::*;

use potential;
use config::*;

//pub fn show_complex() {
//    let test = Complex::new(52, 20);
//    println!("Complex number: {}", test);
//    let testfloat: Complex<f64> = Complex::new(25.3, 1e4);
//    println!("Complex float: {}", testfloat);
//}
//
//pub fn build_array() {
//    // I think we are going to dynamically fill everything...
//    //    let test = array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
//    //   println!("3D array: {}", test);
//
//    let mut test_fill = Array3::<f64>::zeros((3, 4, 5));
//    println!("{}", test_fill);
//    println!("{:?}", test_fill);
//
//    test_fill[[2, 2, 2]] += 0.5;
//    println!("{}", test_fill);
//}

pub fn load_potential_arrays(config: &Config) {
    let mut minima: f64 = 1e20;

    //TODO: This is a bit messy, figure out how to clean it up.
    let mut v: Array3<_> = match config.potential {
        Potential::FromFile => {
            match potential::from_file() {
                Ok(result) => result,
                Err(err) => panic!("Error: {}", err),
            }},
        Potential::FromScript => {
            match potential::from_script() {
                Ok(result) => result,
                Err(err) => panic!("Error: {}", err),
            }},
        _ => { 
            match potential::generate(config) {
                Ok(result) => result,
                Err(err) => panic!("Error: {}", err),
            }},
    };

    let b = 1. / (1. + config.grid.dt * &v / 2.);
    let a = (1. - config.grid.dt * &v / 2.)*b;
    // We can't do this in a par.
    // AFAIK, this is the safest way to work with the float here.
    for el in v.iter_mut() {
        if el.is_finite() {
            minima = minima.min(*el);
        }
    }
    //Get 2*abs(min(potential)) for offset of beta
    let epsilon = 2.*minima.abs();
}

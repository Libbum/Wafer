use ndarray::Array3;

use potential;
use config::*;

#[derive(Debug)]
pub struct Potentials {
    v: Array3<f64>,
    a: Array3<f64>,
    b: Array3<f64>,
    epsilon: f64,
}


pub fn load_potential_arrays(config: &Config) -> Potentials {
    let mut minima: f64 = 1e20;

    let result = match config.potential {
        PotentialType::FromFile => potential::from_file(),
        PotentialType::FromScript => potential::from_script(),
        _ => potential::generate(config),
    };
    let v: Array3<f64> = match result {
        Ok(r) => r,
        Err(err) => panic!("Error: {}", err),
    };


    let b = 1. / (1. + config.grid.dt * &v / 2.);
    let a = (1. - config.grid.dt * &v / 2.) * &b;

    // We can't do this in a par.
    // AFAIK, this is the safest way to work with the float here.
    for el in v.iter() {
        if el.is_finite() {
            minima = minima.min(*el);
        }
    }
    //Get 2*abs(min(potential)) for offset of beta
    let epsilon = 2. * minima.abs();

    Potentials { v, a, b, epsilon }
}

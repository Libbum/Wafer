use ndarray::{Array3, Zip};
use ndarray_parallel::prelude::*;
use std::error::Error;
use std::fmt;

use config::*;

//TODO: Add failure modes for file read and scripting.
#[derive(Debug)]
pub enum PotentialError {
    NotAvailable,
}

impl fmt::Display for PotentialError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            PotentialError::NotAvailable => {
                write!(f,
                       "Not able to calculate potential value at an index for this potential type.")
            }
        }
    }
}

impl Error for PotentialError {
    fn description(&self) -> &str {
        match *self {
            PotentialError::NotAvailable => "not available",
        }
    }

    fn cause(&self) -> Option<&Error> {
        match *self {
            PotentialError::NotAvailable => None,
        }
    }
}

pub fn generate(config: &Config) -> Result<Array3<f64>, PotentialError> {
    let num = &config.grid.size;
    //NOTE: Don't forget that sizes are non inclusive. We want num.n + 5 to be our last value, so we need num.n + 6 here.
    let init_size: [usize; 3] = [(num.x + 6) as usize,
                                 (num.y + 6) as usize,
                                 (num.z + 6) as usize];
    let mut v = Array3::<f64>::zeros(init_size);

    Zip::indexed(&mut v).par_apply(|(i, j, k), x| match potential(config,
                                                                  &Index3 { x: i, y: j, z: k }) {
                                       Ok(result) => *x = result,
                                       Err(err) => panic!("Error: {}", err),
                                   });
    Ok(v)
}

/// Loads a pre-calculated potential from a user defined file.
pub fn from_file() -> Result<Array3<f64>, PotentialError> {
    //TODO: This is currently just a placeholder.
    let mut v = Array3::<f64>::zeros((2, 2, 2));

    Zip::from(&mut v).par_apply(|x| *x = 7.);
    Ok(v)
}

/// Loads a pre-calculated potential from a user defined script.
pub fn from_script() -> Result<Array3<f64>, PotentialError> {
    //TODO: This is currently just a placeholder.
    let ret = 2.3;
    let mut v = Array3::<f64>::zeros((2, 2, 2));

    Zip::from(&mut v).par_apply(|x| *x = ret);
    Ok(v)
}

//TODO: For now we're dropping complex all together, but this is needed.
fn potential(config: &Config, idx: &Index3) -> Result<f64, PotentialError> {
    let num = &config.grid.size;
    match config.potential {
        PotentialType::NoPotential => Ok(0.0),
        PotentialType::Cube => {
            if (idx.x > num.x / 4 && idx.x <= 3 * num.x / 4) &&
               (idx.y > num.y / 4 && idx.y <= 3 * num.y / 4) &&
               (idx.z > num.z / 4 && idx.z <= 3 * num.z / 4) {
                Ok(-10.0)
            } else {
                Ok(0.0)
            }
        }
        PotentialType::QuadWell => {
            if (idx.x > num.x / 4 && idx.x <= 3 * num.x / 4) &&
               (idx.y > num.y / 4 && idx.y <= 3 * num.y / 4) &&
               (idx.z > 3 * num.z / 8 && idx.z <= 5 * num.z / 8) {
                Ok(-10.0)
            } else {
                Ok(0.0)
            }
        }
        PotentialType::ComplexCoulomb => {
            let r = calculate_r(idx, &config.grid);
            if r < config.grid.dn {
                Ok(-1. / config.grid.dn)
            } else {
                Ok(-1. / r)
            }
        }
        PotentialType::FromFile |
        PotentialType::FromScript => Err(PotentialError::NotAvailable), //TODO: Script may not need to error.
    }
}

fn calculate_r(idx: &Index3, grid: &Grid) -> f64 {
    let dx = (idx.x as f64) - ((grid.size.x as f64) + 1.) / 2.;
    let dy = (idx.y as f64) - ((grid.size.y as f64) + 1.) / 2.;
    let dz = (idx.z as f64) - ((grid.size.z as f64) + 1.) / 2.; //TODO: DISTNUMZ (if needed)
    grid.dn * (dx * dx + dy * dy + dz * dz).sqrt()
}

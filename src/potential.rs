//use num::Complex;

use ndarray::{Array3, Zip};
use ndarray_parallel::prelude::*;
use config::*;

//struct PotentialValue<T>(T); //Some kind of generic like this may work.

pub fn generate(config: &Config) -> Array3<f64> {
    let num = &config.grid.size;
    let init_size: [usize; 3] = [(num.x + 5) as usize, (num.y + 5) as usize, (num.z + 5) as usize];
    let mut v = Array3::<f64>::zeros(init_size);

    Zip::indexed(&mut v)
        .par_apply(|(i, j, k), x| *x = potential(config, &Index3 { x: i, y: j, z: k }));
    v
}


//TODO: Maybe have the OPTION to be complex rather than force it.
//For now we're dropping complex all together here.
pub fn potential(config: &Config, idx: &Index3) -> f64 {
    let num = &config.grid.size;
    match config.potential {
        Potential::NoPotential => 0.0,
        Potential::Cube => {
            if (idx.x > num.x / 4 && idx.x <= 3 * num.x / 4) &&
               (idx.y > num.y / 4 && idx.y <= 3 * num.y / 4) &&
               (idx.z > num.z / 4 && idx.z <= 3 * num.z / 4) {
                -10.0
            } else {
                0.0
            }
        }
        Potential::QuadWell => {
            if (idx.x > num.x / 4 && idx.x <= 3 * num.x / 4) &&
               (idx.y > num.y / 4 && idx.y <= 3 * num.y / 4) &&
               (idx.z > 3 * num.z / 8 && idx.z <= 5 * num.z / 8) {
                -10.0
            } else {
                0.0
            }
        }
        Potential::ComplexCoulomb => {
            let r = calculate_r(idx, &config.grid);
            if r < config.grid.dn {
                -1. / config.grid.dn
            } else {
                -1. / r
            }
        }
        Potential::FromFile => 0.0,
        Potential::FromScript => 0.0,
    }
}

fn calculate_r(idx: &Index3, grid: &Grid) -> f64 {
    let dx = (idx.x as f64) - ((grid.size.x as f64) + 1.) / 2.;
    let dy = (idx.y as f64) - ((grid.size.y as f64) + 1.) / 2.;
    let dz = (idx.z as f64) - ((grid.size.z as f64) + 1.) / 2.; //TODO: DISTNUMZ (if needed)
    grid.dn * (dx * dx + dy * dy + dz * dz).sqrt()
}

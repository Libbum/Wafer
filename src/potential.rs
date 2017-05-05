//use num::Complex;

//use config::{Potential, Config, Index3};
use config::*;

//struct PotentialValue<T>(T); //Some kind of generic like this may work.

//TODO: Maybe have the OPTION to be complex rather than force it.
//For now we're dropping complex all together here.
pub fn potential(config: &Config, idx: Index3) -> f64 {
    match config.potential {
        Potential::NoPotential => 0.0,
        Potential::Cube => {
            if (idx.x > config.grid.size.x / 4 && idx.x <= 3 * config.grid.size.x / 4) &&
               (idx.y > config.grid.size.y / 4 && idx.y <= 3 * config.grid.size.y / 4) &&
               (idx.z > config.grid.size.z / 4 && idx.z <= 3 * config.grid.size.z / 4) {
                -10.0
            } else {
                0.0
            }
        }
        Potential::QuadWell => {
            if (idx.x > config.grid.size.x / 4 && idx.x <= 3 * config.grid.size.x / 4) &&
               (idx.y > config.grid.size.y / 4 && idx.y <= 3 * config.grid.size.y / 4) &&
               (idx.z > 3 * config.grid.size.z / 8 && idx.z <= 5 * config.grid.size.z / 8) {
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
    }
}

fn calculate_r(idx: Index3, grid: &Grid) -> f64 {
    let dx = (idx.x as f64) - ((grid.size.x as f64) + 1.) / 2.;
    let dy = (idx.y as f64) - ((grid.size.y as f64) + 1.) / 2.;
    let dz = (idx.z as f64) - ((grid.size.z as f64) + 1.) / 2.; //TDOD: DISTNUMZ
    grid.dn * (dx * dx + dy * dy + dz * dz).sqrt()
}

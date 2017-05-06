//use num::Complex;

use std::time::Instant;
use ndarray::{Array3, Zip};
use ndarray_parallel::prelude::*;
use config::*;

//struct PotentialValue<T>(T); //Some kind of generic like this may work.
pub fn generate(config: &Config) {
    //For now, just print at the end.
    let num = &config.grid.size;
    let init_size: [usize; 3] = [(num.x + 5) as usize, (num.y + 5) as usize, (num.z + 5) as usize];
    let mut v = Array3::<f64>::zeros(init_size);
    let mut u = Array3::<f64>::zeros(init_size);

    let start_time = Instant::now();
    Zip::indexed(&mut v).par_apply(|i, x| {
        *x = potential(config,
                       &Index3 {
                           x: i.0 as u32,
                           y: i.1 as u32,
                           z: i.2 as u32,
                       })
    });
    let elapsed = start_time.elapsed();
    let time_taken = (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64 / 1000_000_000.0);
    println!("Par: {} seconds.", time_taken);
    //We can par_iter, but not for indexed data it seems.
    //v.par_iter_mut().for_each(|elt| *elt += 1.);
    let start_time2 = Instant::now();
    for ((i, j, k), elt) in u.indexed_iter_mut() {
        *elt = potential(config,
                         &Index3 {
                             x: i as u32,
                             y: j as u32,
                             z: k as u32,
                         });
    }
    let elapsed2 = start_time2.elapsed();
    let time_taken2 = (elapsed2.as_secs() as f64) +
                      (elapsed2.subsec_nanos() as f64 / 1000_000_000.0);
    println!("Seq: {} seconds.", time_taken2);
    println!("{}", v == u);
}


//TODO: Maybe have the OPTION to be complex rather than force it.
//For now we're dropping complex all together here.
fn potential(config: &Config, idx: &Index3) -> f64 {
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

fn calculate_r(idx: &Index3, grid: &Grid) -> f64 {
    let dx = (idx.x as f64) - ((grid.size.x as f64) + 1.) / 2.;
    let dy = (idx.y as f64) - ((grid.size.y as f64) + 1.) / 2.;
    let dz = (idx.z as f64) - ((grid.size.z as f64) + 1.) / 2.; //TDOD: DISTNUMZ
    grid.dn * (dx * dx + dy * dy + dz * dz).sqrt()
}

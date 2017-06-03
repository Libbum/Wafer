use ndarray::{Array3, Zip};
use ndarray_parallel::prelude::*;
use slog::Logger;
use std::f64::MAX;
use std::f64::consts::PI;

use config::{Config, Index3, Grid, PotentialType};
use input;
use output;
use grid;
use errors::*;

#[derive(Debug)]
/// Holds the potential arrays for the current simulation.
pub struct Potentials {
    /// The potential
    pub v: Array3<f64>,
    /// Ancillary array `a`
    pub a: Array3<f64>,
    /// Ancillary array `b`
    pub b: Array3<f64>,
}

/// A public wrapper around `potential`. Where `potential` does the calculation for a
/// single point, `generate` builds the entire grid.
///
/// # Arguments
///
/// * `config` - configuration data struct
///
/// # Returns
///
/// A 3D array of potential values of the requested size.
/// Or an error if called on the wrong potential type.
pub fn generate(config: &Config) -> Result<Array3<f64>> {
    let num = &config.grid.size;
    let bb = config.central_difference.bb();
    let init_size: [usize; 3] = [num.x + bb, num.y + bb, num.z + bb];
    let mut v = Array3::<f64>::zeros(init_size);

    Zip::indexed(&mut v)
        .par_apply(|(i, j, k), x| match potential(config, &Index3 { x: i, y: j, z: k }) {
                       Ok(result) => *x = result,
                       Err(err) => panic!("{}", err), //NOTE: We panic here rather than generating an error.
                                                      // First: I'm not sure how to return the error out of the closure,
                                                      // and second: This error can only be `NotAvailable`, so this should
                                                      // never run here. If it does, a panic is probably a better halter.
                   });
    Ok(v)
}

/// Handles the potential loading from file, or generating depending on configuration
/// Ancillary arrays are also generated here.
///
/// # Arguments
///
/// * `config` - Configuration struct
/// * `log` - Logger reference
///
/// # Returns
///
/// A `Potentials` struct with the potential `v` and ancillary arrays `a` and `b`.
pub fn load_arrays(config: &Config, log: &Logger) -> Result<Potentials> {
    let mut minima: f64 = MAX;
    let bb = config.central_difference.bb();
    let v: Array3<f64> = match config.potential {
        PotentialType::FromFile => {
            info!(log, "Loading potential from file");
            let num = &config.grid.size;
            let init_size: [usize; 3] = [num.x + bb, num.y + bb, num.z + bb];
            let pot = input::potential(init_size, bb, config.output.binary_files, log)?;
            Ok(pot)
        }
        PotentialType::FromScript => {
            match config.script_location {
                Some(ref file) => {
                    let pot = input::script_potential(file, &config.grid, bb, log)?;
                    Ok(pot)
                }
                None => Err(ErrorKind::ScriptNotFound.into()),
            }
        }
        _ => {
            info!(log, "Calculating potential array");
            generate(config)
        }
    }?; //Note the try here.

    let b = 1. / (1. + config.grid.dt * &v / 2.);
    let a = (1. - config.grid.dt * &v / 2.) * &b;

    // We can't do this in a par.
    // AFAIK, this is the safest way to work with the float here.
    for el in v.iter() {
        if el.is_finite() {
            minima = minima.min(*el);
        }
    }

    if config.output.save_potential {
        info!(log, "Saving potential to disk");
        let work = grid::get_work_area(&v, config.central_difference.ext());
        output::potential(&work, &config.project_name, config.output.binary_files)?;
    }

    Ok(Potentials { v: v, a: a, b: b })
}

//TODO: For now we're dropping complex all together, but this is needed.
/// Generates a potential for the current simulation at a particular index point.
///
/// # Arguments
///
/// * `config` - configuration data struct
/// * `idx` - an index to calculate the potential value at
///
/// # Returns
///
/// A double with the potential value at the requested index, or an error if the function
/// is called for an invalid potential type.
fn potential(config: &Config, idx: &Index3) -> Result<f64> {
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
        PotentialType::Periodic => {
            let mut temp = (2. * PI * (idx.x as f64 - 1.) / (num.x as f64 - 1.)).sin() *
                           (2. * PI * (idx.x as f64 - 1.) / (num.x as f64 - 1.)).sin();
            temp *= (2. * PI * (idx.y as f64 - 1.) / (num.y as f64 - 1.)).sin() *
                    (2. * PI * (idx.y as f64 - 1.) / (num.y as f64 - 1.)).sin();
            temp *= (2. * PI * (idx.z as f64 - 1.) / (num.z as f64 - 1.)).sin() *
                    (2. * PI * (idx.z as f64 - 1.) / (num.z as f64 - 1.)).sin();
            Ok(-temp + 1.)
        }
        PotentialType::Coulomb |
        PotentialType::ComplexCoulomb => {
            //TODO: ComplexCoulomb returns real until we have complex types
            let r = config.grid.dn * (calculate_r2(idx, &config.grid)).sqrt();
            if r < config.grid.dn {
                Ok(-1. / config.grid.dn)
            } else {
                Ok(-1. / r)
            }
        }
        PotentialType::ElipticalCoulomb => {
            let dx = idx.x as f64 - (num.x as f64 + 1.) / 2.;
            let dy = idx.y as f64 - (num.y as f64 + 1.) / 2.;
            let dz = (idx.z as f64 - (num.z as f64 + 1.) / 2.) * 2.;
            let r = config.grid.dn * (dx * dx + dy * dy + dz * dz).sqrt();
            if r < config.grid.dn {
                Ok(0.0)
            } else {
                Ok(-1. / r + 1. / config.grid.dn)
            }
        }
        PotentialType::SimpleCornell => {
            // NOTE: units here are GeV for energy/momentum and GeV^(-1) for distance
            let r = config.grid.dn * (calculate_r2(idx, &config.grid)).sqrt();
            if r < config.grid.dn {
                Ok(4. * config.mass)
            } else {
                Ok(-0.5 * (4. / 3.) / r + config.sig * r + 4. * config.mass)
            }
        }
        PotentialType::FullCornell => {
            //NOTE: units here are GeV for energy/momentum and GeV^(-1) for distance
            let t = 1.0; //TODO: This should be an optional parameter for FullCornell only
            let xi: f64 = 0.0; //TODO: This should be an optional parameter for FullCornell only
            let dz = idx.z as f64 - (config.grid.size.z as f64 + 1.) / 2.;
            let r = config.grid.dn * (calculate_r2(idx, &config.grid)).sqrt();
            let md = mu(t) *
                     (1. +
                      0.07 * xi.powf(0.2) *
                      (1. - config.grid.dn * config.grid.dn * dz * dz / (r * r))) *
                     (1. + xi).powf(-0.29);
            if r < config.grid.dn {
                Ok(4. * config.mass)
            } else {
                Ok(-alphas(2. * PI * t) * (4. / 3.) * (-md * r).exp() / r +
                   config.sig * (1. - (-md * r).exp()) / md -
                   0.8 * config.sig / (4. * config.mass * config.mass * r) +
                   4. * config.mass)
            }
        }
        PotentialType::Harmonic |
        PotentialType::ComplexHarmonic => {
            //TODO: ComplexHarmonic is real until we have Complex types
            let r = config.grid.dn * (calculate_r2(idx, &config.grid)).sqrt();
            Ok(r * r / 2.)
        }
        PotentialType::Dodecahedron => {
            //Varför inte?
            let dx = idx.x as f64 - (num.x as f64 + 1.) / 2.;
            let dy = idx.y as f64 - (num.y as f64 + 1.) / 2.;
            let dz = idx.z as f64 - (num.z as f64 + 1.) / 2.;
            let x = dx / ((num.x as f64 - 1.) / 2.);
            let y = dy / ((num.y as f64 - 1.) / 2.);
            let z = dz / ((num.z as f64 - 1.) / 2.);
            if 12.70820393249937 + 11.210068307552588 * x >= 14.674169922690343 * z &&
               11.210068307552588 * x <= 12.70820393249937 + 14.674169922690343 * z &&
               5.605034153776295 * (3.23606797749979 * x - 1.2360679774997896 * z) <=
               6. * (4.23606797749979 + 5.23606797749979 * y) &&
               18.1382715378281 * x + 3.464101615137755 * z <= 12.70820393249937 &&
               9.06913576891405 * x + 15.70820393249937 * y <=
               12.70820393249937 + 3.464101615137755 * z &&
               9.70820393249937 * y <=
               12.70820393249937 + 5.605034153776294 * x + 14.674169922690343 * z &&
               12.70820393249937 + 5.605034153776294 * x + 9.70820393249937 * y +
               14.674169922690343 * z >= 0. &&
               15.70820393249937 * y + 3.464101615137755 * z <=
               12.70820393249937 + 9.06913576891405 * x &&
               5.605034153776295 * (-6.47213595499958 * x - 1.2360679774997896 * z) <=
               25.41640786499874 &&
               3.464101615137755 * z <=
               9.06913576891405 * x + 3. * (4.23606797749979 + 5.23606797749979 * y) &&
               1.7320508075688772 * (3.23606797749979 * x + 8.47213595499958 * z) <=
               3. * (4.23606797749979 + 3.23606797749979 * y) &&
               5.605034153776294 * x + 9.70820393249937 * y + 14.674169922690343 * z <=
               12.70820393249937 {
                Ok(-100.)
            } else {
                Ok(0.0)
            }
        }
        PotentialType::FromFile |
        PotentialType::FromScript => Err(ErrorKind::PotentialNotAvailable.into()), //TODO: Script may not need to error.
    }
}


//TODO: For now we're dropping complex all together, but this is needed.
//TODO: We need potential_sub file outputs for those which require it.
// Then here from_file can be treated differently.
/// Calculate binding energy offset (if any). Follows the `potential` input/output arguments.
/// Used if calculation requires indexing. If not, call `potential_sub` instead. Currency only
/// `FullCornell` requires this routine.
pub fn potential_sub_idx(config: &Config, idx: &Index3) -> Result<f64> {
    match config.potential {
        PotentialType::FullCornell => {
            let dz = idx.z as f64 - (config.grid.size.z as f64 + 1.) / 2.;
            let r = config.grid.dn * (calculate_r2(idx, &config.grid)).sqrt();
            let t = 1.0; //TODO: This should be an optional parameter for FullCornell only
            let xi: f64 = 0.0; //TODO: This should be an optional parameter for FullCornell only
            let md = mu(t) *
                     (1. +
                      0.07 * xi.powf(0.2) *
                      (1. - config.grid.dn * config.grid.dn * dz * dz / (r * r))) *
                     (1. + xi).powf(-0.29);
            Ok(config.sig / md + 4. * config.mass)
        }
        _ => Err(ErrorKind::PotentialNotAvailable.into()),
    }
}

/// Calculate binding energy offset (if any). Follows the `potential` input/output arguments.
/// `FullCornell`, and subsequent potentials that require indexed values must call `potential_sub_idx`.
pub fn potential_sub(config: &Config) -> Result<f64> {
    match config.potential {
        PotentialType::NoPotential |
        PotentialType::Cube |
        PotentialType::QuadWell |
        PotentialType::Periodic |
        PotentialType::Coulomb |
        PotentialType::ComplexCoulomb |
        PotentialType::Harmonic |
        PotentialType::ComplexHarmonic |
        PotentialType::Dodecahedron |
        PotentialType::FromScript | //TODO: Script should be treated differently.
        PotentialType::FromFile => Ok(0.0),
        PotentialType::ElipticalCoulomb => Ok(1. / config.grid.dn),
        PotentialType::SimpleCornell => Ok(4.0 * config.mass),
        PotentialType::FullCornell => Err(ErrorKind::PotentialNotAvailable.into()),
    }
}


/// Calculates squared distance
pub fn calculate_r2(idx: &Index3, grid: &Grid) -> f64 {
    let dx = (idx.x as f64) - ((grid.size.x as f64) + 1.) / 2.;
    let dy = (idx.y as f64) - ((grid.size.y as f64) + 1.) / 2.;
    let dz = (idx.z as f64) - ((grid.size.z as f64) + 1.) / 2.;
    (dx * dx + dy * dy + dz * dz)
}

/// Running coupling. Used for Cornell potentials.
fn alphas(mu: f64) -> f64 {
    let nf = 2.0; //TODO: This should be an optional parameter for FullCornell only
    let b0 = 11. - 2. * nf / 3.;
    let b1 = 51. - 19. * nf / 3.;
    let b2 = 2857. - 5033. * nf / 9. + 325. * nf * nf / 27.;

    let r = 2.3; // scale adjusted to match lattice data from hep-lat/0503017v2

    let l = 2. * (mu / r).ln();

    4. * PI *
    (1. - 2. * b1 * l.ln() / (b0 * b0 * l) +
     4. * b1 * b1 * ((l.ln() - 0.5) * (l.ln() - 0.5) + b2 * b0 / (8. * b1 * b1) - 5.0 / 4.0) /
     (b0 * b0 * b0 * b0 * l * l)) / (b0 * l)
}

/// Debye screening mass. Used for Cornell potentials.
fn mu(t: f64) -> f64 {
    let nf = 2.0; //TODO: This should be an optional parameter for FullCornell only
    let tc = 0.2; //TODO: This should be an optional parameter for FullCornell only
    1.4 * ((1. + nf / 6.) * 4. * PI * alphas(2. * PI * t)).sqrt() * t * tc
}

#[cfg(test)]
mod tests {
    use super::*;
    use config::{Grid, Index3};
    //TODO: Build a test Config to throw in to functions.

    macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => ({
        let eps = 1.0e-6;
        let (a, b) = (&$a, &$b);
        assert!((*a - *b).abs() < eps,
                "assertion failed: `(left !== right)` \
                           (left: `{:?}`, right: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
                 *a, *b, eps, (*a - *b).abs());
    });
    ($a:expr, $b:expr, $eps:expr) => ({
        let (a, b) = (&$a, &$b);
        assert!((*a - *b).abs() < $eps,
                "assertion failed: `(left !== right)` \
                           (left: `{:?}`, right: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
                 *a, *b, $eps, (*a - *b).abs());
    })
    }

    #[test]
    fn distance_squared() {
        let grid = Grid { size: Index3 { x: 5, y: 6, z: 3 }, dn: 0.1, dt: 3e-5 };
        let idx = Index3 { x: 3, y: 3, z: 3 };
        assert_approx_eq!(calculate_r2(&idx, &grid), 1.25);
    }

    #[test]
    fn running_coupling() {
        let md = 3.2;
        assert_approx_eq!(alphas(md), 6.189593433886306, 1e-14);
    }
    #[test]
    fn debye_screening_mass() {
        let t = 5.2;
        assert_approx_eq!(mu(t), 2.604838027702063, 1e-14);
    }
}

use ndarray::{Array3, Zip};
use ndarray_parallel::prelude::*;
use slog::Logger;
use std::f64::consts::PI;
use std::f64::MAX;

use config::{Config, Grid, Index3, PotentialType};
use errors::*;
use grid;
use input;
use output;

#[derive(Debug)]
/// Holds the potential arrays for the current simulation.
pub struct Potentials {
    /// The potential.
    pub v: Array3<f64>,
    /// Ancillary array `a`.
    pub a: Array3<f64>,
    /// Ancillary array `b`.
    pub b: Array3<f64>,
    /// Potsub value.
    pub pot_sub: (Option<Array3<f64>>, Option<f64>),
}

#[derive(Debug, Deserialize, Serialize)]
/// A single value struct for potential_sub outputs that
/// do not require an entire array.
pub struct PotentialSubSingle {
    /// Value of `potential_sub` for the current potential.
    pub pot_sub: f64,
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

    Zip::indexed(&mut v).par_apply(|(i, j, k), x| {
        match potential(config, &Index3 { x: i, y: j, z: k }) {
            Ok(result) => *x = result,
            Err(err) => panic!("{}", err), //NOTE: We panic here rather than generating an error.
                                           // First: I'm not sure how to return the error out of the closure,
                                           // and second: This error can only be `NotAvailable`, so this should
                                           // never run here. If it does, a panic is probably a better halter.
        }
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
    let num = &config.grid.size;
    let v: Array3<f64> = match config.potential {
        PotentialType::FromFile => {
            let init_size: [usize; 3] = [num.x + bb, num.y + bb, num.z + bb];
            info!(log, "Loading potential from file");
            let pot = input::potential(init_size, bb, &config.output.file_type, log)
                .chain_err(|| ErrorKind::LoadPotential)?;
            Ok(pot)
        }
        PotentialType::FromScript => match config.script_location {
            Some(ref file) => {
                let pot = input::script_potential(file, &config.grid, bb, log)
                    .chain_err(|| ErrorKind::LoadPotential)?;
                Ok(pot)
            }
            None => Err(ErrorKind::ScriptNotFound.into()),
        },
        _ => {
            info!(log, "Calculating potential array");
            generate(config)
        }
    }?;

    let b = 1. / (1. + config.grid.dt * &v / 2.);
    let a = (1. - config.grid.dt * &v / 2.) * &b;

    let sub_size: [usize; 3] = [num.x, num.y, num.z];
    // Try to read a potential_sub from file first, and deal with inconsistencies in the setup.
    // If there are no files on disk, then potential_sub should be calculated.
    let pot_sub = if let Ok(pot_sub_info) =
        input::potential_sub(sub_size, &config.output.file_type, log)
    {
        if pot_sub_info.0.is_none()
            && pot_sub_info.1.is_some()
            && config.potential.variable_pot_sub()
        {
            error!(log, "Potential_sub input file contains a singular value, but potential type is FullCornell. Update or remove the potential file in the input directory before continuing.");
            return Err(ErrorKind::WrongPotentialSubDims.into());
        } else if pot_sub_info.0.is_some()
            && pot_sub_info.1.is_none()
            && !config.potential.variable_pot_sub()
        {
            error!(log, "Potential_sub input file contains an array, but potential type is not FullCornell. Update or remove the potential file in the input directory before continuing.");
            return Err(ErrorKind::WrongPotentialSubDims.into());
        } else {
            info!(log, "Potential_sub loaded from disk");
            pot_sub_info
        }
    } else if config.potential.variable_pot_sub() {
        let mut full_sub = Array3::<f64>::zeros((sub_size[0], sub_size[1], sub_size[2]));
        Zip::indexed(&mut full_sub).par_apply(|(i, j, k), full_sub| {
            let idx = Index3 { x: i, y: j, z: k };
            *full_sub = match potential_sub_idx(config, &idx) {
                Ok(p) => p,
                Err(err) => panic!("Calling invalid potential_sub routine: {}", err),
            };
        });
        info!(log, "Variable potential_sub calculated directly");
        (Some(full_sub), None)
    } else {
        let single_sub = potential_sub(config)?;
        info!(log, "Constant potential_sub calculated directly");
        if single_sub > 0.0 {
            (None, Some(single_sub))
        } else {
            (None, None)
        }
    };

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
        if let Err(err) = output::potential(&work, &config.project_name, &config.output.file_type) {
            warn!(log, "Could not write potential to disk: {}", err);
        }
        if let Err(err) = output::potential_sub(&config) {
            warn!(log, "Could not write potential_sub to disk: {}", err);
        }
    }

    Ok(Potentials { v, a, b, pot_sub })
}

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
            if (idx.x > num.x / 4 && idx.x <= 3 * num.x / 4)
                && (idx.y > num.y / 4 && idx.y <= 3 * num.y / 4)
                && (idx.z > num.z / 4 && idx.z <= 3 * num.z / 4)
            {
                Ok(-10.0)
            } else {
                Ok(0.0)
            }
        }
        PotentialType::QuadWell => {
            if (idx.x > num.x / 4 && idx.x <= 3 * num.x / 4)
                && (idx.y > num.y / 4 && idx.y <= 3 * num.y / 4)
                && (idx.z > 3 * num.z / 8 && idx.z <= 5 * num.z / 8)
            {
                Ok(-10.0)
            } else {
                Ok(0.0)
            }
        }
        PotentialType::Periodic => {
            let mut temp = (2. * PI * (idx.x as f64 - 1.) / (num.x as f64 - 1.)).sin()
                * (2. * PI * (idx.x as f64 - 1.) / (num.x as f64 - 1.)).sin();
            temp *= (2. * PI * (idx.y as f64 - 1.) / (num.y as f64 - 1.)).sin()
                * (2. * PI * (idx.y as f64 - 1.) / (num.y as f64 - 1.)).sin();
            temp *= (2. * PI * (idx.z as f64 - 1.) / (num.z as f64 - 1.)).sin()
                * (2. * PI * (idx.z as f64 - 1.) / (num.z as f64 - 1.)).sin();
            Ok(-temp + 1.)
        }
        PotentialType::Coulomb | PotentialType::ComplexCoulomb => {
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
            let md = mu(t)
                * (1. + 0.07
                    * xi.powf(0.2)
                    * (1. - config.grid.dn * config.grid.dn * dz * dz / (r * r)))
                * (1. + xi).powf(-0.29);
            if r < config.grid.dn {
                Ok(4. * config.mass)
            } else {
                Ok(-alphas(2. * PI * t) * (4. / 3.) * (-md * r).exp() / r
                    + config.sig * (1. - (-md * r).exp()) / md
                    - 0.8 * config.sig / (4. * config.mass * config.mass * r)
                    + 4. * config.mass)
            }
        }
        PotentialType::Harmonic | PotentialType::ComplexHarmonic => {
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
            if 12.708_203_932_499_37 + 11.210_068_307_552_588 * x >= 14.674_169_922_690_343 * z
                && 11.210_068_307_552_588 * x <= 12.708_203_932_499_37 + 14.674_169_922_690_343 * z
                && 5.605_034_153_776_295 * (3.236_067_977_499_79 * x - 1.236_067_977_499_789_6 * z)
                    <= 6. * (4.236_067_977_499_79 + 5.236_067_977_499_79 * y)
                && 18.138_271_537_828_1 * x + 3.464_101_615_137_755 * z <= 12.708_203_932_499_37
                && 9.069_135_768_914_05 * x + 15.708_203_932_499_37 * y
                    <= 12.708_203_932_499_37 + 3.464_101_615_137_755 * z
                && 9.708_203_932_499_37 * y
                    <= 12.708_203_932_499_37 + 5.605_034_153_776_294 * x + 14.674_169_922_690_343 * z
                && 12.708_203_932_499_37
                    + 5.605_034_153_776_294 * x
                    + 9.708_203_932_499_37 * y
                    + 14.674_169_922_690_343 * z >= 0.
                && 15.708_203_932_499_37 * y + 3.464_101_615_137_755 * z
                    <= 12.708_203_932_499_37 + 9.069_135_768_914_05 * x
                && 5.605_034_153_776_295 * (-6.472_135_954_999_58 * x - 1.236_067_977_499_789_6 * z)
                    <= 25.416_407_864_998_74
                && 3.464_101_615_137_755 * z
                    <= 9.069_135_768_914_05 * x + 3. * (4.236_067_977_499_79 + 5.236_067_977_499_79 * y)
                && 1.732_050_807_568_877_2 * (3.236_067_977_499_79 * x + 8.472_135_954_999_58 * z)
                    <= 3. * (4.236_067_977_499_79 + 3.236_067_977_499_79 * y)
                && 5.605_034_153_776_294 * x + 9.708_203_932_499_37 * y + 14.674_169_922_690_343 * z
                    <= 12.708_203_932_499_37
            {
                Ok(-100.)
            } else {
                Ok(0.0)
            }
        }
        PotentialType::FromFile | PotentialType::FromScript => {
            Err(ErrorKind::PotentialNotAvailable.into())
        } //TODO: Script may not need to error.
    }
}

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
            let md = mu(t)
                * (1. + 0.07
                    * xi.powf(0.2)
                    * (1. - config.grid.dn * config.grid.dn * dz * dz / (r * r)))
                * (1. + xi).powf(-0.29);
            Ok(config.sig / md + 4. * config.mass)
        }
        _ => Err(ErrorKind::PotentialNotAvailable.into()),
    }
}

/// Calculate binding energy offset (if any). Follows the `potential`
/// input/output arguments. `FullCornell`, and subsequent potentials that require
/// indexed values must call `potential_sub_idx`.
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

    4. * PI
        * (1. - 2. * b1 * l.ln() / (b0 * b0 * l)
            + 4.
                * b1
                * b1
                * ((l.ln() - 0.5) * (l.ln() - 0.5) + b2 * b0 / (8. * b1 * b1) - 5.0 / 4.0)
                / (b0 * b0 * b0 * b0 * l * l)) / (b0 * l)
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
        ($a:expr, $b:expr) => {{
            let eps = 1.0e-6;
            let (a, b) = (&$a, &$b);
            assert!(
                (*a - *b).abs() < eps,
                "assertion failed: `(left !== right)` \
                 (left: `{:?}`, right: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
                *a,
                *b,
                eps,
                (*a - *b).abs()
            );
        }};
        ($a:expr, $b:expr, $eps:expr) => {{
            let (a, b) = (&$a, &$b);
            assert!(
                (*a - *b).abs() < $eps,
                "assertion failed: `(left !== right)` \
                 (left: `{:?}`, right: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
                *a,
                *b,
                $eps,
                (*a - *b).abs()
            );
        }};
    }

    #[test]
    fn distance_squared() {
        let grid = Grid {
            size: Index3 { x: 5, y: 6, z: 3 },
            dn: 0.1,
            dt: 3e-5,
        };
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

use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array3, ArrayView3, ArrayViewMut3, Zip};
use ndarray_parallel::prelude::*;
use slog::Logger;
use std::f64::MAX;
use config;
use config::{Config, CentralDifference, Grid, Index3, InitialCondition, PotentialType};
use potential;
use potential::Potentials;
use input;
use output;
use errors::*;

#[derive(Debug)]
/// Holds all computed observables for the current wavefunction.
pub struct Observables {
    /// Normalised total energy.
    pub energy: f64,
    /// A squared normalisation. Square root occurs later when needed to apply a complete
    /// normalisation condition. This needs to be separate as we include other adjustments
    /// from time to time.
    pub norm2: f64,
    /// The value of the potential at infinity. This is used to calculate the binding energy.
    pub v_infinity: f64,
    /// Coefficient of determination
    pub r2: f64,
}

/// Runs the calculation and holds long term (system time) wavefunction storage
pub fn run(config: &Config, log: &Logger) -> Result<()> {

    let potentials = potential::load_arrays(config, log)?;

    let mut w_store: Vec<Array3<f64>> = Vec::new();
    if config.wavenum > 0 {
        //We require wavefunctions from disk, even if initial condition is not `FromFile`
        //The wavenum = 0 case is handled later
        input::load_wavefunctions(config, log, &mut w_store)?;
    }

    info!(log, "Starting calculation");

    for wnum in config.wavenum..config.wavemax + 1 {
        solve(config, log, &potentials, wnum, &mut w_store)?;
    }
    Ok(())
}


/// Runs the actual computation once system is setup and ready.
fn solve(config: &Config,
         log: &Logger,
         potentials: &Potentials,
         wnum: u8,
         w_store: &mut Vec<Array3<f64>>)
         -> Result<()> {

    // Initial conditions from config file if ground state,
    // but start from previously converged wfn if we're an excited state.
    let mut phi: Array3<f64> = if wnum > 0 {
        let num = &config.grid.size;
        let bb = config.central_difference.bb();
        let init_size: [usize; 3] = [num.x as usize + bb,
                                     num.y as usize + bb,
                                     num.z as usize + bb];
        // Try to load current wavefunction from disk.
        // If not, we start with the previously converged function
        if let Ok(wfn) = input::wavefunction(wnum, init_size, bb, config.output.binary_files, log) {
            info!(log, "Loaded (current) wavefunction {} from disk", wnum);
            // If people are lazy or forget, their input files may contaminate
            // the run here.
            // For example: starting wfn = 0 with random gaussian, max = 3.
            // User has wavefunction_{0,1}.csv in `input` from old run.
            // System will generate the random ground state and calculate it fine,
            // then try to load the old excited state 1 from disk, which is most likely
            // something completely irrelevant. Throw a warning for this scenario.
            if config.init_condition != InitialCondition::FromFile && wnum > config.wavenum {
                warn!(log,
                      "Loaded a higher order wavefunction from disk although Initial conditions are set to '{}'.",
                      config.init_condition);
            }
            wfn
        } else {
            info!(log,
                  "Loaded wavefunction {} from memory as initial condition",
                  wnum - 1);
            // If we fail to load due to a error or missing, we just start from the previous
            // stored wavefunction.
            // We have to clone here, otherwise we cannot borrow w_store for the GS routines.
            w_store[wnum as usize - 1].clone()
        }
    } else {
        //This sorts out loading from disk if we are on wavefunction 0.
        config::set_initial_conditions(config, log)?
    };

    output::print_observable_header(wnum);
    let prog_bar = ProgressBar::new(100);
    prog_bar.set_style(ProgressStyle::default_bar()
                      .template("{msg}\n\n[{elapsed_precise}] |{wide_bar:.cyan/blue}| {spinner:.green} ETA: {eta:>3}")
                      .progress_chars("█▓░")
                      .tick_chars("⣾⣽⣻⢿⡿⣟⣯⣷ "));
    prog_bar.set_position(0);

    let mut step = 0;
    let mut done = false;
    let mut converged = false;
    let mut last_energy = MAX; //std::f64::MAX
    let mut diff_old = MAX;
    let mut display_energy = MAX;
    while !done {

        let observables = compute_observables(config, potentials, &phi);
        let norm_energy = observables.energy / observables.norm2;
        let tau = (step as f64) * config.grid.dt;
        // Orthoganalise wavefunction
        if wnum > 0 {
            normalise_wavefunction(&mut phi, observables.norm2);
            orthogonalise_wavefunction(wnum, &mut phi, w_store);
        }
        //NOTE: Need to do a floating point comparison here if we want steps to be more than 2^64 (~1e19)
        // But I think it's just best to not have this option. 1e19 max.
        if step % config.output.snap_update == 0 {
            //TODO: I think we can do away with SNAPUPDATE now. Kill this if.
            config::symmetrise_wavefunction(config, &mut phi);
            normalise_wavefunction(&mut phi, observables.norm2);
            let diff = (norm_energy - last_energy).abs();
            if diff < config.tolerance {
                prog_bar.finish_and_clear();
                println!("{}", output::print_measurements(tau, diff, &observables));
                output::finalise_measurement(&observables,
                                             wnum,
                                             config.grid.size.x as f64,
                                             &config.project_name,
                                             config.output.binary_files)?;
                converged = true;
                break;
            } else {
                display_energy = last_energy;
                last_energy = norm_energy;
            }
        }
        let diff = (display_energy - norm_energy).abs();

        // Output status to screen
        if let Some(estimate) = eta(step, diff_old, diff, config) {
            let percent = (100. -
                           (estimate /
                            ((step as f64 / config.output.screen_update as f64) + estimate) *
                            100.))
                    .floor();
            if percent.is_finite() {
                prog_bar.set_position(percent as u64);
            }
        }
        prog_bar.set_message(&output::print_measurements(tau, diff, &observables));

        // Evolve solution until next screen update
        if step < config.max_steps {
            evolve(wnum, config, potentials, &mut phi, w_store);
        }
        diff_old = diff;
        step += config.output.screen_update;
        done = step > config.max_steps;
    }

    if config.output.save_wavefns {
        //NOTE: This wil save regardless of whether it is converged or not, so we flag it if that's the case.
        info!(log, "Saving wavefunction {} to disk", wnum);
        let work = get_work_area(&phi, config.central_difference.ext());
        if let Err(err) = output::wavefunction(&work,
                                               wnum,
                                               converged,
                                               &config.project_name,
                                               config.output.binary_files) {
            warn!(log, "Could not write wavefunction to disk: {}", err);
        }
    }

    if converged {
        info!(log, "Caluculation Converged");
        w_store.push(phi); //Save state
        Ok(())
    } else {
        Err(ErrorKind::MaxStep.into())
    }
}

/// Estimates completion time for the convergence of the current wavefunction.
///
/// # Returns
///
/// An estimate of the number of `screen_update` cycles to go until convergence.
/// Uses an option as it may not be finite.
fn eta(step: u64, diff_old: f64, diff_new: f64, config: &Config) -> Option<f64> {
    //Convergenge is done in exponential time after a short stabilisation stage.
    //So we can use the point slope form of a linear equation to find an estimate to hit the tolerance on a semilogy scale.
    //y - y1 = m(x-x1); where here we use (x1,y1) as (step,diff_new), y is tolerance, find m then solve for x
    let x1 = step as f64;
    let y1 = diff_new.log10();
    let rise = y1 - diff_old.log10();
    let run = config.output.screen_update as f64;
    let m = rise / run;

    //Step at which we estimate reaching tolerance.
    let x = ((config.tolerance.log10() - y1) / m) + x1;

    //Now to return an expectation
    // Initially, we obtain a -inf which needs to be treated, and we can't correctly estimate the runtime until
    // the unstable region has been crossed. Luckily, we can use a previous estimate to identify this region.
    // We'll handle the second issue outside though and just return the estimate here.
    if x.is_finite() {
        let estimate = ((x - x1) / run).floor();
        //This catch stops our percentage from going above 100% and making indicatif throw a memory error.
        if estimate > 0. {
            return Some(estimate);
        }
    }
    None
}

/// Computes observable values of the system, for example the energy
fn compute_observables(config: &Config, potentials: &Potentials, phi: &Array3<f64>) -> Observables {
    let energy = wfnc_energy(config, potentials, phi);
    let work = get_work_area(phi, config.central_difference.ext());
    let norm2 = get_norm_squared(&work);
    let v_infinity = get_v_infinity_expectation_value(&work, config);
    let r2 = get_r_squared_expectation_value(&work, &config.grid);

    Observables {
        energy: energy,
        norm2: norm2,
        v_infinity: v_infinity,
        r2: r2,
    }
}

/// Normalisation of wavefunction
fn get_norm_squared(w: &ArrayView3<f64>) -> f64 {
    //NOTE: No complex conjugation due to all real input for now
    w.into_par_iter().map(|&el| el * el).sum()
}

/// Get v infinity
fn get_v_infinity_expectation_value(w: &ArrayView3<f64>, config: &Config) -> f64 {
    //NOTE: No complex conjugation due to all real input for now
    let mut work = Array3::<f64>::zeros(w.dim());
    //NOTE: I'm unsure if we really need the errors here. We already match types.
    //It may be overkill. If so, then this is just added complexity. Consider removing them.
    //The non indexed version could be sent up the chain too.
    if let PotentialType::FullCornell = config.potential {
        Zip::indexed(&mut work)
            .and(w)
            .par_apply(|(i, j, k), work, &w| {
                           let idx = Index3 { x: i, y: j, z: k };
                           let potsub = match potential::potential_sub_idx(config, &idx) {
                               Ok(p) => p,
                               Err(err) => panic!("Calling invalid potential_sub routine: {}", err),
                           };
                           *work = w * w * potsub;
                       });
    } else {
        let potsub = match potential::potential_sub(config) {
            Ok(p) => p,
            Err(err) => panic!("Calling invalid potential_sub routine: {}", err),
        };
        Zip::from(&mut work)
            .and(w)
            .par_apply(|work, &w| { *work = w * w * potsub; });
    }
    work.scalar_sum()
}

/// Get r2
fn get_r_squared_expectation_value(w: &ArrayView3<f64>, grid: &Grid) -> f64 {
    //NOTE: No complex conjugation due to all real input for now
    let mut work = Array3::<f64>::zeros(w.dim());
    Zip::indexed(&mut work)
        .and(w)
        .par_apply(|(i, j, k), work, &w| {
                       let idx = Index3 { x: i, y: j, z: k };
                       let r2 = potential::calculate_r2(&idx, grid);
                       *work = w * w * r2;
                   });
    work.scalar_sum()
}

/// Gets energy of the corresponding wavefunction
fn wfnc_energy(config: &Config, potentials: &Potentials, phi: &Array3<f64>) -> f64 {
    let ext = config.central_difference.ext();
    let w = get_work_area(phi, ext);
    let v = get_work_area(&potentials.v, ext);

    let mut work = Array3::<f64>::zeros(w.dim());
    //TODO: We don't have any complex conjugation here.
    match config.central_difference {
        CentralDifference::ThreePoint => {
            let denominator = 2. * config.grid.dn * config.grid.dn * config.mass;
            Zip::indexed(&mut work)
                .and(v)
                .and(w)
                .par_apply(|(i, j, k), work, &v, &w| {
                    // Offset indexes as we are already in a slice
                    let lx = i as isize + 1;
                    let ly = j as isize + 1;
                    let lz = k as isize + 1;
                    let o = 1;
                    // get a slice which gives us our matrix of central difference points
                    let l = phi.slice(s![lx - 1..lx + 2, ly - 1..ly + 2, lz - 1..lz + 2]);
                    // l can now be indexed with local offset `o` and modifiers
                    *work = v * w * w -
                            w *
                            (l[[o + 1, o, o]] + l[[o - 1, o, o]] + l[[o, o + 1, o]] +
                             l[[o, o - 1, o]] + l[[o, o, o + 1]] +
                             l[[o, o, o - 1]] - 6. * w) / denominator;
                });
        }
        CentralDifference::FivePoint => {
            let denominator = 24. * config.grid.dn * config.grid.dn * config.mass;
            Zip::indexed(&mut work)
                .and(v)
                .and(w)
                .par_apply(|(i, j, k), work, &v, &w| {
                    // Offset indexes as we are already in a slice
                    let lx = i as isize + 2;
                    let ly = j as isize + 2;
                    let lz = k as isize + 2;
                    let o = 2;
                    // get a slice which gives us our matrix of central difference points
                    let l = phi.slice(s![lx - 2..lx + 3, ly - 2..ly + 3, lz - 2..lz + 3]);
                    // l can now be indexed with local offset `o` and modifiers
                    *work = v * w * w -
                            w *
                            (-l[[o + 2, o, o]] + 16. * l[[o + 1, o, o]] + 16. * l[[o - 1, o, o]] -
                             l[[o - 2, o, o]] - l[[o, o + 2, o]] +
                             16. * l[[o, o + 1, o]] +
                             16. * l[[o, o - 1, o]] -
                             l[[o, o - 2, o]] - l[[o, o, o + 2]] +
                             16. * l[[o, o, o + 1]] +
                             16. * l[[o, o, o - 1]] -
                             l[[o, o, o - 2]] - 90. * w) / denominator;
                });
        }
        CentralDifference::SevenPoint => {
            let denominator = 360. * config.grid.dn * config.grid.dn * config.mass;
            Zip::indexed(&mut work)
                .and(v)
                .and(w)
                .par_apply(|(i, j, k), work, &v, &w| {
                    // Offset indexes as we are already in a slice
                    let lx = i as isize + 3;
                    let ly = j as isize + 3;
                    let lz = k as isize + 3;
                    let o = 3;
                    // get a slice which gives us our matrix of central difference points
                    let l = phi.slice(s![lx - 3..lx + 4, ly - 3..ly + 4, lz - 3..lz + 4]);
                    // l can now be indexed with local offset `o` and modifiers
                    *work =
                        v * w * w -
                        w *
                        (2. * l[[o + 3, o, o]] - 27. * l[[o + 2, o, o]] + 270. * l[[o + 1, o, o]] +
                         270. * l[[o - 1, o, o]] - 27. * l[[o - 2, o, o]] +
                         2. * l[[o - 3, o, o]] + 2. * l[[o, o + 3, o]] -
                         27. * l[[o, o + 2, o]] + 270. * l[[o, o + 1, o]] +
                         270. * l[[o, o - 1, o]] - 27. * l[[o, o - 2, o]] +
                         2. * l[[o, o - 3, o]] + 2. * l[[o, o, o + 3]] -
                         27. * l[[o, o, o + 2]] + 270. * l[[o, o, o + 1]] +
                         270. * l[[o, o, o - 1]] - 27. * l[[o, o, o - 2]] +
                         2. * l[[o, o, o - 3]] - 1470. * w) / denominator;
                });
        }
    }
    // Sum result for total energy.
    work.scalar_sum()
}

/// Normalisation of the wavefunction
fn normalise_wavefunction(w: &mut Array3<f64>, norm2: f64) {
    let norm = norm2.sqrt();
    w.par_map_inplace(|el| *el /= norm);
}

/// Uses Gram Schmidt orthogonalisation to identify the next excited state's wavefunction, even if it's degenerate
fn orthogonalise_wavefunction(wnum: u8, w: &mut Array3<f64>, w_store: &[Array3<f64>]) {
    for lower in w_store.iter().take(wnum as usize) {
        let overlap = (lower * &w.view()).scalar_sum(); //TODO: par this multiplication if possible. A temp work array and par_applied zip is slower, even with an unassigned array
        Zip::from(w.view_mut())
            .and(lower)
            .par_apply(|w, &lower| *w -= lower * overlap);
    }
}

/// Shortcut to getting a slice of the workable area of the current array.
/// In other words, the finite element only cells are removed
///
/// # Arguments
///
/// * `arr` - A reference to the array which requires slicing.
/// * `ext` - Extent of central difference limits. From `config.central_difference.ext()`.
///
/// # Returns
///
/// An array view containing only the workable area of the array.
pub fn get_work_area(arr: &Array3<f64>, ext: usize) -> ArrayView3<f64> {
    let dims = arr.dim();
    let exti = ext as isize;
    arr.slice(s![exti..dims.0 as isize - exti,
                 exti..dims.1 as isize - exti,
                 exti..dims.2 as isize - exti])
}

/// Shortcut to getting a mutable slice of the workable area of the current array.
/// In other words, the finite element only cells are removed
///
/// # Arguments
///
/// * `arr` - A mutable reference to the array which requires slicing.
/// * `ext` - Extent of central difference limits. From `config.central_difference.ext()`.
///
/// # Returns
///
/// A mutable arrav view containing only the workable area of the array.
pub fn get_mut_work_area(arr: &mut Array3<f64>, ext: usize) -> ArrayViewMut3<f64> {
    let dims = arr.dim();
    let exti = ext as isize;
    arr.slice_mut(s![exti..dims.0 as isize - exti,
                     exti..dims.1 as isize - exti,
                     exti..dims.2 as isize - exti])
}

/// Evolves the solution a number of `steps`
fn evolve(wnum: u8,
          config: &Config,
          potentials: &Potentials,
          phi: &mut Array3<f64>,
          w_store: &[Array3<f64>]) {
    //without mpi, this is just update interior (which is really updaterule if we dont need W)
    let bb = config.central_difference.bb();
    let ext = config.central_difference.ext();
    let mut work_dims = phi.dim();
    work_dims.0 -= bb;
    work_dims.1 -= bb;
    work_dims.2 -= bb;
    let mut steps = 0;
    loop {

        let mut work = Array3::<f64>::zeros(work_dims);
        {
            let w = get_work_area(phi, ext);
            let pa = get_work_area(&potentials.a, ext);
            let pb = get_work_area(&potentials.b, ext);


            //TODO: We don't have any complex conjugation here.
            match config.central_difference {
                CentralDifference::ThreePoint => {
                    let denominator = 2. * config.grid.dn * config.grid.dn * config.mass;
                    Zip::indexed(&mut work)
                        .and(pa)
                        .and(pb)
                        .and(w)
                        .par_apply(|(i, j, k), work, &pa, &pb, &w| {
                            // Offset indexes as we are already in a slice
                            let lx = i as isize + 1;
                            let ly = j as isize + 1;
                            let lz = k as isize + 1;
                            let o = 1;
                            // get a slice which gives us our matrix of central difference points
                            let l = phi.slice(s![lx - 1..lx + 2, ly - 1..ly + 2, lz - 1..lz + 2]);
                            // l can now be indexed with local offset `o` and modifiers
                            *work = w * pa +
                                    pb * config.grid.dt *
                                    (l[[o + 1, o, o]] + l[[o - 1, o, o]] + l[[o, o + 1, o]] +
                                     l[[o, o - 1, o]] +
                                     l[[o, o, o + 1]] +
                                     l[[o, o, o - 1]] -
                                     6. * w) / denominator;
                        });
                }
                CentralDifference::FivePoint => {
                    let denominator = 24. * config.grid.dn * config.grid.dn * config.mass;
                    Zip::indexed(&mut work)
                        .and(pa)
                        .and(pb)
                        .and(w)
                        .par_apply(|(i, j, k), work, &pa, &pb, &w| {
                            // Offset indexes as we are already in a slice
                            let lx = i as isize + 2;
                            let ly = j as isize + 2;
                            let lz = k as isize + 2;
                            let o = 2;
                            // get a slice which gives us our matrix of central difference points
                            let l = phi.slice(s![lx - 2..lx + 3, ly - 2..ly + 3, lz - 2..lz + 3]);
                            // l can now be indexed with local offset `o` and modifiers
                            *work = w * pa +
                                    pb * config.grid.dt *
                                    (-l[[o + 2, o, o]] + 16. * l[[o + 1, o, o]] +
                                     16. * l[[o - 1, o, o]] -
                                     l[[o - 2, o, o]] -
                                     l[[o, o + 2, o]] +
                                     16. * l[[o, o + 1, o]] +
                                     16. * l[[o, o - 1, o]] -
                                     l[[o, o - 2, o]] -
                                     l[[o, o, o + 2]] +
                                     16. * l[[o, o, o + 1]] +
                                     16. * l[[o, o, o - 1]] -
                                     l[[o, o, o - 2]] -
                                     90. * w) / denominator;
                        });
                }
                CentralDifference::SevenPoint => {
                    let denominator = 360. * config.grid.dn * config.grid.dn * config.mass;
                    Zip::indexed(&mut work)
                        .and(pa)
                        .and(pb)
                        .and(w)
                        .par_apply(|(i, j, k), work, &pa, &pb, &w| {
                            // Offset indexes as we are already in a slice
                            let lx = i as isize + 3;
                            let ly = j as isize + 3;
                            let lz = k as isize + 3;
                            let o = 3;
                            // get a slice which gives us our matrix of central difference points
                            let l = phi.slice(s![lx - 3..lx + 4, ly - 3..ly + 4, lz - 3..lz + 4]);
                            // l can now be indexed with local offset `o` and modifiers
                            *work = w * pa +
                                    pb * config.grid.dt *
                                    (2. * l[[o + 3, o, o]] - 27. * l[[o + 2, o, o]] +
                                     270. * l[[o + 1, o, o]] +
                                     270. * l[[o - 1, o, o]] -
                                     27. * l[[o - 2, o, o]] +
                                     2. * l[[o - 3, o, o]] +
                                     2. * l[[o, o + 3, o]] -
                                     27. * l[[o, o + 2, o]] +
                                     270. * l[[o, o + 1, o]] +
                                     270. * l[[o, o - 1, o]] -
                                     27. * l[[o, o - 2, o]] +
                                     2. * l[[o, o - 3, o]] +
                                     2. * l[[o, o, o + 3]] -
                                     27. * l[[o, o, o + 2]] +
                                     270. * l[[o, o, o + 1]] +
                                     270. * l[[o, o, o - 1]] -
                                     27. * l[[o, o, o - 2]] +
                                     2. * l[[o, o, o - 3]] -
                                     1470. * w) / denominator;
                        });
                }
            }
        }
        {
            let mut w_fill = get_mut_work_area(phi, ext);
            Zip::from(&mut w_fill)
                .and(&work)
                .par_apply(|w_fill, &work| { *w_fill = work; });
        }
        if wnum > 0 {
            let norm2 = {
                let work = get_work_area(phi, ext);
                get_norm_squared(&work)
            };
            normalise_wavefunction(phi, norm2);
            orthogonalise_wavefunction(wnum, phi, w_store);
        }
        steps += 1;
        if steps >= config.output.screen_update {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn work_area() {
        let test = Array3::<f64>::zeros((5,8,7));
        let work = get_work_area(&test, 1);
        let dims = work.dim();
        assert_eq!(dims.0, 3);
        assert_eq!(dims.1, 6);
        assert_eq!(dims.2, 5);
    }
}

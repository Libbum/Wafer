use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array3, ArrayView3, ArrayViewMut3, Zip};
use ndarray_parallel::prelude::*;
use slog::Logger;
use std::f64::MAX;
use config;
use config::{Config, Grid, Index3, InitialCondition};
use potential;
use potential::Potentials;
use input;
use output;


#[derive(Debug)]
/// Holds all computed observables for the current wavefunction.
pub struct Observables {
    /// Normalised total energy.
    pub energy: f64,
    /// A squared normalisation. Squareroot occurs later when needed to apply a complete
    /// normalisation condition. This needs to be separate as we include other adjutsments
    /// from time to time.
    pub norm2: f64,
    /// The value of the potential at infinity. This is used to calculate the binding energy.
    pub v_infinity: f64,
    /// Coefficient of determination
    pub r2: f64,
}

/// Runs the calculation and holds long term (system time) wavefunction storage
pub fn run(config: &Config, log: &Logger) -> Result<(), input::Error> { //TODO: Generalise error here.

    let potentials = potential::load_arrays(config, log);

    let mut w_store: Vec<Array3<f64>> = Vec::new();
    if config.wavenum > 0 {
        //We require wavefunctions from disk, even if initial condition is not `FromFile`
        //The wavenum = 0 case is handled later
        input::load_wavefunctions(config, log, config.output.binary_files, &mut w_store)?;
    }

    info!(log, "Starting calculation");

    for wnum in config.wavenum..config.wavemax + 1 {
        solve(config, log, &potentials, wnum, &mut w_store);
    }
    Ok(())
}


/// Runs the actual computation once system is setup and ready.
fn solve(config: &Config,
         log: &Logger,
         potentials: &Potentials,
         wnum: u8,
         w_store: &mut Vec<Array3<f64>>) {

    // Initial conditions from config file if ground state,
    // but start from previously converged wfn if we're an excited state.
    let mut phi: Array3<f64> = if wnum > 0 {
        let num = &config.grid.size;
        let init_size: [usize; 3] = [(num.x + 6) as usize,
                                     (num.y + 6) as usize,
                                     (num.z + 6) as usize];
        // Try to load current wavefunction from disk.
        // If not, we start with the previously converged function
        if let Ok(wfn) = input::wavefunction(wnum, init_size, config.output.binary_files, log) {
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
        match config::set_initial_conditions(config, log) {
            Ok(wfn) => wfn,
            Err(err) => panic!("{}", err),
        }
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
                if let Err(err) = output::finalise_measurement(&observables,
                                                               wnum,
                                                               config.grid.size.x as f64,
                                                               &config.project_name,
                                                               config.output.binary_files) {
                    panic!("Error with ouput: {}", err);
                }
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
        let work = get_work_area(&phi);
        match output::wavefunction(&work,
                                   wnum,
                                   converged,
                                   &config.project_name,
                                   config.output.binary_files) {
            Ok(_) => {}
            Err(err) => crit!(log, "Could not write wavefunction to disk: {}", err),
        }
    }

    if converged {
        info!(log, "Caluculation Converged");
        w_store.push(phi); //Save state
    } else {
        crit!(log, "Caluculation stopped due to maximum step limit.");
        panic!("Maximum step limit reached. Cannot continue, but restart files have been output.");
    }
}

/// Estamates completion time for the convergence of the current wavefunction.
///
/// # Returns
///
/// An estimate of the numer of `screen_update` cycles to go until convergece.
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
    let work = get_work_area(phi);
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
    Zip::indexed(&mut work)
        .and(w)
        .par_apply(|(i, j, k), work, &w| {
                       let idx = Index3 { x: i, y: j, z: k };
                       let potsub = match potential::potential_sub(config, &idx) {
                           Ok(p) => p,
                           Err(err) => panic!("Error: {}", err),
                       };
                       *work = w * w * potsub;
                   });
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
//TODO: We can probably drop the config requirement and replace it with a grid modifier of dn*mass
fn wfnc_energy(config: &Config, potentials: &Potentials, phi: &Array3<f64>) -> f64 {

    let w = get_work_area(phi);
    let v = get_work_area(&potentials.v);

    // Simplify what we can here.
    let denominator = 360. * config.grid.dn * config.grid.dn * config.mass;

    let mut work = Array3::<f64>::zeros(w.dim());
    //NOTE: TODO: We don't have any complex conjugation here.
    // Complete matrix multiplication step using 7 point central differenc
    // TODO: Option for 3 or 5 point caclulation
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
            *work = v * w * w -
                    w *
                    (2. * l[[o + 3, o, o]] - 27. * l[[o + 2, o, o]] + 270. * l[[o + 1, o, o]] +
                     270. * l[[o - 1, o, o]] -
                     27. * l[[o - 2, o, o]] + 2. * l[[o - 3, o, o]] +
                     2. * l[[o, o + 3, o]] - 27. * l[[o, o + 2, o]] +
                     270. * l[[o, o + 1, o]] +
                     270. * l[[o, o - 1, o]] -
                     27. * l[[o, o - 2, o]] + 2. * l[[o, o - 3, o]] +
                     2. * l[[o, o, o + 3]] - 27. * l[[o, o, o + 2]] +
                     270. * l[[o, o, o + 1]] +
                     270. * l[[o, o, o - 1]] -
                     27. * l[[o, o, o - 2]] + 2. * l[[o, o, o - 3]] -
                     1470. * w) / denominator;
        });
    // Sum result for total energy.
    work.scalar_sum()
}

/// Normalisation of the wavefunction
fn normalise_wavefunction(w: &mut Array3<f64>, norm2: f64) {
    //TODO: This can be moved directly into the calculation for now. It's only here due to normalisationCollect
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
///
/// # Returns
///
/// An arrav view containing only the workable area of the array.
pub fn get_work_area(arr: &Array3<f64>) -> ArrayView3<f64> {
    // TODO: This is hardcoded to a 7 point stencil
    let dims = arr.dim();
    arr.slice(s![3..(dims.0 as isize) - 3,
                 3..(dims.1 as isize) - 3,
                 3..(dims.2 as isize) - 3])
}

/// Shortcut to getting a mutable slice of the workable area of the current array.
/// In other words, the finite element only cells are removed
///
/// # Arguments
///
/// * `arr` - A mutable reference to the array which requires slicing.
///
/// # Returns
///
/// A mutable arrav view containing only the workable area of the array.
pub fn get_mut_work_area(arr: &mut Array3<f64>) -> ArrayViewMut3<f64> {
    // TODO: This is hardcoded to a 7 point stencil
    let dims = arr.dim();
    arr.slice_mut(s![3..(dims.0 as isize) - 3,
                     3..(dims.1 as isize) - 3,
                     3..(dims.2 as isize) - 3])
}

/// Evolves the solution a number of `steps`
fn evolve(wnum: u8,
          config: &Config,
          potentials: &Potentials,
          phi: &mut Array3<f64>,
          w_store: &[Array3<f64>]) {
    //without mpi, this is just update interior (which is really updaterule if we dont need W)

    let mut work_dims = phi.dim();
    work_dims.0 -= 6;
    work_dims.1 -= 6;
    work_dims.2 -= 6;
    let mut steps = 0;
    loop {

        let mut work = Array3::<f64>::zeros(work_dims);
        {
            let w = get_work_area(phi);
            let pa = get_work_area(&potentials.a);
            let pb = get_work_area(&potentials.b);

            let denominator = 360. * config.grid.dn * config.grid.dn * config.mass;

            //NOTE: TODO: We don't have any complex conjugation here.
            // Complete matrix multiplication step using 7 point central difference
            // TODO: Option for 3 or 5 point caclulation
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
                    *work =
                        w * pa +
                        pb * config.grid.dt *
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
        {
            let mut w_fill = get_mut_work_area(phi);
            Zip::from(&mut w_fill)
                .and(&work)
                .par_apply(|w_fill, &work| { *w_fill = work; });
        }
        if wnum > 0 {
            let norm2 = {
                let work = get_work_area(phi);
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

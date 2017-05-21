use ndarray::Array3;
use ordinal::Ordinal;
use rayon;
use std::fs::{create_dir, File};
use std::io::Error;
use std::io::prelude::*;
use std::path::Path;
use term_size;
use ansi_term::Colour::Blue;

use grid;
/// Simply prints the Wafer banner with current commit info and thread count.
pub fn print_banner(sha: &str) {
    println!("                    {}", Blue.paint("___"));
    println!("   __      ____ _  {}__ _ __", Blue.paint("/ __\\"));
    println!("   \\ \\ /\\ / / _` |{} / _ \\ '__|", Blue.paint("/ /"));
    println!("    \\ V  V / (_| {}|  __/ |    Current build SHA1: {}",
             Blue.paint("/ _\\"),
             sha);
    println!("     \\_/\\_/ \\__,{}   \\___|_|    Parallel tasks running on {} threads.",
             Blue.paint("/ /"),
             rayon::current_num_threads());
    println!("              {}", Blue.paint("\\__/"));
    println!("");
}

/// Outputs the current potential to disk in a plain, csv format
///
/// # Arguments
/// *`v` - The potential to output
///
/// # Returns
/// * A result type with a `std::io::Error`. The result value is a true bool
/// as we really only want to error check the result.
pub fn potential_plain(v: &Array3<f64>) -> Result<bool, Error> {
    let mut buffer = File::create("output/potential.csv")?;
    let dims = v.dim();
    let work = v.slice(s![3..(dims.0 as isize) - 3,
                          3..(dims.1 as isize) - 3,
                          3..(dims.2 as isize) - 3]);
    for ((i, j, k), el) in work.indexed_iter() {
        let output = format!("{}, {}, {}, {:e}\n", i, j, k, el);
        buffer.write_all(output.as_bytes())?;
    }
    Ok(true)
}

/// Outputs a wavefunction to disk in a plain, csv format
///
/// # Arguments
/// * `phi` - The wavefunction to output
/// * `num` - The wavefunction's excited state value for file naming.
/// * `converged` - a bool advising the state of the wavefunction. If false, the filename
/// will have `_partial` appended to it to indicate a restart is required.
///
/// # Returns
/// * A result type with a `std::io::Error`. The result value is a true bool
/// as we really only want to error check the result.
pub fn wavefunction_plain(phi: &Array3<f64>, num: u8, converged: bool) -> Result<bool, Error> {
    let filename = format!("output/wavefunction_{}{}.csv",
                           num,
                           if converged { "" } else { "_partial" });
    let mut buffer = File::create(filename)?;
    let dims = phi.dim();
    let work = phi.slice(s![3..(dims.0 as isize) - 3,
                            3..(dims.1 as isize) - 3,
                            3..(dims.2 as isize) - 3]);
    for ((i, j, k), el) in work.indexed_iter() {
        let output = format!("{}, {}, {}, {:e}\n", i, j, k, el);
        buffer.write_all(output.as_bytes())?;
    }
    Ok(true)
}

/// Pretty prints a header for the subsequent observable data
pub fn print_observable_header(wnum: u8) {
    let width = get_term_size();
    let spacer = (width - 69) / 2;
    let col2 = 37; //Energy+rrms+1
    println!("");
    match wnum {
        0 => {
            println!("{:═^lspace$}╤{:═^twidth$}╤{:═^w$}╤{:═^width$}╤{:═^rspace$}",
                     "",
                     "",
                     " Ground state caclulation ",
                     "",
                     "",
                     twidth = 12,
                     width = 16,
                     w = col2,
                     lspace = spacer,
                     rspace = if 2 * spacer + 69 < width {
                         spacer + 1
                     } else {
                         spacer
                     });
        }
        _ => {
            println!("{:═^lspace$}╤{:═^twidth$}╤{:═^w$}╤{:═^width$}╤{:═^rspace$}",
                     "",
                     "",
                     format!(" {} excited state caclulation ", Ordinal::from(wnum)),
                     "",
                     "",
                     twidth = 12,
                     width = 16,
                     w = col2,
                     lspace = spacer,
                     rspace = if 2 * spacer + 69 < width {
                         spacer + 1
                     } else {
                         spacer
                     });
        }
    }
    println!("{:^space$}│{:^twidth$}│{:^ewidth$}│{:^width$}│{:^width$}│",
             "",
             "Time",
             "Energy",
             "rᵣₘₛ",
             "Difference",
             twidth = 12,
             ewidth = 20,
             width = 16,
             space = spacer);
    println!("{:─^lspace$}┼{:─^twidth$}┼{:─^ewidth$}┼{:─^width$}┼{:─^width$}┼{:─^rspace$}",
             "",
             "",
             "",
             "",
             "",
             "",
             twidth = 12,
             ewidth = 20,
             width = 16,
             lspace = spacer,
             rspace = if 2 * spacer + 69 < width {
                 spacer + 1
             } else {
                 spacer
             });
}

/// Pretty prints measurements at current step to screen
pub fn measurements(tau: f64, diff: f64, observables: &grid::Observables) -> String {
    //TODO: This is going to be called every output. Maybe generate a lazy_static value?
    let width = get_term_size();
    let spacer = (width - 69) / 2;
    if tau > 0.0 {
        format!("{:^space$}│{:>11.3} │{:>19.10e} │{:15.5} │{:15.5e} │",
                 "",
                 tau,
                 observables.energy / observables.norm2,
                 (observables.r2 / observables.norm2).sqrt(),
                 diff,
                 space = spacer)
    } else {
        format!("{:^space$}│{:>11.3} │{:>19.10e} │{:15.5} │{:>15} │",
                 "",
                 tau,
                 observables.energy / observables.norm2,
                 (observables.r2 / observables.norm2).sqrt(),
                 "--   ",
                 space = spacer)

    }
}

/// Pretty print final summary
pub fn summary(observables: &grid::Observables, wnum: u8, numx: f64) {
    let width = get_term_size();
    let spacer = (width - 69) / 2;
    let energy = observables.energy / observables.norm2;
    let binding = observables.energy - observables.v_infinity / observables.norm2;
    let r_norm = (observables.r2 / observables.norm2).sqrt();
    println!("{:═^lspace$}╧{:═^twidth$}╧{:═^ewidth$}╧{:═^width$}╧{:═^width$}╧{:═^rspace$}",
             "",
             "",
             "",
             "",
             "",
             "",
             twidth = 12,
             ewidth = 20,
             width = 16,
             lspace = spacer,
             rspace = if 2 * spacer + 69 < width {
                 spacer + 1
             } else {
                 spacer
             });
    match wnum {
        0 => {
            println!("══▶ Ground state energy = {}", energy);
            println!("══▶ Ground state binding energy = {}", binding);
        }
        _ => {
            let state = Ordinal::from(wnum);
            println!("══▶ {} excited state energy = {}", state, energy);
            println!("══▶ {} excited state binding energy = {}",
                     state,
                     binding);
        }
    }
    println!("══▶ rᵣₘₛ = {}", r_norm);
    println!("══▶ L/rᵣₘₛ = {}", numx / r_norm);
    println!("");
}

/// Checks that the folder `output` exists. If not, creates it.
///
/// # Panics
/// * If directory can not be created. Gives `std::io::Error`.
pub fn check_output_dir() {
    if !Path::new("./output").exists() {
        let result = create_dir("./output");
        match result {
            Ok(_) => {}
            Err(err) => panic!("Cannot create output directory: {}", err),
        }
    }
}

/// Uses `term_size` to pull in the terminal width and from there sets the output
/// pretty printing value to an appropriate value (between 70-100).
pub fn get_term_size() -> usize {
    let mut term_width = 100;
    if let Some((width, _)) = term_size::dimensions() {
        if width <= 70 {
            term_width = 70;
        } else if width < term_width {
            term_width = width;
        }
    }
    term_width
}

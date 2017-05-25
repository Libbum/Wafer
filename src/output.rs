use chrono::Local;
use ndarray::Array3;
use ordinal::Ordinal;
use rayon;
use serde::Serialize;
use serde_json;
use rmps::Serializer;
use std::fs::{copy, create_dir_all, File};
use std::io::Error;
use std::io::prelude::*;
use term_size;
use ansi_term::Colour::Blue;

use grid;

lazy_static! {
    /// Date & time at which the simulation was started. Used as a unique identifier for
    /// the output directory of a run.
    static ref PROJDATE: String = Local::now().format("%Y-%m-%d_%H:%M:%S").to_string();
}

#[derive(Debug, PartialEq, Deserialize, Serialize)]
/// Structured output of observable values
struct ObservablesOutput {
    /// Excited state number.
    state: u8,
    /// Total energy.
    energy: f64,
    /// Binding energy.
    binding_energy: f64,
    /// Coefficient of determination
    r: f64,
    /// Grid size / Coefficient of determination
    l_r: f64,
}

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
pub fn potential_plain(v: &Array3<f64>, project: &str) -> Result<bool, Error> {
    let mut buffer = File::create(get_project_dir(project) + "/potential.csv")?;
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
pub fn wavefunction_plain(phi: &Array3<f64>,
                          num: u8,
                          converged: bool,
                          project: &str)
                          -> Result<bool, Error> {
    let filename = format!("{}/wavefunction_{}{}.csv",
                           get_project_dir(project),
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
    if let 0 = wnum {
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
    } else {
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
    println!("{:^space$}│{:^twidth$}│{:^ewidth$}│{:^width$}│{:^width$}│",
             "",
             "Time (τ)",
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
pub fn summary(observables: &grid::Observables, wnum: u8, numx: f64, project: &str) {
    let width = get_term_size();
    let spacer = (width - 69) / 2;
    let r_norm = (observables.r2 / observables.norm2).sqrt();
    let output = ObservablesOutput {
        state: wnum,
        energy: observables.energy / observables.norm2,
        binding_energy: observables.energy - observables.v_infinity / observables.norm2,
        r: r_norm,
        l_r: numx / r_norm,
    };

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
    if let 0 = wnum {
        println!("══▶ Ground state energy = {}", output.energy);
        println!("══▶ Ground state binding energy = {}", output.binding_energy);
    } else {
        let state = Ordinal::from(wnum);
        println!("══▶ {} excited state energy = {}", state, output.energy);
        println!("══▶ {} excited state binding energy = {}",
                 state,
                 output.binding_energy);
    }
    println!("══▶ rᵣₘₛ = {}", output.r);
    println!("══▶ L/rᵣₘₛ = {}", output.l_r);
    println!("");

    observables_binary(&output, project); //TODO: These are just here for testing, we need to treat this better.
    observables_plain(&output, project);
}

/// Saves the observables to a messagepack binary file.
fn observables_binary(observables: &ObservablesOutput, project: &str) {
    //TODO: I think this would be nice if it was acutally one file rather than many. So we appended to it somehow.
    let filename = format!("{}/observables_{}.mpk", get_project_dir(project), observables.state);
    let mut output = Vec::new();
    observables.serialize(&mut Serializer::new(&mut output)).unwrap(); //TODO: Actual error handling.
    let mut buffer = File::create(filename).expect("Cannot create observable output file");
    buffer.write_all(&output).expect("Unable to write data to observable file");
}

/// Saves the observables to a plain json file.
fn observables_plain(observables: &ObservablesOutput, project: &str) {
    //TODO: I think this would be nice if it was acutally one file rather than many. So we appended to it somehow.
    let filename = format!("{}/observables_{}.json", get_project_dir(project), observables.state);
    let buffer = File::create(filename).expect("Cannot create observable output file");
    serde_json::to_writer_pretty(buffer, observables).expect("Unable to write data to observable file");
    //buffer.write_all(&output).expect("Unable to write data to observable file");
}

/// Generates a unique folder inside an `output` directory for the current simulation.
///
/// # Panics
/// * If directory can not be created. Gives `std::io::Error`.
pub fn check_output_dir(project: &str) {
    let proj_dir = create_dir_all(get_project_dir(project));
    match proj_dir {
        Ok(_) => {}
        Err(err) => panic!("Cannot create project directory: {}", err),
    }
}

/// Each simulation has a unique folder output so as not to overwrite other instances.
/// This function gets the name of the current folder.
///
/// # Arguments
///
/// * `project` - The name of the currently active project as set in the configuration file
///
/// # Retuns
///
/// A string containting the location of the output folder, comprised of a sanitized
/// project name, followed by the start date/time of the simulation.
pub fn get_project_dir(project: &str) -> String {
    format!("./output/{}_{}", sanitize_string(project), &**PROJDATE)
}

/// Copies the current configuration file to the project folder
pub fn copy_config(project: &str) {
    let result = copy("./wafer.cfg", get_project_dir(project) + "/wafer.cfg");
    match result {
        Ok(_) => {}
        Err(err) => {
            panic!("Error copying configuration file to project directory: {}",
                   err)
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

/// Sanitizes strings such that they will create safe filenames.
/// For now, only used with the `project_name` variable in the configuration.
fn sanitize_string(component: &str) -> String {
    let mut buffer = String::with_capacity(component.len());
    for (i, c) in component.chars().enumerate() {
        let is_lower = 'a' <= c && c <= 'z';
        let is_upper = 'A' <= c && c <= 'Z';
        let is_letter = is_upper || is_lower;
        let is_number = '0' <= c && c <= '9';
        let is_space = c == ' ';
        let is_hyphen = c == '-';
        let is_underscore = c == '_';
        let is_period = c == '.' && i != 0; // Disallow accidentally hidden folders
        let is_valid = is_letter || is_number || is_hyphen || is_underscore || is_period;
        if is_valid {
            buffer.push(c);
        } else if is_space {
            buffer.push('_'); //Convert spaces to underscores.
        } else {
            buffer.push_str(&format!(",{},", c as u32));
        }
    }
    buffer
}

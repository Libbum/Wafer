use chrono::Local;
use csv;
use ndarray::{Array3, ArrayView3};
use ordinal::Ordinal;
use rayon;
use serde::Serialize;
use serde_json;
use rmps;
use std::error::Error;
use std::fmt;
use std::fs::{copy, create_dir_all, File};
use std::io;
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

#[derive(Debug,Serialize)]
/// A simple struct to parse data to a plain csv file
struct PlainRecord {
    /// Index in *x*
    i: usize,
    /// Index in *y*
    j: usize,
    /// Index in *z*
    k: usize,
    /// Data at this position
    data: f64,
}

/// Error type for handling file output. Effectively a wapper around multiple error types we encounter.
#[derive(Debug)]
pub enum OutputError {
    /// From disk issues.
    Io(io::Error),
    /// From `serde_json`.
    EncodePlain(serde_json::Error),
    /// From `rmp_serde`.
    EncodeBinary(rmps::encode::Error),
    /// From `csv`.
    Csv(csv::Error),
}

impl fmt::Display for OutputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            OutputError::Io(ref err) => err.fmt(f),
            OutputError::EncodePlain(ref err) => err.fmt(f),
            OutputError::EncodeBinary(ref err) => err.fmt(f),
            OutputError::Csv(ref err) => err.fmt(f),
        }
    }
}

impl Error for OutputError {
    fn description(&self) -> &str {
        match *self {
            OutputError::Io(ref err) => err.description(),
            OutputError::EncodePlain(ref err) => err.description(),
            OutputError::EncodeBinary(ref err) => err.description(),
            OutputError::Csv(ref err) => err.description(),
        }
    }

    fn cause(&self) -> Option<&Error> {
        match *self {
            OutputError::Io(ref err) => Some(err),
            OutputError::EncodePlain(ref err) => Some(err),
            OutputError::EncodeBinary(ref err) => Some(err),
            OutputError::Csv(ref err) => Some(err),
        }
    }
}

impl From<io::Error> for OutputError {
    fn from(err: io::Error) -> OutputError {
        OutputError::Io(err)
    }
}

impl From<serde_json::Error> for OutputError {
    fn from(err: serde_json::Error) -> OutputError {
        OutputError::EncodePlain(err)
    }
}

impl From<rmps::encode::Error> for OutputError {
    fn from(err: rmps::encode::Error) -> OutputError {
        OutputError::EncodeBinary(err)
    }
}

impl From<csv::Error> for OutputError {
    fn from(err: csv::Error) -> OutputError {
        OutputError::Csv(err)
    }
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
pub fn potential_plain(v: &Array3<f64>, project: &str) -> Result<(), OutputError> {
    let mut buffer = File::create(get_project_dir(project) + "/potential.csv")?;
    let work = grid::get_work_area(v);
    for ((i, j, k), el) in work.indexed_iter() {
        let output = format!("{}, {}, {}, {:e}\n", i, j, k, el);
        buffer.write_all(output.as_bytes())?;
    }
    Ok(())
}

/// Saves a wavefunction to disk, and controlls what format (plain text or binary)
/// the data should be handled as.
///
/// # Arguments
/// * `phi` - The wavefunction to output. This should be a view called from `grid::get_work_area()`.
/// * `num` - The wavefunction's excited state value for file naming.
/// * `converged` - A bool advising the state of the wavefunction. If false, the filename
/// will have `_partial` appended to it to indicate a restart is required.
/// * `project` - The project name (for directory to save to).
/// * `binary` - A bool to ascertain if output should be in binary or plain text format.
///
/// # Returns
/// * A result type with a `std::io::Error`. The result value is a true bool
/// as we really only want to error check the result.
pub fn wavefunction(phi: &ArrayView3<f64>,
                    num: u8,
                    converged: bool,
                    project: &str,
                    binary: bool)
                    -> Result<(), OutputError> {
    let filename = format!("{}/wavefunction_{}{}.{}",
                           get_project_dir(project),
                           num,
                           if converged { "" } else { "_partial" },
                           if binary { "mpk" } else { "csv" });
    if binary {
        wavefunction_binary(phi, &filename)
    } else {
        wavefunction_plain(phi, &filename)
    }
}

/// Outputs a wavefunction to disk in the messagepack binary format.
///
/// # Arguments
/// * `phi` - The wavefunction to output. This should be a view called from `grid::get_work_area()`.
/// * `filename` - A string indiciting the location of the output.
///
/// # Returns
/// * A result type with a `std::io::Error`. The result value is a true bool
/// as we really only want to error check the result.
fn wavefunction_binary(phi: &ArrayView3<f64>, filename: &str) -> Result<(), OutputError> {
    //NOTE: The code below should work, but we must wait for ndarray to have serde 1.0 compatability.
    //For now, we just output to plain instead.
    wavefunction_plain(phi, filename)
    //let mut output = Vec::new();
    //phi.serialize(&mut rmps::Serializer::new(&mut output))?;
    //let mut buffer = File::create(filename)?;
    //buffer.write_all(&output)?;
    //Ok(())
}

/// Outputs a wavefunction to disk in a plain, csv format.
///
/// # Arguments
/// * `phi` - The wavefunction to output. This should be a view called from `grid::get_work_area()`.
/// * `filename` - A string indiciting the location of the output.
///
/// # Returns
/// * A result type with a `std::io::Error`. The result value is a true bool
/// as we really only want to error check the result.
fn wavefunction_plain(phi: &ArrayView3<f64>, filename: &str) -> Result<(), OutputError> {
    let mut buffer = csv::Writer::from_path(filename)?;
    for ((i, j, k), data) in phi.indexed_iter() {
        buffer
            .serialize(PlainRecord {
                           i: i,
                           j: j,
                           k: k,
                           data: *data,
                       })?;
    }
    buffer.flush()?;
    Ok(())
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
pub fn print_measurements(tau: f64, diff: f64, observables: &grid::Observables) -> String {
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

/// Sets up the final mesasurements for each wavefunction, printing them to screen and saving
/// them to disk.
///
/// #Arguments
///
/// * `observables` - calculated, un-normalised values.
/// * `wnum` - current wave number.
/// * `numx` - the width of the calculation box.
/// * `project` - current project name (for file output).
/// * `binary` - bool setting binary or plain text output.
pub fn finalise_measurement(observables: &grid::Observables,
                            wnum: u8,
                            numx: f64,
                            project: &str,
                            binary: bool)
                            -> Result<(), OutputError> {
    let r_norm = (observables.r2 / observables.norm2).sqrt();
    let output = ObservablesOutput {
        state: wnum,
        energy: observables.energy / observables.norm2,
        binding_energy: observables.energy - observables.v_infinity / observables.norm2,
        r: r_norm,
        l_r: numx / r_norm,
    };

    print_summary(&output);

    if binary {
        observables_binary(&output, project)
    } else {
        observables_plain(&output, project)
    }
}

/// Pretty print final summary
fn print_summary(output: &ObservablesOutput) {
    let width = get_term_size();
    let spacer = (width - 69) / 2;

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
    if let 0 = output.state {
        println!("══▶ Ground state energy = {}", output.energy);
        println!("══▶ Ground state binding energy = {}",
                 output.binding_energy);
    } else {
        let state = Ordinal::from(output.state);
        println!("══▶ {} excited state energy = {}",
                 state,
                 output.energy);
        println!("══▶ {} excited state binding energy = {}",
                 state,
                 output.binding_energy);
    }
    println!("══▶ rᵣₘₛ = {}", output.r);
    println!("══▶ L/rᵣₘₛ = {}", output.l_r);
    println!("");

}

/// Saves the observables to a messagepack binary file.
fn observables_binary(observables: &ObservablesOutput, project: &str) -> Result<(), OutputError> {
    let filename = format!("{}/observables_{}.mpk",
                           get_project_dir(project),
                           observables.state);
    let mut output = Vec::new();
    observables
        .serialize(&mut rmps::Serializer::new(&mut output))?;
    let mut buffer = File::create(filename)?;
    buffer.write_all(&output)?;
    Ok(())
}

/// Saves the observables to a plain json file.
fn observables_plain(observables: &ObservablesOutput, project: &str) -> Result<(), OutputError> {
    let filename = format!("{}/observables_{}.json",
                           get_project_dir(project),
                           observables.state);
    let buffer = File::create(filename)?;
    serde_json::to_writer_pretty(buffer, observables)?;
    Ok(())
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

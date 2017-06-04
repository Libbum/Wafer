use chrono::Local;
use csv;
use ndarray::ArrayView3;
use ordinal::Ordinal;
use rayon;
use serde::Serialize;
use serde_json;
use rmps;
use std::fs::{copy, create_dir_all, File, remove_file};
use std::io::prelude::*;
use term_size;
use ansi_term::Colour::Blue;

use grid;
use errors::*;

lazy_static! {
    /// Date & time at which the simulation was started. Used as a unique identifier for
    /// the output directory of a run.
    static ref PROJDATE: String = Local::now().format("%Y-%m-%d_%H:%M:%S").to_string();
    /// Width of program output to screen.
    pub static ref TERMWIDTH: usize = get_term_size();
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

/// Handles the saving of potential data to disk.
///
/// # Arguments
/// *`v` - The potential to output.
/// * `project` - The project name (for directory to save to).
/// * `binary` - Bool to identify what type of file to be output (plain text or messagepack).
pub fn potential(v: &ArrayView3<f64>, project: &str, binary: bool) -> Result<()> {
    let filename = format!("{}/potential.{}",
                           get_project_dir(project),
                           if binary { "mpk" } else { "csv" });

    if binary {
        potential_binary(v, &filename)
    } else {
        potential_plain(v, &filename)
    }
}

/// Outputs the current potential to disk in a plain, csv format
///
/// # Arguments
/// *`v` - The potential to output
/// * `filename` - file / directory to save to.
fn potential_plain(v: &ArrayView3<f64>, filename: &str) -> Result<()> {
    let mut buffer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_path(filename)
        .chain_err(|| ErrorKind::CreateFile(filename.to_string()))?;
    for ((i, j, k), data) in v.indexed_iter() {
        buffer
            .serialize(PlainRecord {
                           i: i,
                           j: j,
                           k: k,
                           data: *data,
                       })
            .chain_err(|| ErrorKind::Serialize)?;
    }
    buffer.flush().chain_err(|| ErrorKind::Flush)?;
    Ok(())
}

/// Outputs the current potential to disk in the messagepack binary format
///
/// # Arguments
/// *`v` - The potential to output
/// * `filename` - file / directory to save to.
fn potential_binary(v: &ArrayView3<f64>, filename: &str) -> Result<()> {
    //NOTE: The code below should work, but we must wait for ndarray to have serde 1.0 compatability.
    //For now, we just output to plain instead.
    potential_plain(v, filename)
    //let mut output = Vec::new();
    //v.serialize(&mut rmps::Serializer::new(&mut output))?;
    //let mut buffer = File::create(filename)?;
    //buffer.write_all(&output)?;
    //Ok(())
}

/// Saves a wavefunction to disk, and controls what format (plain text or binary)
/// the data should be handled as.
///
/// # Arguments
/// * `phi` - The wavefunction to output. This should be a view called from `grid::get_work_area()`.
/// * `num` - The wavefunction's excited state value for file naming.
/// * `converged` - A bool advising the state of the wavefunction. If false, the filename
/// will have `_partial` appended to it to indicate a restart is required.
/// * `project` - The project name (for directory to save to).
/// * `binary` - A bool to ascertain if output should be in binary or plain text format.
pub fn wavefunction(phi: &ArrayView3<f64>,
                    num: u8,
                    converged: bool,
                    project: &str,
                    binary: bool)
                    -> Result<()> {
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
///
/// * `phi` - The wavefunction to output. This should be a view called from `grid::get_work_area()`.
/// * `filename` - A string indicting the location of the output.
fn wavefunction_binary(phi: &ArrayView3<f64>, filename: &str) -> Result<()> {
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
///
/// * `phi` - The wavefunction to output. This should be a view called from `grid::get_work_area()`.
/// * `filename` - A string indicting the location of the output.
fn wavefunction_plain(phi: &ArrayView3<f64>, filename: &str) -> Result<()> {
    let mut buffer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_path(filename)
        .chain_err(|| ErrorKind::CreateFile(filename.to_string()))?;
    for ((i, j, k), data) in phi.indexed_iter() {
        buffer
            .serialize(PlainRecord {
                           i: i,
                           j: j,
                           k: k,
                           data: *data,
                       })
            .chain_err(|| ErrorKind::Serialize)?;
    }
    buffer.flush().chain_err(|| ErrorKind::Flush)?;
    Ok(())
}

/// Removes a temporary `_partial` file from the current output directory.
/// Should only be called if a converged file is written.
///
/// # Arguments
///
/// * `wnum` - The current wavenumber.
/// * `project` - Project name of the current simulation.
/// * `binary` - Boolean telling us if we've been outputing plain or binary temp files.
pub fn remove_partial(wnum: u8, project: &str, binary: bool) -> Result<()> {
    let filename = format!("{}/wavefunction_{}_partial.{}",
                           get_project_dir(project),
                           wnum,
                           if binary { "mpk" } else { "csv" });
    remove_file(&filename)
        .chain_err(|| ErrorKind::DeletePartial(wnum))?;
    Ok(())
}

/// Pretty prints a header for the subsequent observable data
pub fn print_observable_header(wnum: u8) {
    let width = *TERMWIDTH;
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
    let width = *TERMWIDTH;
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

/// Sets up the final measurements for each wavefunction, printing them to screen and saving
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
                            -> Result<()> {
    let r_norm = (observables.r2 / observables.norm2).sqrt();
    let output = ObservablesOutput {
        state: wnum,
        energy: observables.energy / observables.norm2,
        binding_energy: (observables.energy - observables.v_infinity) / observables.norm2,
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
    let width = *TERMWIDTH;
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
fn observables_binary(observables: &ObservablesOutput, project: &str) -> Result<()> {
    let filename = format!("{}/observables_{}.mpk",
                           get_project_dir(project),
                           observables.state);
    let mut output = Vec::new();
    observables
        .serialize(&mut rmps::Serializer::new(&mut output))
        .chain_err(|| ErrorKind::Serialize)?;
    let mut buffer = File::create(&filename)
        .chain_err(|| ErrorKind::CreateFile(filename))?;
    buffer
        .write_all(&output)
        .chain_err(|| ErrorKind::SaveObservables)?;
    Ok(())
}

/// Saves the observables to a plain json file.
fn observables_plain(observables: &ObservablesOutput, project: &str) -> Result<()> {
    let filename = format!("{}/observables_{}.json",
                           get_project_dir(project),
                           observables.state);
    let buffer = File::create(&filename)
        .chain_err(|| ErrorKind::CreateFile(filename))?;
    serde_json::to_writer_pretty(buffer, observables)
        .chain_err(|| ErrorKind::SaveObservables)?;
    Ok(())
}

/// Generates a unique folder inside an `output` directory for the current simulation.
pub fn check_output_dir(project: &str) -> Result<()> {
    let proj_dir = get_project_dir(project);
    create_dir_all(&proj_dir)
        .chain_err(|| ErrorKind::CreateOutputDir(proj_dir))?;
    Ok(())
}

/// Each simulation has a unique folder output so as not to overwrite other instances.
/// This function gets the name of the current folder.
///
/// # Arguments
///
/// * `project` - The name of the currently active project as set in the configuration file
///
/// # Returns
///
/// A string containing the location of the output folder, comprised of a sanitized
/// project name, followed by the start date/time of the simulation.
pub fn get_project_dir(project: &str) -> String {
    format!("./output/{}_{}", sanitize_string(project), &**PROJDATE)
}

/// Copies the current configuration file to the project folder
pub fn copy_config(project: &str, file: &str) -> Result<()> {
    let copy_file = get_project_dir(project) + "/" + file;
    copy(file, &copy_file)
        .chain_err(|| ErrorKind::CreateFile(copy_file))?;
    Ok(())
}

/// Uses `term_size` to pull in the terminal width and from there sets the output
/// pretty printing value to an appropriate value (between 70-100).
fn get_term_size() -> usize {
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn term_bounds() {
        let term_size = get_term_size();
        assert!(term_size >= 70 && term_size <= 100);
    }

    #[test]
    fn directory_string() {
        let bad_string = " $//Project*\\";
        assert_eq!(sanitize_string(&bad_string), "_,36,,47,,47,Project,42,,92,");
    }
}

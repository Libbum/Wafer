use chrono::Local;
use csv;
use ndarray::{Array3, ArrayView3, Zip};
use ndarray_parallel::prelude::ParApply2;
use noisy_float::prelude::*;
use ordinal::Ordinal;
use rayon;
use rmps;
use ron::ser::to_string_pretty as ron_string;
use ron::ser::PrettyConfig;
use serde::Serialize;
use serde_json;
use serde_yaml;
use std::fs::{copy, create_dir_all, remove_file, File};
use std::io::prelude::*;
use term_size;
use yansi::Color::Blue;

use config::{Config, FileType, Index3};
use errors::*;
use grid;
use potential::{self, PotentialSubSingle};

lazy_static! {
    /// Date & time at which the simulation was started. Used as a unique identifier for
    /// the output directory of a run.
    static ref PROJDATE: String = Local::now().format("%Y-%m-%d_%H:%M:%S").to_string();
    /// Width of program output to screen.
    pub static ref TERMWIDTH: usize = get_term_size();
}

#[derive(Debug, Serialize)]
/// Structured output of observable values
struct ObservablesOutput {
    /// Excited state number.
    state: u8,
    /// Total energy.
    energy: R64,
    /// Binding energy.
    binding_energy: R64,
    /// Coefficient of determination
    r: R64,
    /// Grid size / Coefficient of determination
    l_r: R64,
}

#[derive(Debug, Serialize)]
/// A simple struct to parse data to a plain csv file
struct PlainRecord {
    /// Index in *x*
    i: usize,
    /// Index in *y*
    j: usize,
    /// Index in *z*
    k: usize,
    /// Data at this position
    data: R64,
}

/// Simply prints the Wafer banner with current commit info and thread count.
pub fn print_banner(sha: &str) {
    println!("                    {}", Blue.paint("___"));
    println!("   __      ____ _  {}__ _ __", Blue.paint("/ __\\"));
    println!("   \\ \\ /\\ / / _` |{} / _ \\ '__|", Blue.paint("/ /"));
    println!(
        "    \\ V  V / (_| {}|  __/ |    Current build SHA1: {}",
        Blue.paint("/ _\\"),
        sha
    );
    println!(
        "     \\_/\\_/ \\__,{}   \\___|_|    Parallel tasks running on {} threads.",
        Blue.paint("/ /"),
        rayon::current_num_threads()
    );
    println!("              {}", Blue.paint("\\__/"));
    println!();
}

/// Handles the saving of potential data to disk.
///
/// # Arguments
/// * `v` - The potential to output.
/// * `project` - The project name (for directory to save to).
/// * `file_type` - What type of file format to use in the output.
pub fn potential(v: &ArrayView3<R64>, project: &str, file_type: &FileType) -> Result<()> {
    let filename = format!(
        "{}/potential{}",
        get_project_dir(project),
        file_type.extentsion()
    );
    match *file_type {
        FileType::Messagepack => write_mpk(v, &filename, ErrorKind::SavePotential),
        FileType::Csv => write_csv(v, &filename),
        FileType::Json => write_json(v, &filename, ErrorKind::SavePotential),
        FileType::Yaml => write_yaml(v, &filename, ErrorKind::SavePotential),
        FileType::Ron => write_ron(v, &filename, ErrorKind::SavePotential),
    }
}

/// Handles the saving of potential_sub data to disk (if required).
///
/// # Arguments
/// * `config` - The configuration struct.
pub fn potential_sub(config: &Config) -> Result<()> {
    let filename = format!(
        "{}/potential_sub{}",
        get_project_dir(&config.project_name),
        config.output.file_type.extentsion()
    );

    let mut sub =
        Array3::<R64>::zeros((config.grid.size.x, config.grid.size.y, config.grid.size.z));
    let (full_sub, single_sub) = if config.potential.variable_pot_sub() {
        // potential_sub is a complete array.
        Zip::indexed(&mut sub).par_apply(|(i, j, k), sub| {
            let idx = Index3 { x: i, y: j, z: k };
            *sub = match potential::potential_sub_idx(config, &idx) {
                Ok(p) => p,
                Err(err) => panic!("Calling invalid potential_sub routine: {}", err),
            };
        });
        (Some(sub.view()), None)
    } else {
        // potential_sub is a single value or zero.
        let sub_val = potential::potential_sub(config)?;
        if sub_val > 0. {
            (None, Some(sub_val))
        } else {
            // no need to write anything. Return early.
            return Ok(());
        }
    };
    match config.output.file_type {
        FileType::Messagepack => write_sub_mpk(&full_sub, single_sub, &filename)?,
        FileType::Csv => write_sub_csv(&full_sub, single_sub, &filename)?,
        FileType::Json => write_sub_json(&full_sub, single_sub, &filename)?,
        FileType::Yaml => write_sub_yaml(&full_sub, single_sub, &filename)?,
        FileType::Ron => write_sub_ron(&full_sub, single_sub, &filename)?,
    }
    Ok(())
}

/// Outputs an array to disk in a plain, csv format
///
/// # Arguments
/// *`array` - The array to output
/// * `filename` - file / directory to save to.
fn write_csv(array: &ArrayView3<R64>, filename: &str) -> Result<()> {
    let mut buffer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_path(filename)
        .chain_err(|| ErrorKind::CreateFile(filename.to_string()))?;
    for ((i, j, k), data) in array.indexed_iter() {
        buffer
            .serialize(PlainRecord {
                i,
                j,
                k,
                data: *data,
            })
            .chain_err(|| ErrorKind::Serialize)?;
    }
    buffer.flush().chain_err(|| ErrorKind::Flush)?;
    Ok(())
}

/// Outputs an array to disk in the messagepack binary format
///
/// # Arguments
/// *`array` - The array to output
/// * `filename` - file / directory to save to.
fn write_mpk(array: &ArrayView3<R64>, filename: &str, err_kind: ErrorKind) -> Result<()> {
    let mut output = Vec::new();
    array
        .serialize(&mut rmps::Serializer::new(&mut output))
        .chain_err(|| ErrorKind::Serialize)?;
    let mut buffer =
        File::create(filename).chain_err(|| ErrorKind::CreateFile(filename.to_string()))?;
    buffer.write_all(&output).chain_err(|| err_kind)?;
    Ok(())
}

/// Outputs an array to disk in json format
///
/// # Arguments
/// *`array` - The array to output
/// * `filename` - file / directory to save to.
fn write_json(array: &ArrayView3<R64>, filename: &str, err_kind: ErrorKind) -> Result<()> {
    let buffer = File::create(&filename).chain_err(|| ErrorKind::CreateFile(filename.to_string()))?;
    serde_json::to_writer_pretty(buffer, array).chain_err(|| err_kind)?;
    Ok(())
}

/// Outputs an array to disk in yaml format
///
/// # Arguments
/// *`array` - The array to output
/// * `filename` - file / directory to save to.
fn write_yaml(array: &ArrayView3<R64>, filename: &str, err_kind: ErrorKind) -> Result<()> {
    let buffer = File::create(&filename).chain_err(|| ErrorKind::CreateFile(filename.to_string()))?;
    serde_yaml::to_writer(buffer, array).chain_err(|| err_kind)?;
    Ok(())
}

/// Outputs an array to disk in ron format
///
/// # Arguments
/// *`array` - The array to output
/// * `filename` - file / directory to save to.
fn write_ron(array: &ArrayView3<R64>, filename: &str, err_kind: ErrorKind) -> Result<()> {
    let mut buffer =
        File::create(&filename).chain_err(|| ErrorKind::CreateFile(filename.to_string()))?;
    let data = ron_string(array, PrettyConfig::default()).chain_err(|| ErrorKind::Serialize)?;
    buffer.write(data.as_bytes()).chain_err(|| err_kind)?;
    Ok(())
}

/// Outputs a potential_sub to disk in messagepack format
///
/// # Arguments
/// * `full_sub` - If the potential_sub is an entire array, this will be a Some.
/// * `single_sub` - If the potential_sub is a singular value, this will be Some.
/// * `filename` - file / directory to save to.
fn write_sub_mpk(
    full_sub: &Option<ArrayView3<R64>>,
    single_sub: Option<R64>,
    filename: &str,
) -> Result<()> {
    let mut buffer =
        File::create(filename).chain_err(|| ErrorKind::CreateFile(filename.to_string()))?;
    if single_sub.is_some() {
        let mut output = Vec::new();
        let sub = PotentialSubSingle {
            pot_sub: single_sub.unwrap(),
        };
        sub.serialize(&mut rmps::Serializer::new(&mut output))
            .chain_err(|| ErrorKind::Serialize)?;
        buffer
            .write_all(&output)
            .chain_err(|| ErrorKind::SavePotentialSub)?;
    } else if full_sub.is_some() {
        let mut output = Vec::new();
        full_sub
            .unwrap()
            .serialize(&mut rmps::Serializer::new(&mut output))
            .chain_err(|| ErrorKind::Serialize)?;
        buffer
            .write_all(&output)
            .chain_err(|| ErrorKind::SavePotentialSub)?;
    }
    Ok(())
}

/// Outputs a potential_sub to disk in csv format
///
/// # Arguments
/// * `full_sub` - If the potential_sub is an entire array, this will be a Some.
/// * `single_sub` - If the potential_sub is a singular value, this will be Some.
/// * `filename` - file / directory to save to.
fn write_sub_csv(
    full_sub: &Option<ArrayView3<R64>>,
    single_sub: Option<R64>,
    filename: &str,
) -> Result<()> {
    let mut buffer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_path(filename)
        .chain_err(|| ErrorKind::CreateFile(filename.to_string()))?;
    if single_sub.is_some() {
        buffer
            .write_record(&[single_sub.unwrap().to_string()])
            .chain_err(|| ErrorKind::Serialize)?;
        buffer.flush().chain_err(|| ErrorKind::Flush)?;
    } else if full_sub.is_some() {
        for ((i, j, k), data) in full_sub.unwrap().indexed_iter() {
            buffer
                .serialize(PlainRecord {
                    i,
                    j,
                    k,
                    data: *data,
                })
                .chain_err(|| ErrorKind::Serialize)?;
        }
        buffer.flush().chain_err(|| ErrorKind::Flush)?;
    }
    Ok(())
}

/// Outputs a potential_sub to disk in json format
///
/// # Arguments
/// * `full_sub` - If the potential_sub is an entire array, this will be a Some.
/// * `single_sub` - If the potential_sub is a singular value, this will be Some.
/// * `filename` - file / directory to save to.
fn write_sub_json(
    full_sub: &Option<ArrayView3<R64>>,
    single_sub: Option<R64>,
    filename: &str,
) -> Result<()> {
    let buffer = File::create(&filename).chain_err(|| ErrorKind::CreateFile(filename.to_string()))?;
    if single_sub.is_some() {
        let sub = PotentialSubSingle {
            pot_sub: single_sub.unwrap(),
        };
        serde_json::to_writer_pretty(buffer, &sub).chain_err(|| ErrorKind::SavePotentialSub)?;
    } else if full_sub.is_some() {
        serde_json::to_writer_pretty(buffer, &full_sub.unwrap())
            .chain_err(|| ErrorKind::SavePotentialSub)?;
    }
    Ok(())
}

/// Outputs a potential_sub to disk in yaml format
///
/// # Arguments
/// * `full_sub` - If the potential_sub is an entire array, this will be a Some.
/// * `single_sub` - If the potential_sub is a singular value, this will be Some.
/// * `filename` - file / directory to save to.
fn write_sub_yaml(
    full_sub: &Option<ArrayView3<R64>>,
    single_sub: Option<R64>,
    filename: &str,
) -> Result<()> {
    let buffer = File::create(&filename).chain_err(|| ErrorKind::CreateFile(filename.to_string()))?;
    if single_sub.is_some() {
        let sub = PotentialSubSingle {
            pot_sub: single_sub.unwrap(),
        };
        serde_yaml::to_writer(buffer, &sub).chain_err(|| ErrorKind::SavePotentialSub)?;
    } else if full_sub.is_some() {
        serde_yaml::to_writer(buffer, &full_sub.unwrap())
            .chain_err(|| ErrorKind::SavePotentialSub)?;
    }
    Ok(())
}

/// Outputs a potential_sub to disk in ron format
///
/// # Arguments
/// * `full_sub` - If the potential_sub is an entire array, this will be a Some.
/// * `single_sub` - If the potential_sub is a singular value, this will be Some.
/// * `filename` - file / directory to save to.
fn write_sub_ron(
    full_sub: &Option<ArrayView3<R64>>,
    single_sub: Option<R64>,
    filename: &str,
) -> Result<()> {
    let mut buffer =
        File::create(&filename).chain_err(|| ErrorKind::CreateFile(filename.to_string()))?;
    if single_sub.is_some() {
        let sub = PotentialSubSingle {
            pot_sub: single_sub.unwrap(),
        };
        let data = ron_string(&sub, PrettyConfig::default()).chain_err(|| ErrorKind::Serialize)?;
        buffer
            .write(data.as_bytes())
            .chain_err(|| ErrorKind::SavePotentialSub)?;
    } else if full_sub.is_some() {
        let data = ron_string(&full_sub.unwrap(), PrettyConfig::default())
            .chain_err(|| ErrorKind::Serialize)?;
        buffer
            .write(data.as_bytes())
            .chain_err(|| ErrorKind::SavePotentialSub)?;
    }
    Ok(())
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
/// * `file_type` - What type of file format to use in the output.
pub fn wavefunction(
    phi: &ArrayView3<R64>,
    num: u8,
    converged: bool,
    project: &str,
    file_type: &FileType,
) -> Result<()> {
    let filename = format!(
        "{}/wavefunction_{}{}{}",
        get_project_dir(project),
        num,
        if converged { "" } else { "_partial" },
        file_type.extentsion()
    );
    match *file_type {
        FileType::Messagepack => write_mpk(phi, &filename, ErrorKind::SaveWavefunction),
        FileType::Csv => write_csv(phi, &filename),
        FileType::Json => write_json(phi, &filename, ErrorKind::SaveWavefunction),
        FileType::Yaml => write_yaml(phi, &filename, ErrorKind::SaveWavefunction),
        FileType::Ron => write_ron(phi, &filename, ErrorKind::SaveWavefunction),
    }
}

/// Removes a temporary `_partial` file from the current output directory.
/// Should only be called if a converged file is written.
///
/// # Arguments
///
/// * `wnum` - The current wavenumber.
/// * `project` - Project name of the current simulation.
/// * `file_type` - What type of file format to use in the output.
pub fn remove_partial(wnum: u8, project: &str, file_type: &FileType) -> Result<()> {
    let filename = format!(
        "{}/wavefunction_{}_partial{}",
        get_project_dir(project),
        wnum,
        file_type.extentsion()
    );
    remove_file(&filename).chain_err(|| ErrorKind::DeletePartial(wnum))?;
    Ok(())
}

/// Pretty prints a header for the subsequent observable data
pub fn print_observable_header(wnum: u8) {
    let width = *TERMWIDTH;
    let spacer = (width - 69) / 2;
    let col2 = 37; //Energy+rrms+1
    println!();
    if let 0 = wnum {
        println!(
            "{:═^lspace$}╤{:═^twidth$}╤{:═^w$}╤{:═^width$}╤{:═^rspace$}",
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
            }
        );
    } else {
        println!(
            "{:═^lspace$}╤{:═^twidth$}╤{:═^w$}╤{:═^width$}╤{:═^rspace$}",
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
            }
        );
    }
    println!(
        "{:^space$}│{:^twidth$}│{:^ewidth$}│{:^width$}│{:^width$}│",
        "",
        "Time (τ)",
        "Energy",
        "rᵣₘₛ",
        "Difference",
        twidth = 12,
        ewidth = 20,
        width = 16,
        space = spacer
    );
    println!(
        "{:─^lspace$}┼{:─^twidth$}┼{:─^ewidth$}┼{:─^width$}┼{:─^width$}┼{:─^rspace$}",
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
        }
    );
}

/// Pretty prints measurements at current step to screen
pub fn print_measurements(tau: R64, diff: R64, observables: &grid::Observables) -> String {
    let width = *TERMWIDTH;
    let spacer = (width - 69) / 2;
    if tau > 0.0 {
        format!(
            "{:^space$}│{:>11.3} │{:>19.10e} │{:15.5} │{:15.5e} │",
            "",
            tau,
            observables.energy / observables.norm2,
            (observables.r2 / observables.norm2).sqrt(),
            diff,
            space = spacer
        )
    } else {
        format!(
            "{:^space$}│{:>11.3} │{:>19.10e} │{:15.5} │{:>15} │",
            "",
            tau,
            observables.energy / observables.norm2,
            (observables.r2 / observables.norm2).sqrt(),
            "--   ",
            space = spacer
        )
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
/// * `file_type` - What type of file format to use in the output.
pub fn finalise_measurement(
    observables: &grid::Observables,
    wnum: u8,
    numx: R64,
    project: &str,
    file_type: &FileType,
) -> Result<()> {
    let r_norm = (observables.r2 / observables.norm2).sqrt();
    let output = ObservablesOutput {
        state: wnum,
        energy: observables.energy / observables.norm2,
        binding_energy: (observables.energy - observables.v_infinity) / observables.norm2,
        r: r_norm,
        l_r: numx / r_norm,
    };

    print_summary(&output);

    match *file_type {
        FileType::Messagepack => observables_mpk(&output, project),
        FileType::Csv => observables_csv(&output, project),
        FileType::Json => observables_json(&output, project),
        FileType::Yaml => observables_yaml(&output, project),
        FileType::Ron => observables_ron(&output, project),
    }
}

/// Pretty print final summary
fn print_summary(output: &ObservablesOutput) {
    let width = *TERMWIDTH;
    let spacer = (width - 69) / 2;

    println!(
        "{:═^lspace$}╧{:═^twidth$}╧{:═^ewidth$}╧{:═^width$}╧{:═^width$}╧{:═^rspace$}",
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
        }
    );
    if let 0 = output.state {
        println!("══▶ Ground state energy = {}", output.energy);
        println!(
            "══▶ Ground state binding energy = {}",
            output.binding_energy
        );
    } else {
        let state = Ordinal::from(output.state);
        println!(
            "══▶ {} excited state energy = {}",
            state, output.energy
        );
        println!(
            "══▶ {} excited state binding energy = {}",
            state, output.binding_energy
        );
    }
    println!("══▶ rᵣₘₛ = {}", output.r);
    println!("══▶ L/rᵣₘₛ = {}", output.l_r);
    println!();
}

/// Saves the observables to a messagepack binary file.
fn observables_mpk(observables: &ObservablesOutput, project: &str) -> Result<()> {
    let filename = format!(
        "{}/observables_{}.mpk",
        get_project_dir(project),
        observables.state
    );
    let mut output = Vec::new();
    observables
        .serialize(&mut rmps::Serializer::new(&mut output))
        .chain_err(|| ErrorKind::Serialize)?;
    let mut buffer = File::create(&filename).chain_err(|| ErrorKind::CreateFile(filename))?;
    buffer
        .write_all(&output)
        .chain_err(|| ErrorKind::SaveObservables)?;
    Ok(())
}

/// Saves the observables to a plain csv file.
fn observables_csv(observables: &ObservablesOutput, project: &str) -> Result<()> {
    let filename = format!(
        "{}/observables_{}.csv",
        get_project_dir(project),
        observables.state
    );
    let mut buffer = csv::Writer::from_path(&filename)
        .chain_err(|| ErrorKind::CreateFile(filename.to_string()))?;
    buffer
        .serialize(observables)
        .chain_err(|| ErrorKind::Serialize)?;
    buffer.flush().chain_err(|| ErrorKind::Flush)?;
    Ok(())
}

/// Saves the observables to a json file.
fn observables_json(observables: &ObservablesOutput, project: &str) -> Result<()> {
    let filename = format!(
        "{}/observables_{}.json",
        get_project_dir(project),
        observables.state
    );
    let buffer = File::create(&filename).chain_err(|| ErrorKind::CreateFile(filename))?;
    serde_json::to_writer_pretty(buffer, observables).chain_err(|| ErrorKind::SaveObservables)?;
    Ok(())
}

/// Saves the observables to a yaml file.
fn observables_yaml(observables: &ObservablesOutput, project: &str) -> Result<()> {
    let filename = format!(
        "{}/observables_{}.yaml",
        get_project_dir(project),
        observables.state
    );
    let buffer = File::create(&filename).chain_err(|| ErrorKind::CreateFile(filename))?;
    serde_yaml::to_writer(buffer, observables).chain_err(|| ErrorKind::SaveObservables)?;
    Ok(())
}

/// Saves the observables to a ron file.
fn observables_ron(observables: &ObservablesOutput, project: &str) -> Result<()> {
    let filename = format!(
        "{}/observables_{}.ron",
        get_project_dir(project),
        observables.state
    );
    let mut buffer =
        File::create(&filename).chain_err(|| ErrorKind::CreateFile(filename.to_string()))?;
    let data = ron_string(observables, PrettyConfig::default()).chain_err(|| ErrorKind::Serialize)?;
    buffer
        .write(data.as_bytes())
        .chain_err(|| ErrorKind::SaveObservables)?;
    Ok(())
}

/// Generates a unique folder inside an `output` directory for the current simulation.
pub fn check_output_dir(project: &str) -> Result<()> {
    let proj_dir = get_project_dir(project);
    create_dir_all(&proj_dir).chain_err(|| ErrorKind::CreateOutputDir(proj_dir))?;
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
    copy(file, &copy_file).chain_err(|| ErrorKind::CreateFile(copy_file))?;
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
    use std::fs;

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

    #[test]
    fn output_directory() {
        assert!(check_output_dir("test").is_ok());
    }

    #[test]
    fn project_directory() {
        let project = "test";
        assert_eq!(
            get_project_dir(project),
            format!("./output/{}_{}", project, &**PROJDATE)
        );
    }

    #[test]
    fn output_observables() {
        let observables = ObservablesOutput {
            state: 1,
            energy: r64(4.0),
            binding_energy: r64(0.0),
            r: r64(1.2),
            l_r: r64(0.3),
        };
        let project = "test";
        // create a dummy output directory.
        let _output = check_output_dir(project);
        assert!(observables_mpk(&observables, &project).is_ok());
        assert!(observables_csv(&observables, &project).is_ok());
        assert!(observables_yaml(&observables, &project).is_ok());
        assert!(observables_json(&observables, &project).is_ok());
        assert!(observables_ron(&observables, &project).is_ok());
        // remove directory
        let _remove = fs::remove_dir_all(get_project_dir(project));
    }

    #[test]
    fn output_potential_sub() {
        let arr = Array3::zeros((2, 2, 2));

        assert!(write_sub_mpk(&None, Some(r64(213.0)), "test.mpk").is_ok());
        assert!(write_sub_csv(&None, Some(r64(21.0)), "test.csv").is_ok());
        assert!(write_sub_yaml(&None, Some(r64(24.8)), "test.yaml").is_ok());
        assert!(write_sub_json(&None, Some(r64(29.1)), "test.json").is_ok());
        assert!(write_sub_ron(&None, Some(r64(94.32)), "test.ron").is_ok());

        assert!(write_sub_mpk(&Some(arr.view()), None, "test.mpk").is_ok());
        assert!(write_sub_csv(&Some(arr.view()), None, "test.csv").is_ok());
        assert!(write_sub_yaml(&Some(arr.view()), None, "test.yaml").is_ok());
        assert!(write_sub_json(&Some(arr.view()), None, "test.json").is_ok());
        assert!(write_sub_ron(&Some(arr.view()), None, "test.ron").is_ok());
        // Remove test files
        let _mpk = fs::remove_file("test.mpk");
        let _csv = fs::remove_file("test.csv");
        let _yaml = fs::remove_file("test.yaml");
        let _json = fs::remove_file("test.json");
        let _ron = fs::remove_file("test.ron");
    }
}

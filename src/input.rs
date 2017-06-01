use csv;
use std::num;
use slog::Logger;
use std::fs::create_dir;
use std::io;
use std::error;
use std::fmt;
use std::path::Path;
use std::process::{Command, Stdio};
use std::io::prelude::*;
use ndarray;
use ndarray::{Array3, Zip};
use ndarray_parallel::prelude::*;
use grid;
use config::{Config, Grid};

#[derive(Debug,Deserialize)]
/// A simple struct to parse data from a plain csv file
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

/// Error type for handling file output. Effectively a wrapper around multiple error types we encounter.
#[derive(Debug)]
pub enum Error {
    /// From disk issues.
    Io(io::Error),
    /// If files are not found on disk,
    NotFound {
        /// Filename or wildcard of missing item.
        value: String
    },
    /// From `csv`.
    Csv(csv::Error),
    /// From `ndarray`.
    Shape(ndarray::ShapeError),
    /// From parsing float.
    FloatParse(num::ParseFloatError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::Io(ref err) => err.fmt(f),
            Error::NotFound { value: ref file } => {
                write!(f, "Cannot find {} in input directory.", file)
            }
            Error::Csv(ref err) => err.fmt(f),
            Error::Shape(ref err) => {
                write!(f,
                       "Calculated and actual size of input data is not aligned ══▶ {}",
                       err)
            }
            Error::FloatParse(ref err) => err.fmt(f),
        }
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::Io(ref err) => err.description(),
            Error::NotFound { .. } => "File not found",
            Error::Csv(ref err) => err.description(),
            Error::Shape(ref err) => err.description(),
            Error::FloatParse(ref err) => err.description(),
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        match *self {
            Error::Io(ref err) => Some(err),
            Error::NotFound { .. } => None,
            Error::Csv(ref err) => Some(err),
            Error::Shape(ref err) => Some(err),
            Error::FloatParse(ref err) => Some(err),
        }
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Error {
        Error::Io(err)
    }
}

impl From<csv::Error> for Error {
    fn from(err: csv::Error) -> Error {
        Error::Csv(err)
    }
}

impl From<ndarray::ShapeError> for Error {
    fn from(err: ndarray::ShapeError) -> Error {
        Error::Shape(err)
    }
}

impl From<num::ParseFloatError> for Error {
    fn from(err: num::ParseFloatError) -> Error {
        Error::FloatParse(err)
    }
}


/// Loads potential file from disk. Handles cases where multiple files exist.
///
/// # Arguments
///
/// * `target_size` - Size of the requested work area for this simulation. If the file on disk does
/// not meet these dimensions, it will be scaled.
/// * `binary` - Configuration flag concerning binary /  plain file output. Will be used as an arbitrator
/// when multiple files are detected.
/// * `log` - Reference to the system logger.
pub fn potential(target_size: [usize; 3],
                 bb: usize,
                 binary: bool,
                 log: &Logger)
                 -> Result<Array3<f64>, Error> {
    let plain_path = "./input/potential.csv";
    let binary_path = "./input/potential.mpk";
    let plain_file = if Path::new(&plain_path).exists() {
        Some(plain_path.to_string())
    } else {
        None
    };
    let binary_file = if Path::new(&binary_path).exists() {
        Some(binary_path.to_string())
    } else {
        None
    };
    println!("{:?}, {:?}", plain_file, binary_file);
    if plain_file.is_some() && binary_file.is_some() {
        warn!(log,
              "Multiple potential files found in input directory. Chosing 'potential.{}' based on configuration settings.",
              if binary { "mpk" } else { "csv" });
        if binary {
            potential_plain(plain_file.unwrap(), target_size, bb)
        } else {
            potential_binary(binary_file.unwrap(), target_size, bb)
        }
    } else if plain_file.is_some() {
        potential_plain(plain_file.unwrap(), target_size, bb)
    } else if binary_file.is_some() {
        potential_binary(binary_file.unwrap(), target_size, bb)
    } else {
        Err(Error::NotFound { value: "potential.*".to_string() })
    }
}

/// Loads a potential from a csv file on disk.
fn potential_plain(file: String, target_size: [usize; 3], bb: usize) -> Result<Array3<f64>, Error> {
    //No need for anything more here, just call the general parser.
    parse_csv_to_array3(file, target_size, bb)
}

/// Loads a potential from a mpk file on disk.
fn potential_binary(file: String,
                    target_size: [usize; 3],
                    bb: usize)
                    -> Result<Array3<f64>, Error> {
    //TODO: Not implemented yet, for now call plain
    let _none = file;
    potential_plain("./input/potential.csv".to_string(), target_size, bb)
}

/// Loads potential file from a script.
///
/// # Arguments
///
/// * `file` - Path of script to generate data from.
/// * `grid` - The `grid` portion of the `config` struct.
/// * `bb` - Bounding box value for assigning central difference boundaries
/// * `log` - Reference to the system logger.
pub fn script_potential(file: &str, grid: &Grid, bb: usize, log: &Logger) -> Result<Array3<f64>, Error> {
    let target_size: [usize; 3] = [grid.size.x + bb, grid.size.y + bb, grid.size.z + bb];
    info!(log, "Generating potential from script file: {}", file);
    // Spawn python script
    let python = Command::new(file).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;

    // Generate some data for the script to process.
    let input = json!({
        "grid": {
            "x": grid.size.x,
            "y": grid.size.y,
            "z": grid.size.z,
            "dn": grid.dn
        }
    });
    // Write a string to the stdin of the python script.
    // stdin has type `Option<ChildStdin>`, but since we know this instance
    // must have one, we can directly unwrap it.
    python.stdin.unwrap().write_all(input.to_string().as_bytes())?;

    // Because stdin does not live after the above calls, it is `drop`ed,
    // and the pipe is closed.
    // This is very important, otherwise python wouldn't start processing the
    // input we just sent.
    // The stdout field also has type `Option<ChildStdout>` so must be unwrapped.
    let mut python_stdout = String::new();
    python.stdout.unwrap().read_to_string(&mut python_stdout)?;

    // Finally, parse the captured string.
    // NOTE: I investigated passing this using messagepack. Ends up being more bytes.
    // Well, that may not be totally true, but printing the byte array to screen is problematic...
    let mut values: Vec<f64> = Vec::new();
    for line in python_stdout.lines() {
        let value = line.parse::<f64>()?;
        values.push(value);
    }
    let generated = Array3::<f64>::from_shape_vec((grid.size.x, grid.size.y, grid.size.z), values)?;

    // generated is now the work area. We need to return a full framed array.
    let mut complete = Array3::<f64>::zeros(target_size);
    {
        let mut work = grid::get_mut_work_area(&mut complete, bb / 2); //NOTE: This is a bit of a hack. But it works.
        // generated is the right size by definition: copy down.
        Zip::from(&mut work)
            .and(generated.view())
            .par_apply(|work, &generated| *work = generated);
    }
    Ok(complete)
}

/// Loads previously computed wavefunctions from disk.
///
/// # Arguments
///
/// * `config` - Reference to the `config` struct.
/// * `log` - Reference to the system logger.
/// * `wstore` - Vector of stored (calculated) wavefunctions.
pub fn load_wavefunctions(config: &Config,
                          log: &Logger,
                          w_store: &mut Vec<Array3<f64>>)
                          -> Result<(), Error> {
    let num = &config.grid.size;
    let bb = config.central_difference.bb();
    let init_size: [usize; 3] = [num.x + bb, num.y + bb, num.z + bb];
    // Load required wavefunctions. If the current state resides on disk as well, we load that later.
    for wnum in 0..config.wavenum {
        let wfn = wavefunction(wnum, init_size, bb, config.output.binary_files, log);
        match wfn {
            Ok(w) => w_store.push(w),
            Err(err) => return Err(err),
            //TODO: Probably need to make these errors a litte more expressive in thier format sections. For example:
            //    panic!("Cannot load any wavefunction_{}* file from input folder: {}", wnum, err)
        }
        info!(log, "Loaded (previous) wavefunction {} from disk", wnum);
    }
    Ok(())
}

/// Loads wavefunction file from disk. Handles cases where multiple files exist.
///
/// # Arguments
///
/// * `wnum` - Excited state level of the wavefunction to load.
/// * `target_size` - Size of the requested work area for this simulation. If the file on disk does
/// not meet these dimensions, it will be scaled.
/// * `binary` - Configuration flag concerning binary /  plain file output. Will be used as an arbitrator
/// when multiple files are detected.
/// * `log` - Reference to the system logger.
pub fn wavefunction(wnum: u8,
                    target_size: [usize; 3],
                    bb: usize,
                    binary: bool,
                    log: &Logger)
                    -> Result<Array3<f64>, Error> {
    let plain_path = format!("./input/wavefunction_{}.csv", wnum);
    let plain_path_partial = format!("./input/wavefunction_{}_partial.csv", wnum);
    let plain_file = if Path::new(&plain_path).exists() {
        Some(plain_path)
    } else if Path::new(&plain_path_partial).exists() {
        Some(plain_path_partial)
    } else {
        None
    };

    let binary_path = format!("./input/wavefunction_{}.mpk", wnum);
    let binary_path_partial = format!("./input/wavefunction_{}_partial.mpk", wnum);
    let binary_file = if Path::new(&binary_path).exists() {
        Some(binary_path)
    } else if Path::new(&binary_path_partial).exists() {
        Some(binary_path_partial)
    } else {
        None
    };

    if plain_file.is_some() && binary_file.is_some() {
        warn!(log,
              "Multiple wavefunction_{} files found in input directory. Chosing '{}' version based on configuration settings.",
              wnum,
              if binary { "mpk" } else { "csv" });
        if binary {
            wavefunction_plain(plain_file.unwrap(), target_size, bb)
        } else {
            wavefunction_binary(binary_file.unwrap(), target_size, bb)
        }
    } else if plain_file.is_some() {
        wavefunction_plain(plain_file.unwrap(), target_size, bb)
    } else if binary_file.is_some() {
        wavefunction_binary(binary_file.unwrap(), target_size, bb)
    } else {
        let missing = format!("wavefunction_{}*.*", wnum);
        Err(Error::NotFound { value: missing })
    }
}

/// Loads a wafefunction from a csv file on disk.
fn wavefunction_plain(file: String,
                      target_size: [usize; 3],
                      bb: usize)
                      -> Result<Array3<f64>, Error> {
    //No more to add here, just parse the file in the generic parser.
    parse_csv_to_array3(file, target_size, bb)
}

/// Loads a wafefunction from a mpk file on disk.
fn wavefunction_binary(file: String,
                       target_size: [usize; 3],
                       bb: usize)
                       -> Result<Array3<f64>, Error> {
    //TODO: Not implemented yet, call plain
    //NOTE: This will guarentee a failure from the file name.
    wavefunction_plain(file, target_size, bb)
}

/// Checks that the folder `input` exists. If not, creates it.
/// This doesn't specifically need to happen for all instances,
/// but we may want to put restart values in there later on.
pub fn check_input_dir() -> Result<(), Error> {
    //std::io::Error
    if !Path::new("./input").exists() {
        create_dir("./input")?;
    }
    Ok(())
}

/// Given a filename, this function reads in the data of a csv file and parses
/// the values into a 3D array. There are a few caveats to this as the file
/// may be of a different shape to the requested size in the configuration file.
/// The routine therefore attempts to resample/interpolate the data to fit the required
/// parameters.
///
/// # Arguments
///
/// * `file` - A filename wrapped in an option. This function is called from filename parsers
/// which may not be able to obtain a valid location.
/// * `target_size` - Requested size of the resultant array. If this size does not match the data
/// pulled from the file, interpolation or resampling will occur.
///
/// # Returns
///
/// * A 3D array loaded with data from the file and resampled/interpolated if required.
/// If something goes wrong in the parsing or file handling, a `csv::Error` is passed.
fn parse_csv_to_array3(file: String,
                       target_size: [usize; 3],
                       bb: usize)
                       -> Result<Array3<f64>, Error> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(file)?;
    let mut max_i = 0;
    let mut max_j = 0;
    let mut max_k = 0;
    let mut data: Vec<f64> = Vec::new();
    for result in rdr.deserialize() {
        let record: PlainRecord = result?;
        if record.i > max_i {
            max_i = record.i
        };
        if record.j > max_j {
            max_j = record.j
        };
        if record.k > max_k {
            max_k = record.k
        };
        data.push(record.data);
    }
    let numx = max_i + 1;
    let numy = max_j + 1;
    let numz = max_k + 1;
    match Array3::<f64>::from_shape_vec((numx, numy, numz), data) {
        Ok(result) => {
            //result is now a parsed Array3 with the work area inside.
            //We must fill this into an array with CD boundaries, provided
            //it is the correct size. If not, we must scale it.
            let init_size: [usize; 3] = [numx + bb, numy + bb, numz + bb];
            let mut complete = Array3::<f64>::zeros(target_size);
            {
                let mut work = grid::get_mut_work_area(&mut complete, bb / 2); //NOTE: This is a bit of a hack. But it works.
                let same: bool = init_size
                    .iter()
                    .zip(target_size.iter())
                    .all(|(a, b)| a == b);
                let smaller: bool = init_size.iter().zip(target_size.iter()).all(|(a, b)| a < b);
                let larger: bool = init_size.iter().zip(target_size.iter()).all(|(a, b)| a > b);
                if same {
                    // Input is the same size, copy down.
                    Zip::from(&mut work)
                        .and(result.view())
                        .par_apply(|work, &result| *work = result);
                } else if smaller {
                    //TODO: Input has lower resolution. Spread it out.
                    panic!("Wavefunction is lower in resolution than requested");
                } else if larger {
                    //TODO: Input has higer resolution. Sample it.
                    panic!("Wavefunction is higher in resolution than requested");
                } else {
                    //TODO: Dimensons are all over the shop. Sample and interp
                    panic!("Wavefunction differs in resolution from requested");
                }
            }
            Ok(complete)
        }
        Err(err) => Err(Error::Shape(err)),
    }
}

use config::{Config, FileType, Grid};
use csv;
use errors::*;
use grid;
use ndarray::{Array1, Array3, ArrayViewMut3, Axis, Zip};
use ndarray_parallel::prelude::*;
use noisy_float::prelude::*;
use potential::PotentialSubSingle;
use rmps;
use ron::de::from_reader as ron_reader;
use serde_json;
use serde_yaml;
use slog::Logger;
use std::fs::{create_dir, File};
use std::io::prelude::*;
use std::path::Path;
use std::process::{Command, Stdio};

#[derive(Debug, Deserialize)]
/// A simple struct to parse data from a plain csv file
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

/// Checks if a potential file exists in the input directory
///
/// #Arguments
///
/// * `extension` - File extension
fn check_potential_file(extension: &str) -> Option<String> {
    let file_path = format!("./input/potential.{}", extension);
    if Path::new(&file_path).exists() {
        Some(file_path)
    } else {
        None
    }
}

/// Checks if a potential_sub file exists in the input directory
///
/// #Arguments
///
/// * `extension` - File extension
fn check_potential_sub_file(extension: &str) -> Option<String> {
    let file_path = format!("./input/potential_sub.{}", extension);
    if Path::new(&file_path).exists() {
        Some(file_path)
    } else {
        None
    }
}

/// Loads potential file from disk. Handles cases where multiple files exist.
///
/// # Arguments
///
/// * `target_size` - Size of the requested work area for this simulation. If the file on disk does
/// not meet these dimensions, it will be scaled.
/// * `file_type` - What type of file format to use in the output. Will be used as an arbitrator
/// when multiple files are detected.
/// * `log` - Reference to the system logger.
pub fn potential(
    target_size: [usize; 3],
    bb: usize,
    file_type: &FileType,
    log: &Logger,
) -> Result<Array3<R64>> {
    let mpk_file = check_potential_file("mpk");
    let csv_file = check_potential_file("csv");
    let json_file = check_potential_file("json");
    let yaml_file = check_potential_file("yaml");
    let ron_file = check_potential_file("ron");

    let file_count = {
        let files = [&mpk_file, &csv_file, &json_file, &yaml_file, &ron_file];
        files.iter().filter(|x| x.is_some()).count()
    };
    if file_count > 1 {
        warn!(
            log,
            "Multiple potential files found in input directory. Chosing '{}' based on configuration settings.",
            file_type
        );
        match *file_type {
            FileType::Messagepack => read_mpk(&mpk_file.unwrap(), target_size, bb, log),
            FileType::Csv => read_csv(&csv_file.unwrap(), target_size, bb, log),
            FileType::Json => read_json(&json_file.unwrap(), target_size, bb, log),
            FileType::Yaml => read_yaml(&yaml_file.unwrap(), target_size, bb, log),
            FileType::Ron => read_ron(&ron_file.unwrap(), target_size, bb, log),
        }
    } else if mpk_file.is_some() {
        read_mpk(&mpk_file.unwrap(), target_size, bb, log)
    } else if csv_file.is_some() {
        read_csv(&csv_file.unwrap(), target_size, bb, log)
    } else if json_file.is_some() {
        read_json(&json_file.unwrap(), target_size, bb, log)
    } else if yaml_file.is_some() {
        read_yaml(&yaml_file.unwrap(), target_size, bb, log)
    } else if ron_file.is_some() {
        read_ron(&ron_file.unwrap(), target_size, bb, log)
    } else {
        Err(ErrorKind::FileNotFound("input/potential.*".to_string()).into())
    }
}

/// Loads an array from a mpk file on disk.
fn read_mpk(file: &str, target_size: [usize; 3], bb: usize, log: &Logger) -> Result<Array3<R64>> {
    let reader = File::open(&file).chain_err(|| ErrorKind::FileNotFound(file.to_string()))?;
    let data: Array3<R64> = rmps::decode::from_read(reader).chain_err(|| ErrorKind::Deserialize)?;

    Ok(fill_data(&file, &data, target_size, bb, log))
}

/// Loads an array from a json file on disk.
fn read_json(file: &str, target_size: [usize; 3], bb: usize, log: &Logger) -> Result<Array3<R64>> {
    let reader = File::open(&file).chain_err(|| ErrorKind::FileNotFound(file.to_string()))?;
    let data: Array3<R64> = serde_json::from_reader(reader).chain_err(|| ErrorKind::Deserialize)?;

    Ok(fill_data(&file, &data, target_size, bb, log))
}

/// Loads an array from a yaml file on disk.
fn read_yaml(file: &str, target_size: [usize; 3], bb: usize, log: &Logger) -> Result<Array3<R64>> {
    let reader = File::open(&file).chain_err(|| ErrorKind::FileNotFound(file.to_string()))?;
    let data: Array3<R64> = serde_yaml::from_reader(reader).chain_err(|| ErrorKind::Deserialize)?;

    Ok(fill_data(&file, &data, target_size, bb, log))
}

/// Loads an array from a ron file on disk.
fn read_ron(file: &str, target_size: [usize; 3], bb: usize, log: &Logger) -> Result<Array3<R64>> {
    let reader = File::open(&file).chain_err(|| ErrorKind::FileNotFound(file.to_string()))?;
    let data: Array3<R64> = ron_reader(reader).chain_err(|| ErrorKind::Deserialize)?;

    Ok(fill_data(&file, &data, target_size, bb, log))
}

/// Once data has been pulled from disk into a convertable format,
/// we contstruct an array ready to calculate on.
/// This requires the addition of a central difference buffer zone and may require
/// the data to be resampled if input sizes differ from the size of the data on disk.
fn fill_data(
    file: &str,
    data: &Array3<R64>,
    target_size: [usize; 3],
    bb: usize,
    log: &Logger,
) -> Array3<R64> {
    let dims = data.dim();
    let init_size: [usize; 3] = [dims.0, dims.1, dims.2];
    let mut complete = Array3::<R64>::zeros(target_size);
    {
        let mut work = grid::get_mut_work_area(&mut complete, bb / 2);
        warn!(log, "{:?}, {:?}", work.dim(), data.dim());
        let same: bool = init_size
            .iter()
            .zip(target_size.iter())
            .all(|(a, b)| a == b);
        if same {
            Zip::from(&mut work)
                .and(data.view())
                .par_apply(|work, &data| *work = data);
        } else {
            info!(log, "Interpolating {} from {:?} to requested size of {:?} (size includes central difference padding).", file, init_size, target_size);
            trilerp_resize(&data, &mut work, target_size);
        }
    }
    complete
}

/// Loads potential file from a script.
///
/// # Arguments
///
/// * `file` - Path of script to generate data from.
/// * `grid` - The `grid` portion of the `config` struct.
/// * `bb` - Bounding box value for assigning central difference boundaries
/// * `log` - Reference to the system logger.
pub fn script_potential(file: &str, grid: &Grid, bb: usize, log: &Logger) -> Result<Array3<R64>> {
    let target_size: [usize; 3] = [grid.size.x + bb, grid.size.y + bb, grid.size.z + bb];
    info!(log, "Generating potential from script file: {}", file);
    // Spawn python script
    let python = Command::new(file)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .chain_err(|| ErrorKind::SpawnPython)?;

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
    python
        .stdin
        .unwrap()
        .write_all(input.to_string().as_bytes())
        .chain_err(|| ErrorKind::StdIn)?;

    // Because stdin does not live after the above calls, it is `drop`ed,
    // and the pipe is closed.
    // This is very important, otherwise python wouldn't start processing the
    // input we just sent.
    // The stdout field also has type `Option<ChildStdout>` so must be unwrapped.
    let mut python_stdout = String::new();
    python
        .stdout
        .unwrap()
        .read_to_string(&mut python_stdout)
        .chain_err(|| ErrorKind::StdOut)?;

    // Finally, parse the captured string.
    // NOTE: I investigated passing this using messagepack. Ends up being more bytes.
    // Well, that may not be totally true, but printing the byte array to screen is problematic...
    let mut values: Vec<R64> = Vec::new();
    for line in python_stdout.lines() {
        let value = line.parse::<f64>().chain_err(|| ErrorKind::ParseFloat)?;
        values.push(r64(value));
    }
    let vlen = values.len();
    let generated = Array3::<R64>::from_shape_vec((grid.size.x, grid.size.y, grid.size.z), values)
        .chain_err(|| ErrorKind::ArrayShape(vlen, [grid.size.x, grid.size.y, grid.size.z]))?;

    // generated is now the work area. We need to return a full framed array.
    let mut complete = Array3::<R64>::zeros(target_size);
    {
        let mut work = grid::get_mut_work_area(&mut complete, bb / 2);
        // generated is the right size by definition: copy down.
        Zip::from(&mut work)
            .and(generated.view())
            .par_apply(|work, &generated| *work = generated);
    }
    Ok(complete)
}

/// Loads potential_sub file from disk. Handles cases where multiple files exist.
///
/// # Arguments
///
/// * `target_size` - Size of the requested work area for this simulation. If the file on disk
/// is a full array, and does not meet these dimensions, it will be scaled.
/// * `file_type` - What type of file format to use in the output. Will be used as an arbitrator
/// when multiple files are detected.
/// * `log` - Reference to the system logger.
pub fn potential_sub(
    target_size: [usize; 3],
    file_type: &FileType,
    log: &Logger,
) -> Result<(Option<Array3<R64>>, Option<R64>)> {
    let mpk_file = check_potential_sub_file("mpk");
    let csv_file = check_potential_sub_file("csv");
    let json_file = check_potential_sub_file("json");
    let yaml_file = check_potential_sub_file("yaml");
    let ron_file = check_potential_sub_file("ron");

    let file_count = {
        let files = [&mpk_file, &csv_file, &json_file, &yaml_file, &ron_file];
        files.iter().filter(|x| x.is_some()).count()
    };
    if file_count > 1 {
        warn!(
            log,
            "Multiple potential_sub files found in input directory. Chosing '{}' based on configuration settings.",
            file_type
        );
        match *file_type {
            FileType::Messagepack => read_sub_mpk(&mpk_file.unwrap(), target_size, log),
            FileType::Csv => read_sub_csv(&csv_file.unwrap(), target_size, log),
            FileType::Json => read_sub_json(&json_file.unwrap(), target_size, log),
            FileType::Yaml => read_sub_yaml(&yaml_file.unwrap(), target_size, log),
            FileType::Ron => read_sub_ron(&ron_file.unwrap(), target_size, log),
        }
    } else if mpk_file.is_some() {
        read_sub_mpk(&mpk_file.unwrap(), target_size, log)
    } else if csv_file.is_some() {
        read_sub_csv(&csv_file.unwrap(), target_size, log)
    } else if json_file.is_some() {
        read_sub_json(&json_file.unwrap(), target_size, log)
    } else if yaml_file.is_some() {
        read_sub_yaml(&yaml_file.unwrap(), target_size, log)
    } else if ron_file.is_some() {
        read_sub_ron(&ron_file.unwrap(), target_size, log)
    } else {
        //No data, potential_sub can be calculated instead
        Err(ErrorKind::FileNotFound("input/potential_sub.*".to_string()).into())
    }
}

/// Loads a potential_sub value or array from a messagepack file on disk.
fn read_sub_mpk(
    file: &str,
    target_size: [usize; 3],
    log: &Logger,
) -> Result<(Option<Array3<R64>>, Option<R64>)> {
    let reader = File::open(&file).chain_err(|| ErrorKind::FileNotFound(file.to_string()))?;
    let full_data: Array3<R64> = if let Ok(data) = rmps::decode::from_read(reader) {
        data
    } else {
        // We didn't match on a full array, so try a single value
        let reader = File::open(&file).chain_err(|| ErrorKind::FileNotFound(file.to_string()))?;
        let single_data: PotentialSubSingle =
            rmps::decode::from_read(reader).chain_err(|| ErrorKind::Deserialize)?;

        return Ok((None, Some(single_data.pot_sub)));
    };

    fill_sub_data(full_data, target_size, log)
}

/// Loads a potential_sub value or array from a csv file on disk.
fn read_sub_csv(
    file: &str,
    target_size: [usize; 3],
    log: &Logger,
) -> Result<(Option<Array3<R64>>, Option<R64>)> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(&file)
        .chain_err(|| ErrorKind::ReadFile(file.to_string()))?;
    let mut max_i = 0;
    let mut max_j = 0;
    let mut max_k = 0;
    let mut data: Vec<R64> = Vec::new();
    let mut rdr_iter = rdr.deserialize();
    // Check the first entry separately. If it contains a PlainRecord, then
    // continue looping.
    if let Some(result) = rdr_iter.next() {
        let record: PlainRecord = if let Ok(r) = result {
            r
        } else {
            // We didn't match on a full array, so try a single value.
            // No need to invoke a csv parser here, it's just a number
            // we can import directly.
            let mut buffer =
                File::open(&file).chain_err(|| ErrorKind::FileNotFound(file.to_string()))?;
            let mut data = String::new();
            buffer
                .read_to_string(&mut data)
                .chain_err(|| ErrorKind::ReadFile(file.to_string()))?;
            let single_data = data
                .trim()
                .parse::<f64>()
                .chain_err(|| ErrorKind::ParseFloat)?;

            return Ok((None, Some(r64(single_data))));
        };
        max_i = record.i;
        max_j = record.j;
        max_k = record.k;
        data.push(record.data);
    };
    for result in rdr_iter {
        let record: PlainRecord =
            result.chain_err(|| ErrorKind::ParsePlainRecord(file.to_string()))?;
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
    let dlen = data.len();
    let full_data = Array3::<R64>::from_shape_vec((numx, numy, numz), data)
        .chain_err(|| ErrorKind::ArrayShape(dlen, [numx, numy, numz]))?;

    fill_sub_data(full_data, target_size, log)
}

/// Loads a potential_sub value or array from a json file on disk.
fn read_sub_json(
    file: &str,
    target_size: [usize; 3],
    log: &Logger,
) -> Result<(Option<Array3<R64>>, Option<R64>)> {
    let reader = File::open(&file).chain_err(|| ErrorKind::FileNotFound(file.to_string()))?;
    let full_data: Array3<R64> = if let Ok(data) = serde_json::from_reader(reader) {
        data
    } else {
        // We didn't match on a full array, so try a single value
        let reader = File::open(&file).chain_err(|| ErrorKind::FileNotFound(file.to_string()))?;
        let single_data: PotentialSubSingle =
            serde_json::from_reader(reader).chain_err(|| ErrorKind::Deserialize)?;

        return Ok((None, Some(single_data.pot_sub)));
    };

    fill_sub_data(full_data, target_size, log)
}

/// Loads a potential_sub value or array from a yaml file on disk.
fn read_sub_yaml(
    file: &str,
    target_size: [usize; 3],
    log: &Logger,
) -> Result<(Option<Array3<R64>>, Option<R64>)> {
    let reader = File::open(&file).chain_err(|| ErrorKind::FileNotFound(file.to_string()))?;
    let full_data: Array3<R64> = if let Ok(data) = serde_yaml::from_reader(reader) {
        data
    } else {
        // We didn't match on a full array, so try a single value
        let reader = File::open(&file).chain_err(|| ErrorKind::FileNotFound(file.to_string()))?;
        let single_data: PotentialSubSingle =
            serde_yaml::from_reader(reader).chain_err(|| ErrorKind::Deserialize)?;

        return Ok((None, Some(single_data.pot_sub)));
    };

    fill_sub_data(full_data, target_size, log)
}

/// Loads a potential_sub value or array from a ron file on disk.
fn read_sub_ron(
    file: &str,
    target_size: [usize; 3],
    log: &Logger,
) -> Result<(Option<Array3<R64>>, Option<R64>)> {
    let reader = File::open(&file).chain_err(|| ErrorKind::FileNotFound(file.to_string()))?;
    let full_data: Array3<R64> = if let Ok(data) = ron_reader(reader) {
        data
    } else {
        // We didn't match on a full array, so try a single value
        let reader = File::open(&file).chain_err(|| ErrorKind::FileNotFound(file.to_string()))?;
        let single_data: PotentialSubSingle =
            ron_reader(reader).chain_err(|| ErrorKind::Deserialize)?;

        return Ok((None, Some(single_data.pot_sub)));
    };

    fill_sub_data(full_data, target_size, log)
}

/// Returns a variable array of `potential_sub` data, which is resized if needed.
fn fill_sub_data(
    full_data: Array3<R64>,
    target_size: [usize; 3],
    log: &Logger,
) -> Result<(Option<Array3<R64>>, Option<R64>)> {
    let fdim = full_data.dim();
    let init_size = [fdim.0, fdim.1, fdim.2];
    let mut work = Array3::<R64>::zeros((target_size[0], target_size[1], target_size[2]));
    let same: bool = init_size
        .iter()
        .zip(target_size.iter())
        .all(|(a, b)| a == b);
    if same {
        Ok((Some(full_data), None))
    } else {
        info!(
            log,
            "Interpolating potential_sub from {:?} to requested size of {:?}.",
            init_size,
            target_size
        );
        trilerp_resize(&full_data, &mut work.view_mut(), target_size);
        Ok((Some(work), None))
    }
}

/// Loads previously computed wavefunctions from disk.
///
/// # Arguments
///
/// * `config` - Reference to the `config` struct.
/// * `log` - Reference to the system logger.
/// * `wstore` - Vector of stored (calculated) wavefunctions.
pub fn load_wavefunctions(
    config: &Config,
    log: &Logger,
    w_store: &mut Vec<Array3<R64>>,
) -> Result<()> {
    let num = &config.grid.size;
    let bb = config.central_difference.bb();
    let init_size: [usize; 3] = [num.x + bb, num.y + bb, num.z + bb];
    // Load required wavefunctions. If the current state resides on disk as well, we load that later.
    for wnum in 0..config.wavenum {
        let wfn = wavefunction(wnum, init_size, bb, &config.output.file_type, log);
        match wfn {
            Ok(w) => w_store.push(w),
            Err(_) => return Err(ErrorKind::LoadWavefunction(wnum).into()),
        }
        info!(log, "Loaded (previous) wavefunction {} from disk", wnum);
    }
    Ok(())
}

/// Checks if a wavefunction file exists in the input directory
///
/// #Arguments
///
/// * `wnum` - Excited state level of the wavefunction.
/// * `extension` - File extension
fn check_wavefunction_file(wnum: u8, extension: &str) -> Option<String> {
    let file_path = format!("./input/wavefunction_{}.{}", wnum, extension);
    let file_path_partial = format!("./input/wavefunction_{}_partial.{}", wnum, extension);
    if Path::new(&file_path).exists() {
        Some(file_path)
    } else if Path::new(&file_path_partial).exists() {
        Some(file_path_partial)
    } else {
        None
    }
}

/// Loads wavefunction file from disk. Handles cases where multiple files exist.
///
/// # Arguments
///
/// * `wnum` - Excited state level of the wavefunction to load.
/// * `target_size` - Size of the requested work area for this simulation. If the file on disk does
/// not meet these dimensions, it will be scaled.
/// * `file_type` - Configuration flag concerning output file types. Will be used as an arbitrator
/// when multiple files are detected.
/// * `log` - Reference to the system logger.
pub fn wavefunction(
    wnum: u8,
    target_size: [usize; 3],
    bb: usize,
    file_type: &FileType,
    log: &Logger,
) -> Result<Array3<R64>> {
    let mpk_file = check_wavefunction_file(wnum, "mpk");
    let csv_file = check_wavefunction_file(wnum, "csv");
    let json_file = check_wavefunction_file(wnum, "json");
    let yaml_file = check_wavefunction_file(wnum, "yaml");
    let ron_file = check_wavefunction_file(wnum, "ron");

    let file_count = {
        let files = [&mpk_file, &csv_file, &json_file, &yaml_file, &ron_file];
        files.iter().filter(|x| x.is_some()).count()
    };
    if file_count > 1 {
        warn!(log,
              "Multiple wavefunction_{} files found in input directory. Chosing '{}' version based on configuration settings.",
              wnum,
              file_type);
        match *file_type {
            FileType::Messagepack => read_mpk(&mpk_file.unwrap(), target_size, bb, log),
            FileType::Csv => read_csv(&csv_file.unwrap(), target_size, bb, log),
            FileType::Json => read_json(&json_file.unwrap(), target_size, bb, log),
            FileType::Yaml => read_yaml(&yaml_file.unwrap(), target_size, bb, log),
            FileType::Ron => read_ron(&ron_file.unwrap(), target_size, bb, log),
        }
    } else if mpk_file.is_some() {
        read_mpk(&mpk_file.unwrap(), target_size, bb, log)
    } else if csv_file.is_some() {
        read_csv(&csv_file.unwrap(), target_size, bb, log)
    } else if json_file.is_some() {
        read_json(&json_file.unwrap(), target_size, bb, log)
    } else if yaml_file.is_some() {
        read_yaml(&yaml_file.unwrap(), target_size, bb, log)
    } else if ron_file.is_some() {
        read_ron(&ron_file.unwrap(), target_size, bb, log)
    } else {
        let missing = format!("input/wavefunction_{}*.*", wnum);
        Err(ErrorKind::FileNotFound(missing.to_string()).into())
    }
}

/// Checks that the folder `input` exists. If not, creates it.
/// This doesn't specifically need to happen for all instances,
/// but we may want to put restart values in there later on.
pub fn check_input_dir() -> Result<()> {
    if !Path::new("./input").exists() {
        create_dir("./input").chain_err(|| ErrorKind::CreateInputDir)?;
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
fn read_csv(file: &str, target_size: [usize; 3], bb: usize, log: &Logger) -> Result<Array3<R64>> {
    let parse_file = &file.to_owned();
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(&file)
        .chain_err(|| ErrorKind::CreateFile(file.to_string()))?;
    let mut max_i = 0;
    let mut max_j = 0;
    let mut max_k = 0;
    let mut data: Vec<R64> = Vec::new();
    for result in rdr.deserialize() {
        let record: PlainRecord =
            result.chain_err(|| ErrorKind::ParsePlainRecord(parse_file.to_string()))?;
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
    let dlen = data.len();
    match Array3::<R64>::from_shape_vec((numx, numy, numz), data) {
        Ok(result) => {
            //result is now a parsed Array3 with the work area inside.
            //We must fill this into an array with CD boundaries, provided
            //it is the correct size. If not, we must scale it.
            let init_size: [usize; 3] = [numx + bb, numy + bb, numz + bb];
            let mut complete = Array3::<R64>::zeros(target_size);
            {
                let mut work = grid::get_mut_work_area(&mut complete, bb / 2);
                let same: bool = init_size
                    .iter()
                    .zip(target_size.iter())
                    .all(|(a, b)| a == b);
                if same {
                    // Input is the same size, copy down.
                    Zip::from(&mut work)
                        .and(result.view())
                        .par_apply(|work, &result| *work = result);
                } else {
                    info!(log, "Interpolating {} from {:?} to requested size of {:?} (size includes central difference padding).", file, init_size, target_size);
                    trilerp_resize(&result, &mut work, target_size);
                }
            }
            Ok(complete)
        }
        Err(_) => Err(ErrorKind::ArrayShape(dlen, [numx, numy, numz]).into()),
    }
}

/// Trilinear interpolation to resize an array.
/// i.e, if we have v.size = (50,50,50), and size = (100, 100, 100)
/// then the output will be (100,100,100) linearly interpolated
fn trilerp_resize(v: &Array3<R64>, output: &mut ArrayViewMut3<R64>, size: [usize; 3]) -> () {
    let nx = v.len_of(Axis(0)) - 1;
    let ny = v.len_of(Axis(1)) - 1;
    let nz = v.len_of(Axis(2)) - 1;

    // Set the basis
    let xi = Array1::linspace(0., nx as f64, size[0]);
    let yi = Array1::linspace(0., ny as f64, size[1]);
    let zi = Array1::linspace(0., nz as f64, size[2]);

    let op = |c0, c1, d| c0 * (1. - d) + c1 * d;
    Zip::indexed(output).par_apply(|(x, y, z), output| {
        // we need to find x,y,z values in the basis of v.
        let xlook = xi[x];
        let ylook = yi[y];
        let zlook = zi[z];
        //No need to bounds check this since we just built it. By construction
        //the value is here somewhere.
        let (x0, x1) = match (0..nx).position(|xx| xx as f64 > xlook) {
            Some(idx) => (idx - 1, idx),
            None => (nx - 1, nx),
        };
        let (y0, y1) = match (0..ny).position(|yy| yy as f64 > ylook) {
            Some(idx) => (idx - 1, idx),
            None => (ny - 1, ny),
        };
        let (z0, z1) = match (0..nz).position(|zz| zz as f64 > zlook) {
            Some(idx) => (idx - 1, idx),
            None => (nz - 1, nz),
        };

        // Calculate distances
        let xd = (xlook - x0 as f64) / (x1 as f64 - x0 as f64);
        let yd = (ylook - y0 as f64) / (y1 as f64 - y0 as f64);
        let zd = (zlook - z0 as f64) / (z1 as f64 - z0 as f64);

        // Interp over x
        let c00 = op(v[(x0, y0, z0)], v[(x1, y0, z0)], xd);
        let c01 = op(v[(x0, y0, z1)], v[(x1, y0, z1)], xd);
        let c10 = op(v[(x0, y1, z0)], v[(x1, y1, z0)], xd);
        let c11 = op(v[(x0, y1, z1)], v[(x1, y1, z1)], xd);

        // Interp over y
        let c0 = op(c00, c10, yd);
        let c1 = op(c01, c11, yd);

        // Interp over z
        *output = op(c0, c1, zd);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn potential_file_doesnt_exist() {
        assert_eq!(check_potential_file("slkj"), None);
    }

    #[test]
    fn wavefunction_file_doesnt_exist() {
        assert_eq!(check_wavefunction_file(80, "ssdt"), None);
    }

    #[test]
    fn interpolation() {
        let array = Array3::<R64>::from_shape_vec(
            (2, 2, 2),
            vec![
                r64(1.),
                r64(2.),
                r64(3.),
                r64(4.),
                r64(5.),
                r64(6.),
                r64(7.),
                r64(8.),
            ],
        ).unwrap();

        let mut complete = Array3::<R64>::zeros((6, 6, 6));
        let mut work = grid::get_mut_work_area(&mut complete, 1);
        println!("{:?}", work.dim());
        trilerp_resize(&array, &mut work, [4, 4, 4]);
        assert_eq!(
            work,
            Array3::<R64>::from_shape_vec(
                (4, 4, 4),
                vec![
                    r64(1.0),
                    r64(1.3333333333333335),
                    r64(1.6666666666666665),
                    r64(2.0),
                    r64(1.6666666666666667),
                    r64(2.0000000000000004),
                    r64(2.3333333333333335),
                    r64(2.666666666666667),
                    r64(2.3333333333333335),
                    r64(2.666666666666667),
                    r64(3.0),
                    r64(3.333333333333333),
                    r64(3.0),
                    r64(3.333333333333333),
                    r64(3.6666666666666665),
                    r64(4.0),
                    r64(2.333333333333333),
                    r64(2.666666666666667),
                    r64(3.0),
                    r64(3.3333333333333335),
                    r64(3.0),
                    r64(3.3333333333333335),
                    r64(3.666666666666667),
                    r64(4.000000000000001),
                    r64(3.666666666666666),
                    r64(4.0),
                    r64(4.333333333333333),
                    r64(4.666666666666667),
                    r64(4.333333333333333),
                    r64(4.666666666666667),
                    r64(5.0),
                    r64(5.333333333333334),
                    r64(3.6666666666666665),
                    r64(4.0),
                    r64(4.333333333333334),
                    r64(4.666666666666667),
                    r64(4.333333333333333),
                    r64(4.666666666666667),
                    r64(5.0),
                    r64(5.333333333333334),
                    r64(5.0),
                    r64(5.333333333333334),
                    r64(5.666666666666667),
                    r64(6.0),
                    r64(5.666666666666666),
                    r64(6.0),
                    r64(6.333333333333332),
                    r64(6.666666666666666),
                    r64(5.0),
                    r64(5.333333333333334),
                    r64(5.666666666666667),
                    r64(6.0),
                    r64(5.666666666666667),
                    r64(6.0),
                    r64(6.333333333333333),
                    r64(6.666666666666666),
                    r64(6.333333333333333),
                    r64(6.666666666666666),
                    r64(7.0),
                    r64(7.333333333333333),
                    r64(7.0),
                    r64(7.333333333333334),
                    r64(7.666666666666666),
                    r64(8.0),
                ]
            ).unwrap()
        );
    }
}

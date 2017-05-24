use csv;
use slog::Logger;
use std::fs::create_dir;
use std::io::{Error, ErrorKind};
use std::path::Path;
use ndarray::{Array3, Zip};
use ndarray_parallel::prelude::*;
use grid;
use config::Config;

/// Loads a wafefunction from a csv file on disk.
pub fn wavefunction_plain(wnum: u8, target_size: [usize; 3]) -> Result<Array3<f64>, csv::Error> {
    let filename = format!("./input/wavefunction_{}.csv", wnum);
    let filename_parital = format!("./input/wavefunction_{}_partial.csv", wnum);
    let file = if Path::new(&filename).exists() {
        Some(filename)
    } else if Path::new(&filename_parital).exists() {
        Some(filename_parital)
    } else {
        None
    };
    parse_csv_to_array3(file, target_size)
}

/// Loads a potential from a csv file on disk.
pub fn potential_plain(target_size: [usize; 3]) -> Result<Array3<f64>, csv::Error> {
    let filename = "./input/potential.csv";
    let file = if Path::new(&filename).exists() {
        Some(filename.to_string())
    } else {
        None
    };
    parse_csv_to_array3(file, target_size)
}


/// Loads previously computed wavefunctions from disk.
pub fn load_wavefunctions(config: &Config, log: &Logger, w_store: &mut Vec<Array3<f64>>) {
    let num = &config.grid.size;
    let init_size: [usize; 3] = [(num.x + 6) as usize,
                                 (num.y + 6) as usize,
                                 (num.z + 6) as usize];
    // Load required wavefunctions. If the current state resides on disk as well, we load that later.
    for wnum in 0..config.wavenum {
        let wfn = wavefunction_plain(wnum, init_size);
        match wfn {
            Ok(w) => w_store.push(w),
            Err(err) => panic!("Cannot load any wavefunction_{}* file from input folder: {}", wnum, err),
        }
        info!(log, "Loaded (previous) wavefunction {} from disk", wnum);
    }
}

/// Checks that the folder `input` exists. If not, creates it.
/// This doesn't specifically need to happen for all instances,
/// but we may want to put restart values in there later on.
///
/// # Panics
/// * If directory can not be created. Gives `std::io::Error`.
pub fn check_input_dir() {
    if !Path::new("./input").exists() {
        let result = create_dir("./input");
        match result {
            Ok(_) => {}
            Err(err) => panic!("Cannot create input directory: {}", err),
        }
    }
}

fn parse_csv_to_array3(file: Option<String>,
                       target_size: [usize; 3])
                       -> Result<Array3<f64>, csv::Error> {
    match file {
        Some(f) => {
            let mut rdr = csv::Reader::from_file(f)?.has_headers(false);
            let mut max_i = 0;
            let mut max_j = 0;
            let mut max_k = 0;
            let mut data: Vec<f64> = Vec::new();
            for record in rdr.decode() {
                let (i, j, k, value): (usize, usize, usize, f64) = record?;
                if i > max_i {
                    max_i = i
                };
                if j > max_j {
                    max_j = j
                };
                if k > max_k {
                    max_k = k
                };
                data.push(value);
            }
            let numx = max_i + 1;
            let numy = max_j + 1;
            let numz = max_k + 1;
            match Array3::<f64>::from_shape_vec((numx, numy, numz), data) {
                Ok(result) => {
                    //result is now a parsed Array3 with the work area inside.
                    //We must fill this into an array with CD boundaries, provided
                    //it is the correct size. If not, we must scale it.
                    let init_size: [usize; 3] = [numx + 6, numy + 6, numz + 6];
                    let mut complete = Array3::<f64>::zeros(target_size);
                    {
                        let mut work = grid::get_mut_work_area(&mut complete);
                        let same: bool = init_size
                            .iter()
                            .zip(target_size.iter())
                            .all(|(a, b)| a == b);
                        let smaller: bool =
                            init_size.iter().zip(target_size.iter()).all(|(a, b)| a < b);
                        let larger: bool =
                            init_size.iter().zip(target_size.iter()).all(|(a, b)| a > b);
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
                Err(err) => panic!("Error parsing file into array: {}", err),
            }
        }
        None => Err(csv::Error::Io(Error::from(ErrorKind::NotFound))),
    }
}

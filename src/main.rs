//! **Wa** ve **f** unction Solv **er**: a parallelized, 3D, Schrödinger equation solver.
//!
//! Wafer exploits a Wick-rotated time-dependent Schrödinger equation to solve for
//! time-independent solutions in three dimensions.
//!
//! Inspired by [quantumfdtd](http://sourceforge.net/projects/quantumfdtd/),
//! which is a proof of concept tool and falls short of a general purpose utility.
//! Wafer attempts to remedy this issue.
//!
//! If you use Wafer in your research, please reference the following articles:
//!
//! M. Strickland and D. Yager-Elorriaga, “A parallel algorithm for solving the 3d
//! Schrödinger equation”,
//! [Journal of Computational Physics __229__, 6015–6026 (2010)](http://dx.doi.org/10.1016/j.jcp.2010.04.032).


#![cfg_attr(feature="clippy", feature(plugin))]
#![cfg_attr(feature="clippy", plugin(clippy))]
#![cfg_attr(feature="clippy", warn(missing_docs_in_private_items))]
#![cfg_attr(feature="clippy", warn(single_match_else))]

extern crate ansi_term;
extern crate chrono;
extern crate csv;
extern crate indicatif;
#[macro_use]
extern crate lazy_static;
#[macro_use(s)]
extern crate ndarray;
extern crate ndarray_parallel;
extern crate num;
extern crate num_cpus;
extern crate ordinal;
extern crate rand;
extern crate rayon;
extern crate rmp_serde as rmps;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;
#[macro_use]
extern crate slog;
extern crate slog_async;
extern crate slog_term;
extern crate term_size;

use slog::{Drain, Duplicate, Logger, Fuse, LevelFilter, Level};
use std::fs::OpenOptions;
use std::process;
use std::time::Instant;
use config::Config;

include!(concat!(env!("OUT_DIR"), "/version.rs"));

/// Config is a (mostly) public module which reads the configuration file `wafer.cfg`
/// and poplulates the `Config` struct with the information required to run the current
/// instance of the application.
pub mod config;
/// The meat of the caclulation is performed on a finite grid. Basically all of the computation
/// work is done within this module.
mod grid;
/// Any required file input (apart from configuration) is handled here. Plain text and binary formats.
mod input;
/// All file output is handled in this module. Plain text and binary options are both here.
mod output;
/// Handles the potential generation, binding energy offsets, callouts to files or scripts
/// if needed etc.
mod potential;

/// System entry point
fn main() {

    let start_time = Instant::now();

    let config = match Config::load() {
        Ok(c) => c,
        Err(err) => {
            println!("Error processing configuration: {}", err);
            process::exit(1);
        }
    };
    if let Err(err) = output::check_output_dir(&config.project_name) {
        println!("Could not communicate with output directory: {}", err);
        process::exit(1);
    };

    let log_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(output::get_project_dir(&config.project_name) + "/simulation.log")
        .expect("Cannot connect to log file");
    let syslog = slog_term::PlainDecorator::new(log_file);
    let sys_drain = slog_term::FullFormat::new(syslog).build().fuse();
    let sys_drain = slog_async::Async::new(sys_drain).build().fuse();
    let screen = slog_term::TermDecorator::new().build();
    let screen_drain = slog_term::FullFormat::new(screen).build().fuse();
    let screen_drain = slog_async::Async::new(screen_drain).build().fuse();

    let log = Logger::root(Fuse::new(Duplicate::new(LevelFilter::new(screen_drain,
                                                                     Level::Warning),
                                                    sys_drain)),
                           o!());

    info!(log, "Starting Wafer solver"; "version" => env!("CARGO_PKG_VERSION"), "build-id" => short_sha());

    info!(log, "Checking/creating directories");
    if let Err(err) = input::check_input_dir() {
        crit!(log, "Could not communicate with input directory: {}", err);
        process::exit(1);
    };

    //Override rayon's defaults of threads (including HT cores) to physical cores
    if let Err(err) = rayon::initialize(rayon::Configuration::new().num_threads(num_cpus::get_physical())) {
        crit!(log, "Failed to initialise thread pool: {}", err);
        process::exit(1);
    };

    let term_width = output::get_term_size();

    let sha = if term_width <= 97 { short_sha() } else { sha() };

    output::print_banner(sha);

    info!(log, "Loading Configuation from disk");
    config.print(term_width);
    if let Err(err) = output::copy_config(&config.project_name) {
       //TODO: Input from CLI if non-default config file
       warn!(log, "Configuration file not copied to output directory: {}", err);
    };

    if let Err(err) = grid::run(&config, &log) {
        crit!(log, "{}", err);
        process::exit(1);
    };

    let elapsed = start_time.elapsed();
    let time_taken = (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64 / 1000_000_000.0);
    match time_taken {
        0.0...60.0 => {
            println!("Simulation complete. Elapsed time: {:.3} seconds.",
                     time_taken)
        }
        60.0...3600.0 => {
            let minutes = (time_taken / 60.).floor();
            let seconds = time_taken - 60. * minutes;
            println!("Simulation complete. Elapsed time: {} minutes, {:.3} seconds.",
                     minutes,
                     seconds);
        }
        _ => {
            let hours = (time_taken / 3600.).floor();
            let minutes = ((time_taken - 3600. * hours) / 60.).floor();
            let seconds = time_taken - 3600. * hours - 60. * minutes;
            println!("Simulation complete. Elapsed time: {} hours, {} minutes, {:.3} seconds.",
                     hours,
                     minutes,
                     seconds);
        }
    }
    info!(log, "Simulation completed");
}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder() {
        let num = 5;
        assert_eq!(num, 5);
    }
}

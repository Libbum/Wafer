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

extern crate ansi_term;
#[macro_use(s)]
extern crate ndarray;
extern crate ndarray_parallel;
extern crate num;
extern crate num_cpus;
extern crate ordinal;
extern crate rand;
extern crate rayon;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;
#[macro_use]
extern crate slog;
extern crate slog_async;
extern crate slog_term;
extern crate term_size;

use slog::Drain;
use std::time::Instant;
use config::Config;

include!(concat!(env!("OUT_DIR"), "/version.rs"));

/// Config is a (mostly) public module which reads the configuration file `wafer.cfg`
/// and poplulates the `Config` struct with the information required to run the current
/// instance of the application.
pub mod config;
mod grid;
mod output;
mod potential;

fn main() {

    let start_time = Instant::now();

    //Override rayon's defaults of threads (including HT cores) to physical cores
    match rayon::initialize(rayon::Configuration::new().num_threads(num_cpus::get_physical())) {
        Ok(_) => {},
        Err(err) => panic!("Failed to initialise thread pool: {}", err),
    };

    let term_width = output::get_term_size();

    let sha = if term_width <= 97 { short_sha() } else { sha() };

    output::print_banner(sha);

    let decorator = slog_term::TermDecorator::new().build();
    let drain = slog_term::FullFormat::new(decorator).build().fuse();
    let drain = slog_async::Async::new(drain).build().fuse();

    let log = slog::Logger::root(drain, o!());

    info!(log, "Loading Configuation from disk");
    let config = Config::load();
    config.print(term_width);

    info!(log, "Checking/creating output directory");
    output::check_output_dir();

    info!(log, "Starting calculation");
    grid::run(&config, &log);

    let elapsed = start_time.elapsed();
    let time_taken = (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64 / 1000_000_000.0);
    match time_taken {
        0.0 ... 60.0 => println!("Simulation complete. Elapsed time: {:.3} seconds.", time_taken),
        60.0 ... 3600.0 => {
            let minutes = (time_taken/60.).floor();
            let seconds = time_taken - 60.*minutes;
            println!("Simulation complete. Elapsed time: {} minutes, {:.3} seconds.", minutes, seconds);
        },
        _ => {
            let hours = (time_taken/3600.).floor();
            let minutes = ((time_taken - 3600.*hours)/60.).floor();
            let seconds = time_taken - 3600.*hours - 60.*minutes;
            println!("Simulation complete. Elapsed time: {} hours, {} minutes, {:.3} seconds.", hours, minutes, seconds);
        },
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder() {
        let num = 5;
        assert_eq!(num, 5);
    }
}

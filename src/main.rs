#![cfg_attr(feature="clippy", feature(plugin))]
#![cfg_attr(feature="clippy", plugin(clippy))]

extern crate ansi_term;
extern crate ndarray;
extern crate ndarray_parallel;
extern crate num;
extern crate rayon;
extern crate serde;
extern crate serde_json;
extern crate term_size;

#[macro_use]
extern crate serde_derive;

use std::time::Instant;
use ansi_term::Colour::Blue;
use config::Config;

include!(concat!(env!("OUT_DIR"), "/version.rs"));

mod config;
mod grid;
mod potential;

fn main() {

    let start_time = Instant::now();

    let mut sha = sha();
    let mut term_width = 100;
    if let Some((width, _)) = term_size::dimensions() {
        if width <= 97 {
            sha = short_sha();
        }
        if width <= 70 {
            term_width = 70;
        } else if width < term_width {
            term_width = width;
        }
    }

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

    let config = Config::load();
    config.print(term_width);
    //   grid::show_complex();
    //   grid::build_array();
    potential::generate(&config);
    //    let idx = config::Index3 { x: 1, y: 2, z: 3 };
    //    println!("Potential at 1,2,3: {}", potential::potential(&config, idx));

    let elapsed = start_time.elapsed();
    let time_taken = (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64 / 1000_000_000.0);
    println!("Elapsed time: {} seconds.", time_taken);
}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder() {
        let num = 5;
        assert_eq!(num, 5);
    }
}

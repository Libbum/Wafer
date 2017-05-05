extern crate ansi_term;
extern crate num;
extern crate serde;
extern crate serde_json;
extern crate term_size;

#[macro_use(array)]
extern crate ndarray;
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
    println!("    \\ V  V / (_| {}|  __/ |", Blue.paint("/ _\\"));
    println!("     \\_/\\_/ \\__,{}   \\___|_|    Current build SHA1: {}",
             Blue.paint("/ /"),
             sha);
    println!("              {}", Blue.paint("\\__/"));
    println!("");

    let config = Config::load();
    config.print(term_width);

    grid::show_complex();
    grid::build_array();
    let idx = config::Index3 { x: 1, y: 2, z: 3 };
    println!("Potential at 1,2,3: {}", potential::potential(&config, idx));

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

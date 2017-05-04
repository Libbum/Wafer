extern crate serde;
extern crate serde_json;

#[macro_use]
extern crate serde_derive;

use std::error::Error;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::path::Path;

include!(concat!(env!("OUT_DIR"), "/version.rs"));

#[derive(Serialize, Deserialize)]
struct Config {
    grid: Grid,
    tolerance: f32,
    max_steps: f64,
    wavenum: u8,
    wavemax: u8,
    clustrun: bool,
    al_clust: Point3,
    output: Output,
    potential: u8,
    mass: f32,
    init_condition: u8,
    sig: f32,
    init_symmetry: u8,
}

#[derive(Serialize, Deserialize)]
struct Grid {
    x: u32,
    y: u32,
    z: u32,
    dn: f32,
    dt: f32,
}

#[derive(Serialize, Deserialize)]
struct Point3 {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Serialize, Deserialize)]
struct Output {
    screen_update: u32,
    snap_update: u32,
    save_wavefns: bool,
    save_potential: bool,
}

fn read_file<P: AsRef<Path>>(file_path: P) -> Result<String, std::io::Error> {
    let mut contents = String::new();
    OpenOptions::new().read(true).open(file_path)?.read_to_string(&mut contents)?;
    Ok(contents)
}

fn load_config() -> Config {
    //Read in configuration file (hjson format)
    let raw_config = match read_file("wafer.cfg") {
        Ok(s) => s,
        Err(err) => panic!("Cannot read configuration file: {}.", err.description()),
    };
    // Decode configuration file.
    let decoded_config: Config = match serde_json::from_str(&raw_config) {
        Ok(c) => c,
        Err(err) => panic!("Error parsing configuration file: {}.", err.description()),
    };
    decoded_config
}


fn main() {

    println!("                    ___");
    println!("   __      ____ _  / __\\__ _ __");
    println!("   \\ \\ /\\ / / _` |/ / / _ \\ '__|");
    println!("    \\ V  V / (_| / _\\|  __/ |");
    println!("     \\_/\\_/ \\__,/ /   \\___|_|    Current build SHA1: {}",
             sha());
    println!("              \\__/");
    println!("");

    let config = load_config();

    println!("{}, {:e}", config.al_clust.z, config.max_steps);

}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder() {
        let num = 5;
        assert_eq!(num, 5);
    }
}

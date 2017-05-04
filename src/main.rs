extern crate ansi_term;
extern crate serde;
extern crate serde_json;
extern crate term_size;

#[macro_use]
extern crate serde_derive;

use ansi_term::Colour::Blue;
use std::error::Error;
use std::fmt;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::path::Path;

include!(concat!(env!("OUT_DIR"), "/version.rs"));

#[derive(Serialize, Deserialize, Debug)]
struct Config {
    grid: Grid,
    tolerance: f32,
    max_steps: f64,
    wavenum: u8,
    wavemax: u8,
    clustrun: bool,
    al_clust: Point3,
    output: Output,
    potential: Potential,
    mass: f32,
    init_condition: InitialCondition,
    sig: f32,
    init_symmetry: SymmetryConstraint,
}

#[derive(Serialize, Deserialize, Debug)]
struct Grid {
    x: u32,
    y: u32,
    z: u32,
    dn: f32,
    dt: f32,
}

#[derive(Serialize, Deserialize, Debug)]
struct Point3 {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Serialize, Deserialize, Debug)]
struct Output {
    screen_update: u32,
    snap_update: u32,
    save_wavefns: bool,
    save_potential: bool,
}

//TODO: This is not a complete list
#[derive(Serialize, Deserialize, Debug)]
enum Potential {
    NoPotential,
    Square,
    QuadWell,
    DoubleWell,
}

impl fmt::Display for Potential {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Potential::NoPotential => write!(f, "No potential (V=0)"),
            Potential::Square => write!(f, "3D square well"),
            Potential::QuadWell => write!(f, "3D quad well (short side along z-axis)"),
            Potential::DoubleWell => write!(f, "3D double well"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
enum InitialCondition {
    FromFile,
    Gaussian,
    Coulomb,
    InteriorConstant,
    BooleanTestGrid,
}

impl fmt::Display for InitialCondition {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            InitialCondition::FromFile => write!(f, "From wavefunction_*.dat on disk"),
            InitialCondition::Gaussian => write!(f, "Random Gaussian"),
            InitialCondition::Coulomb => write!(f, "Coulomb-like"),
            InitialCondition::InteriorConstant => write!(f, "Constant of 0.1 in interior"),
            InitialCondition::BooleanTestGrid => write!(f, "Boolean test grid"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
enum SymmetryConstraint {
    NotConstrained,
    AboutZ,
    AntisymAboutZ,
    AboutY,
    AntisymAboutY,
}

impl fmt::Display for SymmetryConstraint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            SymmetryConstraint::NotConstrained => write!(f, "None"),
            SymmetryConstraint::AboutZ => write!(f, "Symmetric about z-axis"),
            SymmetryConstraint::AntisymAboutZ => write!(f, "Antisymmetric about z-axis"),
            SymmetryConstraint::AboutY => write!(f, "Symmetric about y-axis"),
            SymmetryConstraint::AntisymAboutY => write!(f, "Antisymmetric about y-axis"),
        }
    }
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
        Err(err) => panic!("Error parsing configuration file: {}.", err),
    };
    config_checks(&decoded_config);
    decoded_config
}

fn config_checks(config: &Config) {
    if config.grid.dt > config.grid.dn.powi(2) / 3. {
        panic!("Config Error: Temporal step (grid.dt) must be less than or equal to grid.dn^2/3");
    }
}

fn print_config(config: &Config, w: usize) {
    println!("{:=^width$}", " Configuration ", width = w);
    let mid = w - 10;
    if w > 90 {
        let colwidth = mid / 4;
        let dcolwidth = mid / 2;
        println!("{:5}{:<dwidth$}{:<width$}{:<width$}",
                 "",
                 format!("Grid {{ x: {}, y: {}, z: {} }}",
                         config.grid.x,
                         config.grid.y,
                         config.grid.z),
                 format!("Δ{{x,y,z}}: {:e}", config.grid.dn),
                 format!("Δt: {:e}", config.grid.dt),
                 dwidth = dcolwidth,
                 width = colwidth);
        println!("{:5}{:<width$}{:<width$}{:<width$}{:<width$}",
                 "",
                 format!("Screen update: {}", config.output.screen_update),
                 format!("Snapshot update: {}", config.output.snap_update),
                 format!("Save wavefns: {}", config.output.save_wavefns),
                 format!("Save potential: {}", config.output.save_potential),
                 width = colwidth);
        println!("{:5}{:<twidth$}{:<width$}",
                 "",
                 format!("Potential: {}", config.potential),
                 format!("Mass: {} amu", config.mass),
                 twidth = colwidth * 3,
                 width = colwidth);
        println!("{:5}{:<width$}{:<width$}",
                 "",
                 format!("Energy covergence tolerance: {:e}", config.tolerance),
                 format!("Maximum number of steps: {:e}", config.max_steps),
                 width = dcolwidth);
        println!("{:5}{:<width$}{:<width$}",
                 "",
                 format!("Starting wavefunction: {}", config.wavenum),
                 format!("Maximum wavefunction: {}", config.wavemax),
                 width = dcolwidth);
        if config.clustrun {
            println!("{:5}{:<width$}{:<width$}",
                     "",
                     "Cluster run.",
                     format!("Cluster bounds {{ x: {}, y: {}, z: {} }}",
                             config.al_clust.x,
                             config.al_clust.y,
                             config.al_clust.z),
                     width = dcolwidth);
        }
        if config.init_condition == InitialCondition::Gaussian {
            println!("{:5}{:<width$}{:<width$}",
                     "",
                     format!("Initial conditions: {} ({} σ)",
                             config.init_condition,
                             config.sig),
                     format!("Symmetry Constraints: {}", config.init_symmetry),
                     width = dcolwidth);
        } else {
            println!("{:5}{:<width$}{:<width$}",
                     "",
                     format!("Initial conditions: {}", config.init_condition),
                     format!("Symmetry Constraints: {}", config.init_symmetry),
                     width = dcolwidth);
        }
    } else {
        let colwidth = mid / 2;
        println!("{:5}{}",
                 "",
                 format!("Grid {{ x: {}, y: {}, z: {} }}",
                         config.grid.x,
                         config.grid.y,
                         config.grid.z));
        println!("{:5}{:<width$}{:<width$}",
                 "",
                 format!("Δ{{x,y,z}}: {:e}", config.grid.dn),
                 format!("Δt: {:e}", config.grid.dt),
                 width = colwidth);
        println!("{:5}{:<width$}{:<width$}",
                 "",
                 format!("Screen update: {}", config.output.screen_update),
                 format!("Snapshot update: {}", config.output.snap_update),
                 width = colwidth);
        println!("{:5}{:<width$}{:<width$}",
                 "",
                 format!("Save wavefns: {}", config.output.save_wavefns),
                 format!("Save potential: {}", config.output.save_potential),
                 width = colwidth);
        println!("{:5}{:<twidth$}{:<width$}",
                 "",
                 format!("Potential: {}", config.potential),
                 format!("Mass: {} amu", config.mass),
                 twidth = (mid / 4) * 3,
                 width = mid / 4);
        println!("{:5}{:<width$}{:<width$}",
                 "",
                 format!("Energy covergence tolerance: {:e}", config.tolerance),
                 format!("Maximum number of steps: {:e}", config.max_steps),
                 width = colwidth);
        println!("{:5}{:<width$}{:<width$}",
                 "",
                 format!("Starting wavefunction: {}", config.wavenum),
                 format!("Maximum wavefunction: {}", config.wavemax),
                 width = colwidth);
        if config.clustrun {
            println!("{:5}{:<width$}{:<width$}",
                     "",
                     "Cluster run.",
                     format!("Cluster bounds {{ x: {}, y: {}, z: {} }}",
                             config.al_clust.x,
                             config.al_clust.y,
                             config.al_clust.z),
                     width = colwidth);
        }
        if config.init_condition == InitialCondition::Gaussian {
            println!("{:5}{}",
                     "",
                     format!("Initial conditions: {} ({} σ)",
                             config.init_condition,
                             config.sig));
            println!("{:5}{}",
                     "",
                     format!("Symmetry Constraints: {}", config.init_symmetry));
        } else {
            println!("{:5}{}",
                     "",
                     format!("Initial conditions: {}", config.init_condition));
            println!("{:5}{}",
                     "",
                     format!("Symmetry Constraints: {}", config.init_symmetry));
        }
    }
    println!("{:=^width$}", "=", width = w);
}

fn main() {

    let mut sha = sha();
    let mut term_width = 80;
    if let Some((width, _)) = term_size::dimensions() {
        if width < 90 {
            sha = short_sha();
        }
        term_width = width;
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

    let config = load_config();

    print_config(&config, term_width);

}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder() {
        let num = 5;
        assert_eq!(num, 5);
    }
}

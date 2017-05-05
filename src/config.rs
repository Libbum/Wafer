extern crate serde;
extern crate serde_json;

use std::fmt;
use std::io::Error;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::path::Path;

fn read_file<P: AsRef<Path>>(file_path: P) -> Result<String, Error> {
    let mut contents = String::new();
    OpenOptions::new().read(true).open(file_path)?.read_to_string(&mut contents)?;
    Ok(contents)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Config {
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

//TODO: This isn't implimented at all yet. May not be needed.
#[derive(Serialize, Deserialize, Debug)]
enum RunType {
    Grid,
    Cluster,
    Auto,
}

impl fmt::Display for RunType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            RunType::Grid => write!(f, "Boundary at grid bounds"),
            RunType::Cluster => write!(f, "Boundary at cluster"),
            RunType::Auto => write!(f, "Boundary at the extent of input atoms"),
        }
    }
}


impl Config {
    pub fn load() -> Config {
        //Read in configuration file (hjson format)
        let raw_config = match read_file("wafer.cfg") {
            Ok(s) => s,
            Err(err) => panic!("Cannot read configuration file: {}.", err),
        };
        // Decode configuration file.
        let decoded_config: Config = match serde_json::from_str(&raw_config) {
            Ok(c) => c,
            Err(err) => panic!("Error parsing configuration file: {}.", err),
        };
        Config::parse(&decoded_config);
        decoded_config
    }

    fn parse(&self) {
        if self.grid.dt > self.grid.dn.powi(2) / 3. {
            panic!("Config Error: Temporal step (grid.dt) must be less than or equal to \
                    grid.dn^2/3");
        }
    }

    pub fn print(&self, w: usize) {
        println!("{:=^width$}", " Configuration ", width = w);
        let mid = w - 10;
        if w > 95 {
            let colwidth = mid / 4;
            let dcolwidth = mid / 2;
            println!("{:5}{:<dwidth$}{:<width$}{:<width$}",
                     "",
                     format!("Grid {{ x: {}, y: {}, z: {} }}",
                             self.grid.x,
                             self.grid.y,
                             self.grid.z),
                     format!("Δ{{x,y,z}}: {:e}", self.grid.dn),
                     format!("Δt: {:e}", self.grid.dt),
                     dwidth = dcolwidth,
                     width = colwidth);
            println!("{:5}{:<width$}{:<width$}{:<width$}{:<width$}",
                     "",
                     format!("Screen update: {}", self.output.screen_update),
                     format!("Snapshot update: {}", self.output.snap_update),
                     format!("Save wavefns: {}", self.output.save_wavefns),
                     format!("Save potential: {}", self.output.save_potential),
                     width = colwidth);
            println!("{:5}{:<twidth$}{:<width$}",
                     "",
                     format!("Potential: {}", self.potential),
                     format!("Mass: {} amu", self.mass),
                     twidth = colwidth * 3,
                     width = colwidth);
            println!("{:5}{:<width$}{:<width$}",
                     "",
                     format!("Energy covergence tolerance: {:e}", self.tolerance),
                     format!("Maximum number of steps: {:e}", self.max_steps),
                     width = dcolwidth);
            println!("{:5}{:<width$}{:<width$}",
                     "",
                     format!("Starting wavefunction: {}", self.wavenum),
                     format!("Maximum wavefunction: {}", self.wavemax),
                     width = dcolwidth);
            if self.clustrun {
                println!("{:5}{:<width$}{:<width$}",
                         "",
                         "Cluster run.",
                         format!("Cluster bounds {{ x: {}, y: {}, z: {} }}",
                                 self.al_clust.x,
                                 self.al_clust.y,
                                 self.al_clust.z),
                         width = dcolwidth);
            }
            if self.init_condition == InitialCondition::Gaussian {
                println!("{:5}{:<width$}{:<width$}",
                         "",
                         format!("Initial conditions: {} ({} σ)",
                                 self.init_condition,
                                 self.sig),
                         format!("Symmetry Constraints: {}", self.init_symmetry),
                         width = dcolwidth);
            } else {
                println!("{:5}{:<width$}{:<width$}",
                         "",
                         format!("Initial conditions: {}", self.init_condition),
                         format!("Symmetry Constraints: {}", self.init_symmetry),
                         width = dcolwidth);
            }
        } else {
            let colwidth = mid / 2;
            println!("{:5}{}",
                     "",
                     format!("Grid {{ x: {}, y: {}, z: {} }}",
                             self.grid.x,
                             self.grid.y,
                             self.grid.z));
            println!("{:5}{:<width$}{:<width$}",
                     "",
                     format!("Δ{{x,y,z}}: {:e}", self.grid.dn),
                     format!("Δt: {:e}", self.grid.dt),
                     width = colwidth);
            println!("{:5}{:<width$}{:<width$}",
                     "",
                     format!("Screen update: {}", self.output.screen_update),
                     format!("Snapshot update: {}", self.output.snap_update),
                     width = colwidth);
            println!("{:5}{:<width$}{:<width$}",
                     "",
                     format!("Save wavefns: {}", self.output.save_wavefns),
                     format!("Save potential: {}", self.output.save_potential),
                     width = colwidth);
            println!("{:5}{:<twidth$}{:<width$}",
                     "",
                     format!("Potential: {}", self.potential),
                     format!("Mass: {} amu", self.mass),
                     twidth = (mid / 4) * 3,
                     width = mid / 4);
            println!("{:5}{:<width$}{:<width$}",
                     "",
                     format!("Energy covergence tolerance: {:e}", self.tolerance),
                     format!("Maximum number of steps: {:e}", self.max_steps),
                     width = colwidth);
            println!("{:5}{:<width$}{:<width$}",
                     "",
                     format!("Starting wavefunction: {}", self.wavenum),
                     format!("Maximum wavefunction: {}", self.wavemax),
                     width = colwidth);
            if self.clustrun {
                println!("{:5}{:<width$}{:<width$}",
                         "",
                         "Cluster run.",
                         format!("Cluster bounds {{ x: {}, y: {}, z: {} }}",
                                 self.al_clust.x,
                                 self.al_clust.y,
                                 self.al_clust.z),
                         width = colwidth);
            }
            if self.init_condition == InitialCondition::Gaussian {
                println!("{:5}{}",
                         "",
                         format!("Initial conditions: {} ({} σ)",
                                 self.init_condition,
                                 self.sig));
                println!("{:5}{}",
                         "",
                         format!("Symmetry Constraints: {}", self.init_symmetry));
            } else {
                println!("{:5}{}",
                         "",
                         format!("Initial conditions: {}", self.init_condition));
                println!("{:5}{}",
                         "",
                         format!("Symmetry Constraints: {}", self.init_symmetry));
            }
        }
        println!("{:=^width$}", "=", width = w);
    }
}

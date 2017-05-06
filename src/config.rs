use std::fmt;
use std::io::Error;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::path::Path;
use serde_json;

/// Grid size information. `size` is an **Index3** for now, but maybe could just
/// be tuple. `dn` is the grid size, i.e. Δ{x,y,z}.
#[derive(Serialize, Deserialize, Debug)]
pub struct Grid {
    pub size: Index3,
    pub dn: f64,
    dt: f64,
}

#[derive(Serialize, Deserialize, Debug)]
struct Point3 {
    x: f64,
    y: f64,
    z: f64,
}

/// A simple index struct to identify an {x,y,z} position.
/// In the future it may be a good idea to cast/imply an
/// `ndarray::NdIndex` from/to this.
#[derive(Serialize, Deserialize, Debug)]
pub struct Index3 {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

#[derive(Serialize, Deserialize, Debug)]
struct Output {
    screen_update: u32,
    snap_update: u32,
    save_wavefns: bool,
    save_potential: bool,
}

//TODO: This is not a complete list
/// Type of potential the user wishes to invoke. There are many potientials
/// built in, or the user can opt for two (three) external possibilites:
///
/// 1. Input a pre-calculated potential: `FromFile`
/// 2. Use a **python** script called from `potential_generator.py`: `FromScript`
/// 3. Sumbit an issue or pull request for a potential you deem worthy of
/// inclusion to the built in selection.
#[derive(Serialize, Deserialize, Debug)]
pub enum Potential {
    NoPotential,
    Cube,
    QuadWell,
    ComplexCoulomb,
    FromFile,
    FromScript,
}

impl fmt::Display for Potential {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Potential::NoPotential => write!(f, "No potential (V=0)"),
            Potential::Cube => write!(f, "3D square (i.e. cubic) well"),
            Potential::QuadWell => write!(f, "3D quad well (short side along z-axis)"),
            Potential::ComplexCoulomb => write!(f, "Complex Coulomb"),
            Potential::FromFile => write!(f, "User generated potential from file"),
            Potential::FromScript => write!(f, "User generated potential from script"),
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

/// The main struct which all input data from `wafer.cfg` is pushed into.
#[derive(Serialize, Deserialize, Debug)]
pub struct Config {
    pub grid: Grid,
    tolerance: f32,
    max_steps: f64,
    wavenum: u8,
    wavemax: u8,
    clustrun: bool,
    al_clust: Point3,
    output: Output,
    pub potential: Potential,
    mass: f32,
    init_condition: InitialCondition,
    sig: f32,
    init_symmetry: SymmetryConstraint,
}

impl Config {
    /// Reads and parses data from the `wafer.cfg` file.
    ///
    /// # Panics
    ///
    /// Will dump if we see a file i/o error reading `wafer.cfg`; if the
    /// contents of this file are not valid *json* (soon to be *hjson*
    /// to minimise this possibility); and finally there are a number of checks
    /// on bounds of the user input. For example; grid.dt ≤ grid.dn²/3.
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

    /// Pretty prints the **Config** contents to stdout.
    ///
    /// # Arguments
    ///
    /// * `w` - width of display. This is limited from 70 to 100 characters before being accessed
    /// here, but no such restriction is required inside this function.
    pub fn print(&self, w: usize) {
        println!("{:=^width$}", " Configuration ", width = w);
        let mid = w - 10;
        if w > 95 {
            let colwidth = mid / 4;
            let dcolwidth = mid / 2;
            println!("{:5}{:<dwidth$}{:<width$}{:<width$}",
                     "",
                     format!("Grid {{ x: {}, y: {}, z: {} }}",
                             self.grid.size.x,
                             self.grid.size.y,
                             self.grid.size.z),
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
                             self.grid.size.x,
                             self.grid.size.y,
                             self.grid.size.z));
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

/// Returns the contents of a file on disk to a string
///
/// # Arguments
///
/// * `file_path` - A path to the file one wishes to read from disk. This is cast, so can be an `&str`.
///
/// # Examples
///
/// ```rust
/// let config = read_file("wafer.cfg");
/// ```
///
/// # Errors
///
/// Returns `std::io::Error` if unsuccesful
///
/// # Remarks
///
/// This reader probably doesn't need to be so generic.
/// For now it is only called for `wafer.cfg`.
fn read_file<P: AsRef<Path>>(file_path: P) -> Result<String, Error> {
    let mut contents = String::new();
    OpenOptions::new().read(true).open(file_path)?.read_to_string(&mut contents)?;
    Ok(contents)
}

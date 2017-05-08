use ndarray::{Array3, Zip};
use ndarray_parallel::prelude::*;
use rand::distributions::{Normal, IndependentSample};
use rand;
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
    pub dt: f64,
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

/// Identifies the frequency of ouput to the screen or disk, as
/// well as toggling the output of wavefunction and potential data.
#[derive(Serialize, Deserialize, Debug)]
pub struct Output {
    screen_update: u32,
    snap_update: u32,
    save_wavefns: bool,
    pub save_potential: bool,
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
pub enum PotentialType {
    NoPotential,
    Cube,
    QuadWell,
    ComplexCoulomb,
    FromFile,
    FromScript,
}

impl fmt::Display for PotentialType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            PotentialType::NoPotential => write!(f, "No potential (V=0)"),
            PotentialType::Cube => write!(f, "3D square (i.e. cubic) well"),
            PotentialType::QuadWell => write!(f, "3D quad well (short side along z-axis)"),
            PotentialType::ComplexCoulomb => write!(f, "Complex Coulomb"),
            PotentialType::FromFile => write!(f, "User generated potential from file"),
            PotentialType::FromScript => write!(f, "User generated potential from script"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
enum InitialCondition {
    FromFile,
    Gaussian,
    Coulomb,
    Constant,
    Boolean,
}

impl fmt::Display for InitialCondition {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            InitialCondition::FromFile => write!(f, "From wavefunction_*.dat on disk"),
            InitialCondition::Gaussian => write!(f, "Random Gaussian"),
            InitialCondition::Coulomb => write!(f, "Coulomb-like"),
            InitialCondition::Constant => write!(f, "Constant of 0.1 in interior"),
            InitialCondition::Boolean => write!(f, "Boolean test grid"),
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
    pub wavenum: u8,
    wavemax: u8,
    clustrun: bool,
    al_clust: Point3,
    pub output: Output,
    pub potential: PotentialType,
    mass: f64,
    init_condition: InitialCondition,
    sig: f64,
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
    OpenOptions::new().read(true)
        .open(file_path)?
        .read_to_string(&mut contents)?;
    Ok(contents)
}

/// Sets initial conditions for the wavefunction `w`.
///
/// # Arguments
///
/// * `config` - a reference to the confguration struct
pub fn set_initial_conditions(config: &Config) {
    let num = &config.grid.size;
    //NOTE: Don't forget that sizes are non inclusive. We want num.n + 5 to be our last value, so we need num.n + 6 here.
    let init_size: [usize; 3] = [(num.x + 6) as usize, (num.y + 6) as usize, (num.z + 6) as usize];
    let mut w: Array3<f64> = match config.init_condition {
        InitialCondition::FromFile => Array3::<f64>::zeros((1, 1, 1)), //TODO.
        InitialCondition::Gaussian => generate_gaussian(config, init_size),
        InitialCondition::Coulomb => generate_coulomb(config, init_size),
        InitialCondition::Constant => Array3::<f64>::from_elem(init_size, 0.1),
        InitialCondition::Boolean => generate_boolean(init_size),
    };

    //Enforce Boundary Conditions
    // NOTE: Don't forget that ranges are non-inclusive. So 0..3 means 'select 0,1,2'.
    // In Z
    w.slice_mut(s![.., .., 0..3]).par_map_inplace(|el| *el = 0.);
    w.slice_mut(s![.., .., (init_size[2] - 3) as isize..init_size[2] as isize])
        .par_map_inplace(|el| *el = 0.);
    // In X
    w.slice_mut(s![0..3, .., ..]).par_map_inplace(|el| *el = 0.);
    w.slice_mut(s![(init_size[0] - 3) as isize..init_size[0] as isize, .., ..])
        .par_map_inplace(|el| *el = 0.);
    // In Y
    w.slice_mut(s![.., 0..3, ..]).par_map_inplace(|el| *el = 0.);
    w.slice_mut(s![.., (init_size[1] - 3) as isize..init_size[1] as isize, ..])
        .par_map_inplace(|el| *el = 0.);
    println!("{:?}", w);

    // Symmetrise the IC.
    symmetrise_wavefunction(config, &w);
    //    println!("{:?}", w);
}

/// Builds a gaussian distribution of values with a mean of 0 and standard
/// distribution of `config.sig`.
///
/// # Arguments
///
/// * `config` - a reference to the confguration struct
/// * `init_size` - {x,y,z} dimensions of the required wavefunction
fn generate_gaussian(config: &Config, init_size: [usize; 3]) -> Array3<f64> {
    let normal = Normal::new(0.0, config.sig);
    let mut w = Array3::<f64>::zeros(init_size);

    w.par_map_inplace(|el| *el = normal.ind_sample(&mut rand::thread_rng()));
    w
}

/// Builds a Coulomb-like initial condition.
///
/// # Arguments
///
/// * `config` - a reference to the confguration struct
/// * `init_size` - {x,y,z} dimensions of the required wavefunction
fn generate_coulomb(config: &Config, init_size: [usize; 3]) -> Array3<f64> {
    let mut w = Array3::<f64>::zeros(init_size);

    Zip::indexed(&mut w).par_apply(|(i, j, k), x| {
        //Coordinate system is centered in simulation volume
        let dx = i as f64 - (init_size[0] as f64 / 2.);
        let dy = j as f64 - (init_size[1] as f64 / 2.);
        let dz = k as f64 - (init_size[2] as f64 / 2.);
        let r = config.grid.dn * (dx.powi(2) + dy.powi(2) + dz.powi(2)).sqrt();
        let costheta = config.grid.dn * dz / r;
        let cosphi = config.grid.dn * dx / r;
        let mr2 = (-config.mass * r / 2.).exp();
        // Terms here represent: n=1; n=2, l=0; n=2,l=1,m=0; n=2,l=1,m±1 respectively.
        *x = (-config.mass * r).exp() + (2. - config.mass * r) * mr2 +
             config.mass * r * mr2 * costheta +
             config.mass * r * mr2 * (1. - costheta.powi(2)).sqrt() * cosphi;
    });
    w
}

// Builds a Boolean test grid initial condition.
///
/// # Arguments
///
/// * `init_size` - {x,y,z} dimensions of the required wavefunction
fn generate_boolean(init_size: [usize; 3]) -> Array3<f64> {
    let mut w = Array3::<f64>::zeros(init_size);

    Zip::indexed(&mut w)
        .par_apply(|(i, j, k), x| { *x = i as f64 % 2. * j as f64 % 2. * k as f64 % 2.; });
    w
}

/// Enforces symmetry conditions on wavefunctions
///
/// # Arguments
///
/// * `config` - a reference to the confguration struct
/// * `w` - Reference to a wavefunction to impose symmetry conditions on.
fn symmetrise_wavefunction(config: &Config, w: &Array3<f64>) {
    //TODO: Need to learn how to properly slice an ndarray for this.

    match config.init_symmetry {
        SymmetryConstraint::NotConstrained => {}
        SymmetryConstraint::AboutZ |
        SymmetryConstraint::AntisymAboutZ => {}
        SymmetryConstraint::AboutY |
        SymmetryConstraint::AntisymAboutY => {}
    };
}

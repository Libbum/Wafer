use ndarray::{Array3, Zip};
use ndarray_parallel::prelude::*;
use rand::distributions::{Normal, IndependentSample};
use rand;
use slog::Logger;
use std::error;
use std::fmt;
use std::io;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::path::Path;
use serde_json;
use input;

/// Grid size information.
#[derive(Serialize, Deserialize, Debug)]
pub struct Grid {
    /// Number of grid points (cartesian coordinates).
    pub size: Index3,
    /// The spatial grid size, i.e. Δ{x,y,z}.
    pub dn: f64,
    /// The temporal step size, i.e. Δτ.
    pub dt: f64,
}

#[derive(Serialize, Deserialize, Debug)]
/// A data point in 3D space.
struct Point3 {
    /// Position in *x*.
    x: f64,
    /// Position in *y*.
    y: f64,
    /// Position in *z*.
    z: f64,
}

// TODO: In the future it may be a good idea to cast/imply an `ndarray::NdIndex` from/to this.
/// A simple index struct to identify an {x,y,z} position.
#[derive(Serialize, Deserialize, Debug)]
pub struct Index3 {
    /// Index in *x*.
    pub x: usize,
    /// Index in *y*.
    pub y: usize,
    /// Index in *z*.
    pub z: usize,
}

/// Identifies the frequency of ouput to the screen or disk, as
/// well as toggling the output of wavefunction and potential data.
#[derive(Serialize, Deserialize, Debug)]
pub struct Output {
    /// How many steps should the system evolve before outputting information to the screen.
    pub screen_update: u64,
    /// How many steps should the system evolve before saving a partially converged wavefunction.
    pub snap_update: u64,
    /// Set `true` for files to be saved in a binary format (messagepack). Smaller files, faster save time, but not
    /// human readable. Set `false` for human readable files (csv and json), which take up more disk space and
    /// are slower to write.
    pub binary_files: bool,
    /// Should wavefunctions be saved at all? Not necessary if energy values are the only interest.
    /// Each excited state is saved once it is converged or if `max_steps` is reached.
    pub save_wavefns: bool,
    /// Should the potential be saved for reference? This is output at the start of the simulation.
    pub save_potential: bool,
}

/// Type of potential the user wishes to invoke. There are many potientials
/// built in, or the user can opt for two (three) external possibilites:
///
/// 1. Input a pre-calculated potential: `FromFile`
/// 2. Use a **python** script called from `potential_generator.py`: `FromScript`
/// 3. Sumbit an issue or pull request for a potential you deem worthy of
/// inclusion to the built in selection.
#[derive(Serialize, Deserialize, Debug)]
pub enum PotentialType {
    /// V = 0, no potential at all.
    NoPotential,
    /// A 3D square (i.e. cubic) well.
    Cube,
    /// Quad well, with short side along the *z*-axis.
    QuadWell,
    /// Periodic (sin squared).
    Periodic,
    /// Standard Coulomb.
    Coulomb,
    /// Complex Coulomb.
    ComplexCoulomb,
    /// Eliptical Coulomb.
    ElipticalCoulomb,
    /// Cornell with no corrections.
    SimpleCornell,
    /// Fully anisotropic screened Cornell + spin correction.
    FullCornell,
    /// Harmonic oscillator.
    Harmonic,
    /// Complex harmonic oscillator.
    ComplexHarmonic,
    /// Dodecahedron, because this totally exists in nature.
    Dodecahedron,
    /// Pull data from file. Good to save a little startup time on restart runs,
    /// or a more complex potential in generated from an external tool.
    FromFile,
    /// Calls a python script the user can implement.
    FromScript,
}

impl fmt::Display for PotentialType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            PotentialType::NoPotential => write!(f, "No potential (V=0)"),
            PotentialType::Cube => write!(f, "3D square (i.e. cubic) well"),
            PotentialType::QuadWell => write!(f, "3D quad well (short side along z-axis)"),
            PotentialType::Periodic => write!(f, "Periodic"),
            PotentialType::Coulomb => write!(f, "Coulomb"),
            PotentialType::ComplexCoulomb => write!(f, "Complex coulomb"),
            PotentialType::ElipticalCoulomb => write!(f, "Eliptical coulomb"),
            PotentialType::SimpleCornell => write!(f, "Cornell"),
            PotentialType::FullCornell => {
                write!(f, "Fully anisotropic screened Cornell + spin correction")
            }
            PotentialType::Harmonic => write!(f, "Harmonic oscillator"),
            PotentialType::ComplexHarmonic => write!(f, "Complex harmonic oscillator"),
            PotentialType::Dodecahedron => write!(f, "Dodecahedron"),
            PotentialType::FromFile => write!(f, "User generated potential from file"),
            PotentialType::FromScript => write!(f, "User generated potential from script"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
/// Defines the type of initial condition, or first guess, given to the wavefunction.
pub enum InitialCondition {
    /// Data will be pulled from file. This could be pre-calculated data from some
    /// inferior wavefunction solver, or more likely than not one of two other options.
    ///
    /// 1. A converged excited state lower than the requested start state: e.g. `wavenum` is
    /// set to 2, and a converged excited state 1 is in the `input` directory.
    /// 2. A converged, low resolution version of the current state is in the `input` directory,
    /// which dramatically assists in the calculation time of high resolution runs.
    FromFile,
    /// A random value from the Gaussian distribution, using the standard deviation `sig`.
    Gaussian,
    /// Coulomb-like.
    Coulomb,
    /// A constant value of 0.1.
    Constant,
    /// A Boolean test grid, good for benchmarks.
    Boolean,
}

impl fmt::Display for InitialCondition {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            InitialCondition::FromFile => write!(f, "From file on disk"),
            InitialCondition::Gaussian => write!(f, "Random Gaussian"),
            InitialCondition::Coulomb => write!(f, "Coulomb-like"),
            InitialCondition::Constant => write!(f, "Constant of 0.1 in interior"),
            InitialCondition::Boolean => write!(f, "Boolean test grid"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
/// Symmetry of the wavefunction can be constrained to assist calculation.
enum SymmetryConstraint {
    /// Don't constrain system at all.
    NotConstrained,
    /// Symmetric about *z*-axis.
    AboutZ,
    /// Antisymmetric about *z*-axis.
    AntisymAboutZ,
    /// Symmetric about *y*-axis.
    AboutY,
    /// Antisymmetric about *y*-axis.
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
/// Sets the type of run Wafer will execute.
enum RunType {
    /// A grid based run.
    Grid,
    /// A cluster input.
    Cluster,
    /// Uses magic.
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

/// Error type for handling the configuration stucts.
#[derive(Debug)]
pub enum Error {
    /// From disk issues.
    Io(io::Error),
    /// From `serde_json`.
    DecodeJson(serde_json::Error),
    /// If temporal step `dt` is larger than `dn`^2/3.
    LargeDt,
    /// If `wavenum` is larger than `wavemax`.
    LargeWavenum,
    /// From `input`.
    Input(input::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::Io(ref err) => err.fmt(f),
            Error::DecodeJson(ref err) => err.fmt(f),
            Error::LargeDt => {
                write!(f,
                       "Config Error: Temporal step (grid.dt) must be less than or equal to grid.dn²/3")
            }
            Error::LargeWavenum => {
                write!(f, "Config Error: wavenum can not be larger than wavemax")
            }
            Error::Input(ref err) => err.fmt(f),
        }
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::Io(ref err) => err.description(),
            Error::DecodeJson(ref err) => err.description(),
            Error::LargeDt => "grid.dt >= grid.dn²/3",
            Error::LargeWavenum => "wavenum > wavemax",
            Error::Input(ref err) => err.description(),
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        match *self {
            Error::Io(ref err) => Some(err),
            Error::DecodeJson(ref err) => Some(err),
            Error::LargeDt | Error::LargeWavenum => None,
            Error::Input(ref err) => Some(err),
        }
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Error {
        Error::Io(err)
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Error {
        Error::DecodeJson(err)
    }
}

impl From<input::Error> for Error {
    fn from(err: input::Error) -> Error {
        Error::Input(err)
    }
}

/// The main struct which all input data from `wafer.cfg` is pushed into.
#[derive(Serialize, Deserialize, Debug)]
pub struct Config {
    /// A name for the current project for easy identification of output files.
    pub project_name: String,
    /// Information about the required grid to calculate on.
    pub grid: Grid,
    /// A convergence value, how accurate the total energy needs to be.
    pub tolerance: f64,
    /// The maximum amount of steps the solver should attempt before giving up.
    pub max_steps: u64,
    /// A starting number pertaining to an excited state energy level. To start
    /// at the ground state, this number should be 0. If it is higher, the solver
    /// expects converged states in the `input` directory before calculating anything.
    pub wavenum: u8,
    /// The maxixum number of excited states to calculate. For example, if this value is
    /// 2, the solver will calculate the ground state (E_0), first excited (E_1) and second
    /// excited (E_2) states.
    pub wavemax: u8,
    /// Set to true if you require atomic positions to generate a potential. Default is *false*.
    clustrun: bool,
    /// Bounding box of cluster data if used.
    al_clust: Point3,
    /// Information about the requested output data.
    pub output: Output,
    /// The type of potential required for the simulation. This can be from the internal list,
    /// directly from a pre-calculated file or from a python script.
    pub potential: PotentialType,
    /// Atomic mass if required by the selected potential.
    pub mass: f64,
    /// A first guess at the wavefunction. Can range from Gaussian noise to a low resolution,
    /// pre-calculated solution which will be scaled up to enable a faster convergence at high
    /// resolution.
    pub init_condition: InitialCondition,
    /// Standard deviation. This sets sigma for the Gaussian initial condition if used and is also
    /// required for the Cornell potential types.
    pub sig: f64,
    /// Symmetry contitions forced upon the wavefuntion.
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
    pub fn load() -> Result<Config, Error> {
        //Read in configuration file (hjson format)
        let raw_config = read_file("wafer.cfg")?;
        // Decode configuration file.
        let decoded_config: Config = serde_json::from_str(&raw_config)?;
        Config::parse(&decoded_config)?;
        Ok(decoded_config)
    }

    /// Additional checks to the configuration file that cannot be done implicily
    /// by the type checker.
    fn parse(&self) -> Result<(), Error> {
        if self.grid.dt > self.grid.dn.powi(2) / 3. {
            return Err(Error::LargeDt);
        }
        if self.wavenum > self.wavemax {
            return Err(Error::LargeWavenum);
        }
        Ok(())
    }

    /// Pretty prints the **Config** contents to stdout.
    ///
    /// # Arguments
    ///
    /// * `w` - width of display. This is limited from 70 to 100 characters before being accessed
    /// here, but no such restriction is required inside this function.
    pub fn print(&self, w: usize) {
        println!("{:═^width$}",
                 format!(" {} - Configuration ", self.project_name),
                 width = w);
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
                     format!("Δ{{x,y,z}}: {:.3e}", self.grid.dn),
                     format!("Δt: {:.3e}", self.grid.dt),
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
                     format!("Energy covergence tolerance: {:.3e}", self.tolerance),
                     format!("Maximum number of steps: {:.3e}", self.max_steps as f64),
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
                     format!("Δ{{x,y,z}}: {:.3e}", self.grid.dn),
                     format!("Δt: {:.3e}", self.grid.dt),
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
                     format!("Energy covergence tolerance: {:.3e}", self.tolerance),
                     format!("Maximum number of steps: {:.3e}", self.max_steps as f64),
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
        println!("{:═^width$}", "", width = w);
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
/// `std::io::Error` are the only types returned here, although they are wrapped into the `config::Error::Io` type.
fn read_file<P: AsRef<Path>>(file_path: P) -> Result<String, Error> {
    let mut contents = String::new();
    OpenOptions::new()
        .read(true)
        .open(file_path)?
        .read_to_string(&mut contents)?;
    Ok(contents)
}

/// Sets initial conditions for the wavefunction `w`.
///
/// # Arguments
///
/// * `config` - a reference to the confguration struct
pub fn set_initial_conditions(config: &Config, log: &Logger) -> Result<Array3<f64>, Error> {
    info!(log, "Setting initial conditions for wavefunction");
    let num = &config.grid.size;
    //NOTE: Don't forget that sizes are non inclusive. We want num.n + 5 to be our last value, so we need num.n + 6 here.
    let init_size: [usize; 3] = [(num.x + 6) as usize,
                                 (num.y + 6) as usize,
                                 (num.z + 6) as usize];
    let mut w: Array3<f64> = match config.init_condition {
        InitialCondition::FromFile => {
            input::wavefunction(config.wavenum, init_size, config.output.binary_files, log)?
        }
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

    //NOTE: qfdtd has a zeroing out of W here. We are yet to impliment (may not need W).

    // Symmetrise the IC.
    symmetrise_wavefunction(config, &mut w);
    Ok(w)
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
        .par_apply(|(i, j, k), el| { *el = i as f64 % 2. * j as f64 % 2. * k as f64 % 2.; });
    w
}

/// Enforces symmetry conditions on wavefunctions
///
/// # Arguments
///
/// * `config` - a reference to the confguration struct
/// * `w` - Reference to a wavefunction to impose symmetry conditions on.
pub fn symmetrise_wavefunction(config: &Config, w: &mut Array3<f64>) {
    let num = &config.grid.size;
    let sign = match config.init_symmetry {
        SymmetryConstraint::NotConstrained => 0.,
        SymmetryConstraint::AntisymAboutY |
        SymmetryConstraint::AntisymAboutZ => -1.,
        SymmetryConstraint::AboutY |
        SymmetryConstraint::AboutZ => 1.,
    };

    match config.init_symmetry {
        SymmetryConstraint::NotConstrained => {}
        SymmetryConstraint::AboutZ |
        SymmetryConstraint::AntisymAboutZ => {
            for sx in 0..(num.x + 6) {
                for sy in 3..(3 + num.y + 1) {
                    for sz in 3..(3 + num.z + 1) {
                        let mut z = sz;
                        if z > (3 + num.z) / 2 {
                            z = (3 + num.z) + 1 - z;
                        }
                        w[[sx, sy, sz]] = sign * w[[sx, sy, z]]; //We have to resort to the loops because of this indexing.
                    }
                }
            }
        }
        SymmetryConstraint::AboutY |
        SymmetryConstraint::AntisymAboutY => {
            for sx in 0..(num.x + 6) {
                for sy in 3..(3 + num.y + 1) {
                    let mut y = sy;
                    if y > (3 + num.y) / 2 {
                        y = (3 + num.y) + 1 - y;
                    }
                    for sz in 3..(3 + num.z + 1) {
                        w[[sx, sy, sz]] = sign * w[[sx, y, sz]];
                    }
                }
            }
        }
    };
}

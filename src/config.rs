use ndarray::{Array3, Zip};
use ndarray_parallel::prelude::*;
use rand::distributions::{Normal, IndependentSample};
use rand;
use slog::Logger;
use std::fmt;
use std::fs::File;
use serde_yaml;
use input;
use output;
use errors::*;

/// Grid size information.
#[derive(Serialize, Deserialize, Debug)]
pub struct Grid {
    /// Number of grid points (Cartesian coordinates).
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

/// Identifies the frequency of output to the screen or disk, as
/// well as toggling the output of wavefunction and potential data.
#[derive(Serialize, Deserialize, Debug)]
pub struct Output {
    /// How many steps should the system evolve before outputting information to the screen.
    pub screen_update: u64,
    /// Optional: How many steps should the system evolve before saving a partially converged wavefunction.
    pub snap_update: Option<u64>,
    /// File format to be used for output. `Messagepack` is the smallest (and fastest) option, but not human readable.
    /// Structured text options are `json` and `yaml`, then for a complete plain text option there is `csv`.
    pub file_type: FileType,
    /// Should wavefunctions be saved at all? Not necessary if energy values are the only interest.
    /// Each excited state is saved once it is converged or if `max_steps` is reached.
    pub save_wavefns: bool,
    /// Should the potential be saved for reference? This is output at the start of the simulation.
    pub save_potential: bool,
}

/// Type of potential the user wishes to invoke. There are many potentials
/// built in, or the user can opt for two (three) external possibilities:
///
/// 1. Input a pre-calculated potential: `FromFile`
/// 2. Use a **python** script called from `potential_generator.py`: `FromScript`
/// 3. Submit an issue or pull request for a potential you deem worthy of
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
    /// Elliptical Coulomb.
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
    /// or a more complex potential generated from an external tool.
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

#[derive(Serialize, Deserialize, Debug)]
/// Sets the precision of the central difference formalism.
pub enum CentralDifference {
    /// 3 point, good to 𝓞(`grid.dn`²).
    ThreePoint,
    /// 5 point, good to 𝓞(`grid.dn`⁴).
    FivePoint,
    /// 7 point, good to 𝓞(`grid.dn`⁶).
    SevenPoint,
}

impl CentralDifference {
    /// Grabs the **B** ounding **B** ox size for the current precision.
    pub fn bb(&self) -> usize {
        match *self {
            CentralDifference::ThreePoint => 2,
            CentralDifference::FivePoint => 4,
            CentralDifference::SevenPoint => 6,
        }
    }
    /// Grabs how much the work area is extended in one direction for the current precision.
    pub fn ext(&self) -> usize {
        match *self {
            CentralDifference::ThreePoint => 1,
            CentralDifference::FivePoint => 2,
            CentralDifference::SevenPoint => 3,
        }
    }
}

impl fmt::Display for CentralDifference {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            CentralDifference::ThreePoint => write!(f, "Three point: O(Δ{{x,y,z}}²)"),
            CentralDifference::FivePoint => write!(f, "Five point: O(Δ{{x,y,z}}⁴)"),
            CentralDifference::SevenPoint => write!(f, "Seven point: O(Δ{{x,y,z}}⁶)"),
        }
    }
}

/// File formats available for data output.
#[derive(Serialize, Deserialize, Debug)]
pub enum FileType {
    /// Messagepack: a binary option. Small file sizes (comparatively), but not human readable - can be converted to be however.
    Messagepack,
    /// CSV: a plain text file with comma separated values.
    Csv,
    /// JSON: a popular structured text format found on the web, but also good for Wafer output.
    Json,
    /// YAML: another structured text format that is a little more feature rich than JSON.
    Yaml,
}

impl fmt::Display for FileType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            FileType::Messagepack => write!(f, "Messagepack"),
            FileType::Csv => write!(f, "CSV"),
            FileType::Json => write!(f, "JSON"),
            FileType::Yaml => write!(f, "YAML"),
        }
    }
}

impl FileType {
    /// Returns the file extension of the current output type.
    pub fn extentsion(&self) -> String {
        match *self {
            FileType::Messagepack => ".mpk".to_string(),
            FileType::Csv => ".csv".to_string(),
            FileType::Json => ".json".to_string(),
            FileType::Yaml => ".yaml".to_string(),
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

/// The main struct which all input data from `wafer.cfg` is pushed into.
#[derive(Serialize, Deserialize, Debug)]
pub struct Config {
    /// A name for the current project for easy identification of output files.
    pub project_name: String,
    /// Information about the required grid to calculate on.
    pub grid: Grid,
    /// A convergence value, how accurate the total energy needs to be.
    pub tolerance: f64,
    /// Precision of the central difference formalism. The higher the value here the
    /// lower the resultant error will be, provided the step size has been optimally chosen.
    pub central_difference: CentralDifference,
    /// Optional: The maximum amount of steps the solver should attempt before giving up.
    pub max_steps: Option<u64>,
    /// A starting number pertaining to an excited state energy level. To start
    /// at the ground state, this number should be 0. If it is higher, the solver
    /// expects converged states in the `input` directory before calculating anything.
    pub wavenum: u8,
    /// The maximum number of excited states to calculate. For example, if this value is
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
    /// Symmetry conditions forced upon the wavefuntion.
    init_symmetry: SymmetryConstraint,
    /// Location of the script if using one. This is not required in the input configuration and will
    /// be set as a default value or derived from command line arguments.
    #[serde(skip_deserializing)]
    pub script_location: Option<String>,
}

impl Config {
    /// Reads and parses data from the `wafer.cfg` file and command line arguments.
    pub fn load(file: &str, script: &str) -> Result<Config> {
        let reader = File::open(file)
            .chain_err(|| ErrorKind::ConfigLoad(file.to_string()))?;
        // Decode configuration file.
        let mut decoded_config: Config = serde_yaml::from_reader(reader)
            .chain_err(|| ErrorKind::Deserialize)?;
        Config::parse(&decoded_config)
            .chain_err(|| ErrorKind::ConfigParse)?;

        if let PotentialType::FromScript = decoded_config.potential {
            let mut locale = "./".to_string();
            locale.push_str(script);
            decoded_config.script_location = Some(locale);
        } else {
            decoded_config.script_location = None;
        }

        // Setup ouput directory and copy configuration.
        output::check_output_dir(&decoded_config.project_name)?;
        output::copy_config(&decoded_config.project_name, file)
            .chain_err(|| ErrorKind::CopyConfig(file.to_string()))?;

        Ok(decoded_config)
    }

    /// Additional checks to the configuration file that cannot be done implicitly
    /// by the type checker.
    fn parse(&self) -> Result<()> {
        if self.grid.dt > self.grid.dn.powi(2) / 3. {
            return Err(ErrorKind::LargeDt.into());
        }
        if self.wavenum > self.wavemax {
            return Err(ErrorKind::LargeWavenum.into());
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
        println!(
            "{:═^width$}",
            format!(" {} - Configuration ", self.project_name),
            width = w
        );
        let mid = w - 10;
        if w > 95 {
            let colwidth = mid / 4;
            let dcolwidth = mid / 2;
            println!(
                "{:5}{:<dwidth$}{:<width$}{:<width$}",
                "",
                format!(
                    "Grid {{ x: {}, y: {}, z: {} }}",
                    self.grid.size.x,
                    self.grid.size.y,
                    self.grid.size.z
                ),
                format!("Δ{{x,y,z}}: {:.3e}", self.grid.dn),
                format!("Δt: {:.3e}", self.grid.dt),
                dwidth = dcolwidth,
                width = colwidth
            );
            println!(
                "{:5}{:<width$}{:<width$}{:<width$}{:<width$}",
                "",
                format!("Screen update: {}", self.output.screen_update),
                if self.output.snap_update.is_some() {
                    format!("Snapshot update: {}", self.output.snap_update.unwrap())
                } else {
                    "Snapshot update: Off".to_string()
                },
                format!("Save wavefns: {}", self.output.save_wavefns),
                format!("Save potential: {}", self.output.save_potential),
                width = colwidth
            );
            println!(
                "{:5}{:<width$}{:<width$}",
                "",
                format!("CD precision: {}", self.central_difference),
                format!("Output file format: {}", self.output.file_type),
                width = dcolwidth
            );
            println!(
                "{:5}{:<twidth$}{:<width$}",
                "",
                format!("Potential: {}", self.potential),
                format!("Mass: {} amu", self.mass),
                twidth = colwidth * 3,
                width = colwidth
            );
            println!(
                "{:5}{:<width$}{:<width$}",
                "",
                format!("Energy covergence tolerance: {:.3e}", self.tolerance),
                if self.max_steps.is_some() {
                    format!(
                        "Maximum number of steps: {:.3e}",
                        self.max_steps.unwrap() as f64
                    )
                } else {
                    "Maximum number of steps: ∞".to_string()
                },
                width = dcolwidth
            );
            println!(
                "{:5}{:<width$}{:<width$}",
                "",
                format!("Starting wavefunction: {}", self.wavenum),
                format!("Maximum wavefunction: {}", self.wavemax),
                width = dcolwidth
            );
            if self.clustrun {
                println!(
                    "{:5}{:<width$}{:<width$}",
                    "",
                    "Cluster run.",
                    format!(
                        "Cluster bounds {{ x: {}, y: {}, z: {} }}",
                        self.al_clust.x,
                        self.al_clust.y,
                        self.al_clust.z
                    ),
                    width = dcolwidth
                );
            }
            if self.init_condition == InitialCondition::Gaussian {
                println!(
                    "{:5}{:<width$}{:<width$}",
                    "",
                    format!(
                        "Initial conditions: {} ({} σ)",
                        self.init_condition,
                        self.sig
                    ),
                    format!("Symmetry Constraints: {}", self.init_symmetry),
                    width = dcolwidth
                );
            } else {
                println!(
                    "{:5}{:<width$}{:<width$}",
                    "",
                    format!("Initial conditions: {}", self.init_condition),
                    format!("Symmetry Constraints: {}", self.init_symmetry),
                    width = dcolwidth
                );
            }
        } else {
            let colwidth = mid / 2;
            println!(
                "{:5}{}",
                "",
                format!(
                    "Grid {{ x: {}, y: {}, z: {} }}",
                    self.grid.size.x,
                    self.grid.size.y,
                    self.grid.size.z
                )
            );
            println!(
                "{:5}{:<width$}{:<width$}",
                "",
                format!("Δ{{x,y,z}}: {:.3e}", self.grid.dn),
                format!("Δt: {:.3e}", self.grid.dt),
                width = colwidth
            );
            println!(
                "{:5}{:<width$}{:<width$}",
                "",
                format!("Screen update: {}", self.output.screen_update),
                if self.output.snap_update.is_some() {
                    format!("Snapshot update: {}", self.output.snap_update.unwrap())
                } else {
                    "Snapshot update: Off".to_string()
                },
                width = colwidth
            );
            println!(
                "{:5}{:<width$}{:<width$}",
                "",
                format!("Save wavefns: {}", self.output.save_wavefns),
                format!("Save potential: {}", self.output.save_potential),
                width = colwidth
            );
            println!(
                "{:5}{:<width$}{:<width$}",
                "",
                format!("CD precision: {}", self.central_difference),
                format!("Output file format: {}", self.output.file_type),
                width = colwidth
            );
            println!(
                "{:5}{:<twidth$}{:<width$}",
                "",
                format!("Potential: {}", self.potential),
                format!("Mass: {} amu", self.mass),
                twidth = (mid / 4) * 3,
                width = mid / 4
            );
            println!(
                "{:5}{:<width$}{:<width$}",
                "",
                format!("Energy covergence tolerance: {:.3e}", self.tolerance),
                if self.max_steps.is_some() {
                    format!(
                        "Maximum number of steps: {:.3e}",
                        self.max_steps.unwrap() as f64
                    )
                } else {
                    "Maximum number of steps: ∞".to_string()
                },
                width = colwidth
            );
            println!(
                "{:5}{:<width$}{:<width$}",
                "",
                format!("Starting wavefunction: {}", self.wavenum),
                format!("Maximum wavefunction: {}", self.wavemax),
                width = colwidth
            );
            if self.clustrun {
                println!(
                    "{:5}{:<width$}{:<width$}",
                    "",
                    "Cluster run.",
                    format!(
                        "Cluster bounds {{ x: {}, y: {}, z: {} }}",
                        self.al_clust.x,
                        self.al_clust.y,
                        self.al_clust.z
                    ),
                    width = colwidth
                );
            }
            if self.init_condition == InitialCondition::Gaussian {
                println!(
                    "{:5}{}",
                    "",
                    format!(
                        "Initial conditions: {} ({} σ)",
                        self.init_condition,
                        self.sig
                    )
                );
                println!(
                    "{:5}{}",
                    "",
                    format!("Symmetry Constraints: {}", self.init_symmetry)
                );
            } else {
                println!(
                    "{:5}{}",
                    "",
                    format!("Initial conditions: {}", self.init_condition)
                );
                println!(
                    "{:5}{}",
                    "",
                    format!("Symmetry Constraints: {}", self.init_symmetry)
                );
            }
        }
        println!("{:═^width$}", "", width = w);
    }
}

/// Sets initial conditions for the wavefunction `w`.
///
/// # Arguments
///
/// * `config` - a reference to the configuration struct.
/// * `log` - a reference to the logger.
pub fn set_initial_conditions(config: &Config, log: &Logger) -> Result<Array3<f64>> {
    info!(log, "Setting initial conditions for wavefunction");
    let num = &config.grid.size;
    let bb = config.central_difference.bb();
    let init_size: [usize; 3] = [
        num.x as usize + bb,
        num.y as usize + bb,
        num.z as usize + bb,
    ];
    let mut w: Array3<f64> = match config.init_condition {
        InitialCondition::FromFile => {
            input::wavefunction(config.wavenum, init_size, bb, &config.output.file_type, log)
                .chain_err(|| ErrorKind::LoadWavefunction(config.wavenum))?
        }
        InitialCondition::Gaussian => generate_gaussian(config, init_size),
        InitialCondition::Coulomb => generate_coulomb(config, init_size),
        InitialCondition::Constant => Array3::<f64>::from_elem(init_size, 0.1),
        InitialCondition::Boolean => generate_boolean(init_size),
    };

    //Enforce Boundary Conditions
    let ext = config.central_difference.ext() as isize;
    // In Z
    w.slice_mut(s![.., .., 0..ext])
        .par_map_inplace(|el| *el = 0.);
    w.slice_mut(s![.., .., init_size[2] as isize - ext..init_size[2] as isize])
        .par_map_inplace(|el| *el = 0.);
    // In X
    w.slice_mut(s![0..ext, .., ..])
        .par_map_inplace(|el| *el = 0.);
    w.slice_mut(s![init_size[0] as isize - ext..init_size[0] as isize, .., ..])
        .par_map_inplace(|el| *el = 0.);
    // In Y
    w.slice_mut(s![.., 0..ext, ..])
        .par_map_inplace(|el| *el = 0.);
    w.slice_mut(s![.., init_size[1] as isize - ext..init_size[1] as isize, ..])
        .par_map_inplace(|el| *el = 0.);

    // Symmetrise the IC.
    symmetrise_wavefunction(config, &mut w);
    Ok(w)
}

/// Builds a gaussian distribution of values with a mean of 0 and standard
/// distribution of `config.sig`.
///
/// # Arguments
///
/// * `config` - a reference to the configuration struct
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
/// * `config` - a reference to the configuration struct
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

/// Builds a Boolean test grid initial condition.
///
/// # Arguments
///
/// * `init_size` - {x,y,z} dimensions of the required wavefunction
fn generate_boolean(init_size: [usize; 3]) -> Array3<f64> {
    let mut w = Array3::<f64>::zeros(init_size);

    Zip::indexed(&mut w).par_apply(|(i, j, k), el| {
        *el = i as f64 % 2. * j as f64 % 2. * k as f64 % 2.;
    });
    w
}

/// Enforces symmetry conditions on wavefunctions
///
/// # Arguments
///
/// * `config` - a reference to the configuration struct
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

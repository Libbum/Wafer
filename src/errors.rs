error_chain!{
    errors {
        ConfigLoad(path: String) {
                description("Config file not found")
                display("Unable to read file `{}`", path)
        }

        ConfigParse {
            description("Error parsing config")
            display("an error occurred trying to parse the configuration file")
        }

        SetInitialConditions {
            description("Error setting initial conditions")
            display("an error occurred trying to set the initialisation conditions on the starting wavefunction")
        }
        LargeDt {
            description("grid.dt >= grid.dn²/3")
            display("Temporal step (grid.dt) must be less than or equal to grid.dn²/3")
        }
        LargeWavenum {
            description("wavenum > wavemax")
            display("Wavenum can not be larger than wavemax")
        }

        CreateLog(path: String) {
                description("Cannot write log file")
                display("Unable to write log file `{}`", path)
        }

        FileNotFound(path: String) {
                description("File not found")
                display("Unable to find file `{}`", path)
        }
        CreateInputDir {
                description("Cannot create input dir")
                display("Unable to create an input directory")
        }
        CreateOutputDir(path: String) {
                description("Cannot create output dir")
                display("Unable to create the output directory '{}'", path)
        }
        CreateFile(file: String) {
                description("Cannot create file")
                display("Unable to create {}", file)
        }
        ReadFile(file: String) {
                description("Cannot read file")
                display("Unable to read {}", file)
        }
        ParseFloat {
                description("Cannot parse float")
                display("Unable to parse string to f64")
        }
        ParsePotentialSubSingle(file: String) {
                description("Cannot parse csv data")
                display("Unable to parse a string of data into a valid record from file {}", file)
        }
        ParsePlainRecord(file: String) {
                description("Cannot parse csv data")
                display("Unable to parse a string of data into a valid record from file {}", file)
        }
        ArrayShape(len: usize, dims: [usize;3]) {
                description("Cannot reshape array")
                display("Unable to reshape vector with length {} into an array with dimensions {:?}", len, dims)
        }
        StdIn {
                description("Cannot write to stdin")
                display("Unable to write to stdin in of the python script process")
        }
        StdOut {
                description("Cannot recieve stdout")
                display("Unable to recieve data from stdout of the python script process")
        }
        SpawnPython {
                description("Cannot spawn script")
                display("Unable to spawn a python script process")
        }
        SaveObservables {
                description("Cannot save observables")
                display("Unable to save observables data to disk")
        }
        SavePotential {
                description("Cannot save potential")
                display("Unable to save potential data to disk")
        }
        SavePotentialSub {
                description("Cannot save potential_sub")
                display("Unable to save potential_sub data to disk")
        }
        WrongPotentialSubDims {
                description("wrong dimensions in potential_sub")
                display("Unable to identify the correct dimensions in potential_sub input file")
        }
        SaveWavefunction {
                description("Cannot save wavefunction")
                display("Unable to save wavefunction data to disk")
        }
        Serialize {
                description("Cannot serialize data")
                display("Unable to serialize data from struct")
        }
        Deserialize {
                description("Cannot deserialize data")
                display("Unable to deserialize data to required struct")
        }
        Flush {
                description("Cannot flush")
                display("Unable to flush io buffer")
        }
        MaxStep {
                description("Maximum step reached")
                display("Maximum step limit reached, halting operation")
        }
        PotentialNotAvailable {
                description("Not available for PotentialType")
                display("Invalid call for current potential type")
        }
        ScriptNotFound {
                description("Cannot find script")
                display("Unable to locate potential script")
        }
        CopyConfig(file: String) {
                description("Cannot copy configuration")
                display("Unable to copy configuration file '{}' to output directory", file)
        }
        LoadWavefunction(wnum: u8) {
                description("Cannot load wavefunction")
                display("Unable to load wavefunction {} from disk", wnum)
        }
        LoadPotential {
                description("Cannot load potential")
                display("Unable to load potential from disk")
        }
        DeletePartial(wnum: u8) {
                description("Cannot delete partial")
                display("Unable remove the temporary file of wavefunction {}", wnum)
        }
    }
}

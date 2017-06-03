error_chain!{
    errors {
        ConfigLoad(path: String) {
                description("Config file not found")
                display("Unable to read file `{}`", path)
        }

        ConfigParse {
            description("Error parsing config")
            display("an error occurred trying to parse the configuratation file")
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
        ParseFloat {
                description("Cannot parse float")
                display("Unable to parse string to f64")
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
    }
}

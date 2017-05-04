extern crate serde;
extern crate serde_hjson;

use serde_hjson::{Map,Value};
use std::error::Error;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::path::Path;

include!(concat!(env!("OUT_DIR"), "/version.rs"));

fn read_file<P: AsRef<Path>>(file_path: P) -> Result<String, std::io::Error> {
    let mut contents = String::new();
    OpenOptions::new().read(true).open(file_path)?.read_to_string(&mut contents)?;
    Ok(contents)
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

    //Read in configuration file (hjson format)
    let raw_config = match read_file("wafer.cfg") {
        Ok(s) => s,
        Err(err) => panic!("Cannot read configuration file: {}.", err.description()),
    };

    println!("{:}", raw_config);
    // Decode configuration file.
    let config: Map<String, Value> = match serde_hjson::from_str(&raw_config) {
        Ok(m) => m,
        Err(err) => panic!("Cannot parse configuration file: {}.", err.description()),
    };
    println!("{:}", config["output"]);
    println!("{:}", config["mass"]);

    // scope to control lifetime of borrow
    {
        // Extract the rate
        //let rate = sample.get("rate").unwrap().as_f64().unwrap();
        //println!("rate: {}", rate);

        // Extract the array
        //let array : &mut Vec<Value> = sample.get_mut("array").unwrap().as_array_mut().unwrap();
        //println!("first: {}", array.get(0).unwrap());

        // Add a value
        //array.push(Value::String("tak".to_string()));
    }

    // Encode to Hjson
    //let sample2 = serde_hjson::to_string(&sample).unwrap();
    //println!("Hjson:\n{}", sample2);

}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder() {
        let num = 5;
        assert_eq!(num, 5);
    }
}

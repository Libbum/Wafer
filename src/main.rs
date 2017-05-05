extern crate ansi_term;
extern crate term_size;

#[macro_use]
extern crate serde_derive;

use ansi_term::Colour::Blue;
use config::Config;

include!(concat!(env!("OUT_DIR"), "/version.rs"));

mod config;

fn main() {

    let mut sha = sha();
    let mut term_width = 100;
    if let Some((width, _)) = term_size::dimensions() {
        if width <= 97 {
            sha = short_sha();
        }
        if width <= 70 {
            term_width = 70;
        } else if width < term_width {
            term_width = width;
        }
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

    let config = Config::load();
    config.print(term_width);

}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder() {
        let num = 5;
        assert_eq!(num, 5);
    }
}

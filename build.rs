extern crate vergen;

use vergen::vergen;

fn main() {
    let mut flags = vergen::OutputFns::all();
    flags.toggle(vergen::COMMIT_DATE);
    flags.toggle(vergen::NOW);
    flags.toggle(vergen::SEMVER);
    flags.toggle(vergen::SHORT_NOW);
    flags.toggle(vergen::TARGET);
    assert!(vergen(flags).is_ok());
}

use std::env;
use std::fs;
use std::path::Path;

use quilting_fe_fixtures::{decode, fixed_raster_source};

fn main() {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    let [fixture, output] = arguments.as_slice() else {
        eprintln!("usage: generate-classic-quilting-fixed-raster <fixture.cqa> <output.fe>");
        std::process::exit(2);
    };
    let fixture_path = Path::new(fixture);
    let bytes = fs::read(fixture_path).expect("read checked atlas fixture");
    let artifact = decode(&bytes).expect("decode checked atlas fixture");
    let label = fixture_path
        .file_name()
        .and_then(|name| name.to_str())
        .expect("UTF-8 fixture file name");
    let source = fixed_raster_source::render(&artifact, label, 0)
        .expect("expand first atlas patch into fixed Fe topology");
    fs::write(output, source).expect("write generated Fe topology");
}

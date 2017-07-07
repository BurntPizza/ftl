
extern crate ftl;

extern crate structopt;
#[macro_use]
extern crate structopt_derive;

use structopt::*;

use std::fs::File;
use std::io::prelude::*;


#[derive(StructOpt)]
struct Opt {
    #[structopt(help = "Input file")]
    file: String,
}

fn main() {
    let Opt { file } = Opt::from_args();

    let mut file = File::open(file).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    let program = ftl::parser::parse_program(&*contents).unwrap();
    let p = ftl::compiler::compile(&program);
    println!("{}", p);
}

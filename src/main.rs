
extern crate petgraph as pg;
extern crate structopt;
#[macro_use]
extern crate structopt_derive;

use structopt::*;

use std::fs::File;
use std::io::prelude::*;

pub mod ftl_parser;
pub mod compiler;

#[derive(StructOpt)]
struct Opt {
    #[structopt(help = "Input file")]
    file: String
}

fn main() {
    let Opt {
        file,
    } = Opt::from_args();

    let mut file = File::open(file).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    let program = ftl_parser::parse_program(&*contents).unwrap();
    // println!("Program: \n{:#?}", program);
    let code = program.compile();
    let asm = code.assemble();
    println!("{}", asm);
}

#[derive(Debug)]
pub struct Program(Vec<Statement>);

impl Program {
    fn compile(&self) -> compiler::Code {
        compiler::compile(self)
    }
}

#[derive(Debug)]
pub enum Expr {
    Var(String),
    I64(i64),
    Read,
}

#[derive(Debug)]
pub enum Statement {
    VarDecl(String, Expr),
    Print(Expr),
    Switch(Switch),
    Block(Vec<Statement>),
    Break,
}

#[derive(Debug)]
pub struct Case(i64, Statement);

#[derive(Debug)]
pub struct Switch {
    arg: Box<Expr>,
    cases: Vec<Case>,
}

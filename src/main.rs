
extern crate ftl;

extern crate structopt;
#[macro_use]
extern crate structopt_derive;

use ftl::{parser, analysis};
use ftl::compiler::{self, Cfg};

use structopt::*;
use std::fs::File;
use std::io::prelude::*;


#[derive(StructOpt)]
struct Opt {
    #[structopt(help = "Input file")]
    file: String,
    #[structopt(short = "d")]
    debug: Option<String>,
}

fn main() {
    let Opt { file, debug } = Opt::from_args();

    let mut file = File::open(file).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    let contents: String = contents.lines().filter(|l| !l.starts_with("//")).collect();

    let program = parser::parse_program(&*contents).unwrap();
    let cfg = compiler::compile(&program);

    if let Some(s) = debug {
        for &(ds, f) in DEBUG_TABLE.iter() {
            if ds == s {
                f(&cfg)
            }
        }

        println!(
            "error: debug option must be one of: {:?}",
            DEBUG_TABLE.iter().map(|&(s, _)| s).collect::<Vec<_>>()
        );
        std::process::exit(1);
    }

    let asm = compiler::codegen(cfg);
    println!("{}", asm);
}


const DEBUG_TABLE: &[(&str, fn(&Cfg) -> !)] = &[
    ("ig", debug_ig),
    ("cfg", debug_cfg),
    ("rm", debug_register_mapping),
    ("lv", debug_liveness),
];


fn debug_ig(cfg: &Cfg) -> ! {
    let lv = analysis::live_variables(cfg);
    let ig = compiler::interference_graph(cfg, &lv);
    ftl::utils::print_graph(&ig.map(|_, n| n, |_, _| ""));
    std::process::exit(0);
}

fn debug_cfg(cfg: &Cfg) -> ! {
    ftl::utils::print_graph(cfg);
    std::process::exit(0);
}

fn debug_register_mapping(cfg: &Cfg) -> ! {
    let lv = analysis::live_variables(cfg);
    let ig = compiler::interference_graph(cfg, &lv);
    let rm = compiler::allocate_registers(ig);
    let mut rm: Vec<_> = rm.into_iter().collect();
    rm.sort_by_key(|&(r, _)| r);
    let rm = rm.into_iter().map(|(r, m)| format!("{}: {}", r, m)).fold(
        String::new(),
        |mut acc, e| {
            acc += &*e;
            acc += "\n";
            acc
        },
    );

    println!("{}", rm);

    std::process::exit(0);
}

fn debug_liveness(cfg: &Cfg) -> ! {
    let lv = analysis::live_variables(cfg);
    for n in cfg.node_indices() {
        println!("Node: {}", n.index());
        let mut v: Vec<_> = lv.internal_liveness(cfg, n).into_iter().collect();
        v.sort_by_key(|&(i, _)| i);
        for (i, live_set) in v {
            let mut ls: Vec<_> = live_set.into_iter().collect();
            ls.sort();
            println!("{}: {:?}", i, ls);
        }
    }

    std::process::exit(0);
}


extern crate dataflow;


#[macro_use]
extern crate strum_macros;
extern crate strum;
extern crate itertools;
extern crate petgraph as pg;

use dataflow::{Forward, Backward, May};
use strum::IntoEnumIterator;
use itertools::*;

pub mod ast;
pub mod parser;
pub mod compiler;

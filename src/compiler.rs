
use pg::prelude::*;

use std::cmp::max;
use std::fmt::{self, Display, Formatter};

use ::*;

#[derive(Debug)]
pub struct Code {
    entry: NodeIndex,
    ir: Graph<Ir, Edge>,
}

impl Code {
    pub fn assemble(&self) -> String {
        let g = &self.ir;
        let mut p = String::from("\tbits 64\n\tglobal _start\n");
        let mut data = String::from("\tsection .data\n");
        let mut text = String::from("\tsection .text\n_start:\n");
        let mut i = self.entry;

        loop {
            let n = &g[i];

            match *n {
                Ir::BB(ref ins) => {
                    for inst in ins {
                        match *inst {
                            Ins::AddImmI64(reg, val) => text.push_str(&format!("\tadd {}, {}\n", reg, val)),
                            Ins::SubImmI64(reg, val) => text.push_str(&format!("\tsub {}, {}\n", reg, val)),
                            Ins::Push(reg) => text.push_str(&format!("\tpush {}\n", reg)),
                            Ins::LoadImmI64(reg, val) => {
                                text.push_str(&format!("\tmov {}, {}\n", reg, val));
                            }
                            Ins::MovRegReg(dest, src) => {
                                text.push_str(&format!("\tmov {}, {}\n", dest, src));
                            }
                            Ins::Store{base, offset, src} => {
                                text.push_str(&format!("\tmov [{} + {}], {}\n", base, offset, src));
                            }
                            Ins::Syscall => text.push_str("\tsyscall\n"),
                        }
                    }
                }
            }

            if i == self.entry {
                let s = format!(
                    "\tmov rax, 60\n\
                     \txor rdi, rdi\n\
                     \tsyscall\n"
                );
                text.push_str(&s);
            }

            if g.edges(i).count() == 0 {
                break;
            } else {
                unimplemented!()
            }
        }

        p += &*data;
        p += &*text;
        p
    }
}

#[derive(Debug)]
enum Edge {
    Jmp,
}

#[derive(Debug, Copy, Clone)]
pub enum Ins {
    LoadImmI64(Reg, i64),
    Store{base:Reg, offset:i64, src:Reg},
    Push(Reg),
    AddImmI64(Reg, i64),
    SubImmI64(Reg, i64),
    MovRegReg(Reg, Reg),
    Syscall,
}

#[derive(Debug, Copy, Clone)]
pub enum Reg {
    Rax,
    Al,
    Rbx,
    Rcx,
    Rdx,

    Rsi,
    Rdi,
    Rsp,

    R8,
}

impl Display for Reg {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let s = match *self {
            Reg::Al => "al",
            Reg::Rax => "rax",
            Reg::Rbx => "rbx",
            Reg::Rcx => "rcx",
            Reg::Rdx => "rdx",
            Reg::Rsi => "rsi",
            Reg::Rdi => "rdi",
            Reg::Rsp => "rsp",
            Reg::R8 => "r8",
        };

        write!(f, "{}", s)
    }
}

#[derive(Debug)]
pub enum Ir {
    BB(Vec<Ins>),
}

pub fn compile(p: &Program) -> Code {
    let mut g = Graph::new();
    let entry = compile_statement(&p.0, &mut g);

    Code { entry, ir: g }
}

fn compile_statement(s: &Statement, g: &mut Graph<Ir, Edge>) -> NodeIndex {
    match *s {
        Statement::Switch(Switch { ref arg, ref cases }) => unimplemented!(),
        Statement::Break => unimplemented!(),
        Statement::Print(ref e) => {
            let mut ins = vec![];

            match *e {
                Expr::I64(v) => {
                    let s = format!("{}\n", v);
                    let stack_space = max(1, s.len() as i64 / 8) * 8;
                    ins.push(Ins::SubImmI64(Reg::Rsp, stack_space));
                    
                    for (i, b) in s.bytes().enumerate() {
                        ins.push(Ins::LoadImmI64(Reg::Rax, b as i64));
                        ins.push(Ins::Store{ base: Reg::Rsp, offset: i as i64, src: Reg::Al });
                    }
                    
                    ins.push(Ins::LoadImmI64(Reg::Rax, 1));
                    ins.push(Ins::LoadImmI64(Reg::Rdi, 1));
                    ins.push(Ins::MovRegReg(Reg::Rsi, Reg::Rsp));
                    ins.push(Ins::LoadImmI64(Reg::Rdx, s.len() as i64));
                    ins.push(Ins::Syscall);
                    ins.push(Ins::AddImmI64(Reg::Rsp, stack_space));
                }
                Expr::Read => unimplemented!(),
            }

            let node = Ir::BB(ins);
            g.add_node(node)
        }
        Statement::Block(ref stmts) => unimplemented!(),
    }
}

fn compile_expr(e: &Expr, ins: &mut Vec<Ins>) {}

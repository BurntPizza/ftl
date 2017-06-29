
use pg::prelude::*;

use std::cmp::max;
use std::fmt::{self, Display, Formatter};

use std::collections::{HashMap, HashSet};

use ::*;

#[derive(Debug)]
pub struct Code {
    entry: NodeIndex,
    ir: Graph<Ir, Edge>,
    fns: HashMap<String, NodeIndex>,
    label_counter: usize,
}
impl Code {
    fn make_fn_available<S>(&mut self, s: S, f: fn() -> Ir)
    where
        S: Into<String>,
    {
        let s = s.into();

        if !self.fns.contains_key(&s) {
            let idx = self.ir.add_node(f());
            self.fns.insert(s, idx);
        }
    }

    pub fn assemble(&self) -> String {
        let mut p = StringBuilder::new();

        p.indent();
        p.line("bits 64");
        p.line("global _start");

        let mut data = StringBuilder::new();
        data.indent();
        data.line("section .data");

        let mut text = StringBuilder::new();
        text.indent();
        text.line("section .text");

        let mut stack = vec![self.entry];
        let mut visited = HashSet::new();

        while let Some(i) = stack.pop() {
            if visited.contains(&i) {
                continue;
            }

            visited.insert(i);

            let node = &self.ir[i];

            match *node {
                Ir::BB(ref bb) => {
                    self.compile_bb(bb, &mut text);
                }
            }

            for i in self.ir.edges(i).filter(|e| *e.weight() == Edge::DependsOn) {
                stack.push(i.target());
            }
        }

        let mut p = p.into();
        p += &*data.into();
        p += &*text.into();
        p
    }

    fn compile_bb(&self, bb: &BB, block: &mut StringBuilder) {
        let BB { ref label, ref ins } = *bb;
        block.label(label);
        for ins in ins {
            self.compile_ins(*ins, block);
        }
    }

    fn compile_ins(&self, ins: Ins, block: &mut StringBuilder) {
        match ins {
            Ins::AddImmI64(reg, val) => {
                block.line(format!("add {}, {}", reg, val));
            }
            Ins::SubImmI64(reg, val) => {
                block.line(format!("sub {}, {}", reg, val));
            }
            Ins::Push(reg) => {
                block.line(format!("push {}", reg));
            }
            Ins::LoadImmI64(reg, val) => {
                block.line(format!("mov {}, {}", reg, val));
            }
            Ins::MovRegReg(dest, src) => {
                block.line(format!("mov {}, {}", dest, src));
            }
            Ins::Store { base, offset, src } => {
                block.line(format!("mov [{} + {}], {}", base, offset, src));
            }
            Ins::Syscall => {
                block.line("syscall");
            }
            Ins::Call(s) => {
                block.line(format!("call {}", s));
            }
            Ins::Ret => {
                block.line("ret");
            }
            Ins::Jmp(s) => {
                block.line(format!("jmp {}", s));
            }
            Ins::Xor(dest, src) => {
                block.line(format!("xor {}, {}", dest, src));
            }
        }
    }
}


#[derive(Debug, PartialEq)]
enum Edge {
    DependsOn,
}

#[derive(Debug, Copy, Clone)]
pub enum Ins {
    LoadImmI64(Reg, i64),
    Store { base: Reg, offset: i64, src: Reg },
    Push(Reg),
    AddImmI64(Reg, i64),
    SubImmI64(Reg, i64),
    MovRegReg(Reg, Reg),
    Syscall,
    Call(&'static str),
    Ret,
    Jmp(&'static str),
    Xor(Reg, Reg),
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
    BB(BB),
}

#[derive(Debug)]
pub struct BB {
    label: String,
    ins: Vec<Ins>,
}

pub fn compile(p: &Program) -> Code {
    let mut code = Code {
        entry: NodeIndex::new(0),
        ir: Graph::new(),
        fns: HashMap::new(),
        label_counter: 0,
    };

    let entry_node = Ir::BB(BB {
        label: "_start:".into(),
        ins: vec![],
    });

    code.entry = code.ir.add_node(entry_node);
    code.make_fn_available("exit", compile_exit);

    for stmt in &p.0 {
        compile_statement(stmt, code.entry, &mut code);
    }

    match code.ir[code.entry] {
        Ir::BB(BB { ref mut ins, .. }) => {
            ins.push(Ins::Jmp("exit"));
        }
    }

    code.ir.add_edge(
        code.entry,
        code.fns["exit"],
        Edge::DependsOn,
    );

    code
}

fn compile_statement(s: &Statement, current_node: NodeIndex, code: &mut Code) {
    match *s {
        Statement::Switch(Switch { ref arg, ref cases }) => unimplemented!(),
        Statement::Break => unimplemented!(),
        Statement::Print(ref e) => {
            code.make_fn_available("print", compile_print);
            {
                let ins = match code.ir[current_node] {
                    Ir::BB(BB { ref mut ins, .. }) => ins,
                };

                match *e {
                    Expr::Var(ref s) => unimplemented!(),
                    Expr::I64(v) => {
                        let s = format!("{}\n", v);
                        let stack_space = max(1, s.len() as i64 / 8) * 8;
                        ins.push(Ins::SubImmI64(Reg::Rsp, stack_space));

                        for (i, b) in s.bytes().enumerate() {
                            ins.push(Ins::LoadImmI64(Reg::Rax, b as i64));
                            ins.push(Ins::Store {
                                base: Reg::Rsp,
                                offset: i as i64,
                                src: Reg::Al,
                            });
                        }

                        ins.push(Ins::MovRegReg(Reg::Rdi, Reg::Rsp));
                        ins.push(Ins::LoadImmI64(Reg::Rsi, s.len() as i64));
                        ins.push(Ins::Call("print"));
                        ins.push(Ins::AddImmI64(Reg::Rsp, stack_space));
                    }
                    Expr::Read => unimplemented!(),
                }
            }

            code.ir.add_edge(
                current_node,
                code.fns["print"],
                Edge::DependsOn,
            );
        }
        Statement::VarDecl(ref s, ref e) => unimplemented!(),
        Statement::Block(ref stmts) => unimplemented!(),
    }
}

fn compile_expr(e: &Expr, ins: &mut Vec<Ins>) {}

fn compile_print() -> Ir {
    let mut ins = vec![];

    ins.push(Ins::MovRegReg(Reg::Rdx, Reg::Rsi));
    ins.push(Ins::MovRegReg(Reg::Rsi, Reg::Rdi));
    ins.push(Ins::LoadImmI64(Reg::Rax, 1));
    ins.push(Ins::MovRegReg(Reg::Rdi, Reg::Rax));
    ins.push(Ins::Syscall);
    ins.push(Ins::Ret);

    Ir::BB(BB {
        label: "print:".into(),
        ins,
    })
}

fn compile_exit() -> Ir {
    let mut ins = vec![];

    ins.push(Ins::LoadImmI64(Reg::Rax, 60));
    ins.push(Ins::Xor(Reg::Rdi, Reg::Rdi));
    ins.push(Ins::Syscall);

    Ir::BB(BB {
        label: "exit:".into(),
        ins,
    })
}

struct StringBuilder {
    string: String,
    indent_level: u32,
}

impl StringBuilder {
    fn into(self) -> String {
        self.string
    }
    fn new() -> Self {
        StringBuilder {
            string: String::new(),
            indent_level: 0,
        }
    }

    fn line<S>(&mut self, s: S)
    where
        S: AsRef<str>,
    {
        for _ in 0..self.indent_level * 4 {
            self.string.push(' ');
        }
        self.string.push_str(s.as_ref());
        self.string.push('\n');
    }

    fn label<S>(&mut self, s: S)
    where
        S: AsRef<str>,
    {
        self.string.push_str(s.as_ref());
        self.string.push('\n');
    }

    fn extend<S>(&mut self, s: S)
    where
        S: AsRef<str>,
    {
        self.string += s.as_ref();
    }

    fn indent(&mut self) {
        self.indent_level += 1;
    }

    fn unindent(&mut self) {
        self.indent_level = self.indent_level.checked_sub(1).unwrap();
    }
}

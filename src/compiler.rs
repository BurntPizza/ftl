
use pg::prelude::*;

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

        splice_graph(&self.ir, self.entry, &mut text);

        let mut p = p.into();
        p += &*data.into();
        p += &*text.into();
        p
    }
}

fn splice_graph(g: &Graph<Ir, Edge>, entry: NodeIndex, block: &mut StringBuilder) {
    if g.node_count() == 0 {
        return;
    }

    let mut visited = HashSet::new();
    let mut stack = vec![entry];

    while let Some(i) = stack.pop() {
        if visited.contains(&i) {
            continue;
        }

        visited.insert(i);

        let node = &g[i];

        match *node {
            Ir::BB(ref bb) => {
                compile_bb(bb, block);
            }
            Ir::Graph(ref g) => {
                splice_graph(g, NodeIndex::new(0), block);
            }
        }


        for i in g.edges(i) {
            stack.push(i.target());
        }
    }
}

fn compile_bb(bb: &BB, block: &mut StringBuilder) {
    let BB { ref label, ref ins } = *bb;
    block.label(label);
    for ins in ins {
        compile_ins(*ins, block);
    }
}

fn compile_ins(ins: Ins, block: &mut StringBuilder) {
    match ins {
        Ins::AddImmI64(reg, val) => block.line(format!("add {}, {}", reg, val)),
        Ins::SubImmI64(reg, val) => block.line(format!("sub {}, {}", reg, val)),
        Ins::Push(reg) => block.line(format!("push {}", reg)),
        Ins::Pop(reg) => block.line(format!("pop {}", reg)),
        Ins::LoadImmI64(reg, val) => block.line(format!("mov {}, {}", reg, val)),
        Ins::MovRegReg(dest, src) => block.line(format!("mov {}, {}", dest, src)),
        Ins::Store(store) => {
            match store {
                Store::ImmReg { base, offset, src } => {
                    block.line(format!("mov [{} + {}], {}", base, offset, src));
                }
                Store::RegReg { base, offset, src } => {
                    block.line(format!("mov [{} + {}], {}", base, offset, src));
                }
                Store::RegImm {
                    base,
                    offset,
                    val,
                    size,
                } => {
                    block.line(format!("mov {} [{} + {}], {}", size, base, offset, val));
                }
                Store::ImmImm {
                    base,
                    offset,
                    val,
                    size,
                } => {
                    block.line(format!("mov {} [{} + {}], {}", size, base, offset, val));
                }
            }
        }
        Ins::Syscall => block.line("syscall"),
        Ins::Call(s) => block.line(format!("call {}", s)),
        Ins::Ret => block.line("ret"),
        Ins::J(cc, label) => block.line(format!("j{} {}", cc, label)),
        Ins::Xor(dest, src) => block.line(format!("xor {}, {}", dest, src)),
        Ins::Cqo => block.line("cqo"),
        Ins::Idiv(reg) => block.line(format!("idiv {}", reg)),
        Ins::Inc(reg) => block.line(format!("inc {}", reg)),
        Ins::Dec(reg) => block.line(format!("dec {}", reg)),
        Ins::ShrImm(reg, v) => block.line(format!("shr {}, {}", reg, v)),
        Ins::Test(a, b) => block.line(format!("test {}, {}", a, b)),
        Ins::AndImm(a, b) => block.line(format!("and {}, {}", a, b)),
        Ins::Cmov(cc, a, b) => block.line(format!("cmov{} {}, {}", cc, a, b)),
        Ins::SarImm(a, b) => block.line(format!("sar {}, {}", a, b)),
        Ins::Cmp(a, b) => block.line(format!("cmp {}, {}", a, b)),
        Ins::Sub(a, b) => block.line(format!("sub {}, {}", a, b)),
        Ins::Add(a, b) => block.line(format!("add {}, {}", a, b)),
        Ins::Imul(a, b) => block.line(format!("imul {}, {}", a, b)),
    }
}


#[derive(Debug, PartialEq)]
pub enum Edge {
    DependsOn,
    Adj,
}

#[derive(Debug, Copy, Clone)]
pub enum Size {
    Byte,
}

impl Display for Size {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let s = match *self {
            Size::Byte => "byte",
        };
        write!(f, "{}", s)
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Store {
    ImmReg { base: Reg, offset: i64, src: Reg },
    RegReg { base: Reg, offset: Reg, src: Reg },
    RegImm {
        base: Reg,
        offset: Reg,
        val: i64,
        size: Size,
    },
    ImmImm {
        base: Reg,
        offset: i64,
        val: i64,
        size: Size,
    },
}

#[derive(Debug, Copy, Clone)]
pub enum Ins {

    // TODO: remove most of this, make it abstract, this is the IR after all
    


    LoadImmI64(Reg, i64),
    Store(Store),
    Push(Reg),
    AddImmI64(Reg, i64),
    SubImmI64(Reg, i64),
    MovRegReg(Reg, Reg),
    Syscall,
    Call(&'static str),
    Ret,
    Xor(Reg, Reg),
    ShrImm(Reg, i64),
    Inc(Reg),
    Dec(Reg),
    Cqo,
    Idiv(Reg),
    Test(Reg, Reg),
    J(&'static str, &'static str),
    AndImm(Reg, i64),
    Cmp(Reg, Reg),
    Cmov(&'static str, Reg, Reg),
    Sub(Reg, Reg),
    SarImm(Reg, i64),
    Add(Reg, Reg),
    Pop(Reg),
    Imul(Reg, Reg),
}

#[derive(Debug, Copy, Clone)]
pub enum Reg {
    Pinned(&'static str),
    Sym(u32),
}

fn pinned(s: &'static str) -> Reg {
    Reg::Pinned(s)
}

fn fresh() -> Reg {
    use std::sync::atomic::*;
    static COUNTER: AtomicUsize = ATOMIC_USIZE_INIT;
    
    Reg::Sym(COUNTER.fetch_add(1, Ordering::SeqCst) as u32)
}

impl Display for Reg {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let s = match *self {
            Reg::Sym(n) => format!("${}", n),
            Reg::Pinned(s) => s.into(),
        };

        write!(f, "{}", s)
    }
}

#[derive(Debug)]
pub enum Ir {
    BB(BB),
    Graph(Graph<Ir, Edge>),
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
            ins.push(Ins::J("mp", "exit"));
        }
        Ir::Graph(_) => unimplemented!(),
    }


    // TODO: automate this as a separate pass?
    // scan blocks starting at entry, adding deps as encountered in jmps and calls, etc.
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
                    Ir::Graph(_) => unimplemented!(),
                };

                compile_expr(e, pinned("rdi"), ins);
                ins.push(Ins::Call("print"));
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

fn compile_expr(e: &Expr, reg: Reg, ins: &mut Vec<Ins>) {
    match *e {
        Expr::I64(v) => ins.push(Ins::LoadImmI64(reg, v)),
        Expr::Var(ref s) => unimplemented!(),
        Expr::Read => unimplemented!(),
        Expr::Add(ref a, ref b) => {
            compile_expr(a, reg, ins);
            let r = fresh();
            compile_expr(b, r, ins);
            ins.push(Ins::Add(reg, r));
        }
        Expr::Mult(ref a, ref b) => {
            compile_expr(a, reg, ins);
            let r = fresh();
            compile_expr(b, r, ins);
            ins.push(Ins::Imul(reg, r));
        }
    }
}

fn compile_print() -> Ir {
    let bb1 = BB {
        label: "print:".into(),
        ins: vec![
            Ins::Push(pinned("r12")),
            Ins::LoadImmI64(pinned("rcx"), 10),
            Ins::MovRegReg(pinned("rax"), pinned("rdi")),
            Ins::MovRegReg(pinned("rsi"), pinned("rdi")),
            Ins::ShrImm(pinned("rdi"), 63),
        ],
    };

    let bb2 = BB {
        label: "__print_loop1:".into(),
        ins: vec![
            Ins::Inc(pinned("rdi")),
            Ins::Cqo,
            Ins::Idiv(pinned("rcx")),
            Ins::Test(pinned("rax"), pinned("rax")),
            Ins::J("nz", "__print_loop1"),

            Ins::Inc(pinned("rdi")),
            Ins::MovRegReg(pinned("rax"), pinned("rsi")),
            Ins::LoadImmI64(pinned("r8"), 8),
            Ins::MovRegReg(pinned("r12"), pinned("rdi")),
            Ins::AndImm(pinned("r12"), 0xFFF0),
            Ins::Cmp(pinned("r8"), pinned("r12")),
            Ins::Cmov("a", pinned("r12"), pinned("r8")),
            Ins::Sub(pinned("rsp"), pinned("r12")),
            Ins::Dec(pinned("rdi")),
            Ins::Test(pinned("rax"), pinned("rax")),
            Ins::J("ns", "__print_skip_neg"),
            Ins::Store(Store::ImmImm {
                base: pinned("rsp"),
                offset: 0,
                val: 45,
                size: Size::Byte,
            }),
        ],
    };

    let bb3 = BB {
        label: "__print_skip_neg:".into(),
        ins: vec![Ins::MovRegReg(pinned("r9"), pinned("rdi"))],
    };

    let bb4 = BB {
        label: "__print_loop2:".into(),
        ins: vec![
            Ins::Dec(pinned("r9")),
            Ins::Cqo,
            Ins::Idiv(pinned("rcx")),
            Ins::MovRegReg(pinned("r8"), pinned("rdx")),
            Ins::SarImm(pinned("r8b"), 7),
            Ins::Xor(pinned("dl"), pinned("r8b")),
            Ins::Sub(pinned("dl"), pinned("r8b")),
            Ins::AddImmI64(pinned("dl"), 48),
            Ins::Store(Store::RegReg {
                base: pinned("rsp"),
                offset: pinned("r9"),
                src: pinned("dl"),
            }),
            Ins::Test(pinned("rax"), pinned("rax")),
            Ins::J("nz", "__print_loop2"),
            Ins::Store(Store::RegImm {
                base: pinned("rsp"),
                offset: pinned("rdi"),
                val: 10,
                size: Size::Byte,
            }),
            Ins::Inc(pinned("rdi")),
            Ins::MovRegReg(pinned("rdx"), pinned("rdi")),
            Ins::LoadImmI64(pinned("rax"), 1),
            Ins::LoadImmI64(pinned("rdi"), 1),
            Ins::MovRegReg(pinned("rsi"), pinned("rsp")),
            Ins::Syscall,
            Ins::Add(pinned("rsp"), pinned("r12")),
            Ins::Pop(pinned("r12")),
            Ins::Ret,
        ],
    };

    let mut g = Graph::new();
    let bb1 = g.add_node(Ir::BB(bb1));
    let bb2 = g.add_node(Ir::BB(bb2));
    let bb3 = g.add_node(Ir::BB(bb3));
    let bb4 = g.add_node(Ir::BB(bb4));

    g.add_edge(bb1, bb2, Edge::Adj);
    g.add_edge(bb2, bb3, Edge::Adj);
    g.add_edge(bb2, bb4, Edge::Adj);

    Ir::Graph(g)
}

fn compile_exit() -> Ir {
    let mut ins = vec![];

    ins.push(Ins::LoadImmI64(pinned("rax"), 60));
    ins.push(Ins::Xor(pinned("rdi"), pinned("rdi")));
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

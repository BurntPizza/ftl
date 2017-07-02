
use pg::prelude::*;

use std::fmt::{self, Display, Formatter};

use std::collections::{HashMap, HashSet};

use ::*;

#[derive(Debug, Clone)]
pub struct Block {
    label: String,
    ins: Vec<Inst>,
}

impl Display for Block {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut s = String::new();
        for i in &self.ins {
            s += &format!("\n  {:?}", i);
        }
        write!(f, "{}{}", self.label, s)
    }
}


pub type Cfg = pg::stable_graph::StableDiGraph<Block, Edge>;

#[derive(PartialEq, Debug, Copy, Clone)]
pub struct R(u32);

#[derive(PartialEq, Debug, Clone)]
pub enum Inst {
    // Nop,
    // a == b?
    Test(R, R),
    // a = b
    Assign(R, R),
    // pinned reg
    AssignFrom(R, &'static str),
    Load(R, i64),
    // a = b + c
    Add(R, R, R),
    Mult(R, R, R),
    Call(&'static str, Vec<R>),
    Jmp(&'static str),
}

type Edge = &'static str;

#[derive(Debug)]
pub struct Prog {
    cfg: Cfg,
}

impl Prog {
    pub fn from(ast: &Program) -> Self {
        let mut g = Cfg::new();
        let mut names = HashMap::new();

        let mut prev = None;

        for stmt in &ast.0 {
            let (begin, end) = c_statement(stmt, &mut names, &mut g);
            if let Some(p_end) = prev {
                g.add_edge(p_end, begin, "next_top");
            }
            prev = Some(end);
        }

        let exit = g.add_node(block(Inst::Jmp("exit")));
        if let Some(prev) = prev {
            g.add_edge(prev, exit, "top_exit");
        }

        contract(&mut g);
        // remove_nops(&mut g);

        {
            let mut gg = Graph::new();
            let mut map = HashMap::new();
            
            for i in g.node_indices() {
                let new_node = gg.add_node(g[i].clone());
                map.insert(i, new_node);
            }


            for i in g.node_indices() {
                for n in g.neighbors(i) {
                    let e = g.find_edge(i, n).unwrap();
                    gg.add_edge(map[&i], map[&n], g[e]);
                }
            }

            let dot = pg::dot::Dot::new(&gg);
            println!("{:#}", dot);
        }

        Prog { cfg: g }
    }
}

fn contract(g: &mut Cfg) {
    let mut stack = vec![NodeIndex::new(0)];
    let mut visited = HashSet::new();

    while let Some(i) = stack.pop() {
        visited.insert(i);

        while g.neighbors(i).count() == 1 {
            let child = g.neighbors(i).next().unwrap();
            if g.neighbors_directed(child, pg::Direction::Incoming).count() > 1 {
                break;
            }

            let mut neighbors = g.neighbors(child).detach();
            while let Some(n) = neighbors.next_node(&*g) {
                let e = g.find_edge(child, n).unwrap();
                let w = g[e];
                g.add_edge(i, n, w);
            }
            
            let mut ins = g.remove_node(child).unwrap().ins;
            g[i].ins.append(&mut ins);
        }

        for n in g.neighbors(i).filter(|i| !visited.contains(i)) {
            stack.push(n);
        }
    }

    let mut dfs = pg::visit::Dfs::new(&*g, NodeIndex::new(0));
    let mut counter = 0;

    while let Some(i) = dfs.next(&*g) {
        g[i].label = format!("L{}:", counter);
        counter += 1;
    }
}

// fn remove_nops(g: &mut Cfg) {
//     let mut dfs = pg::visit::Dfs::new(&*g, NodeIndex::new(0));
    
//     while let Some(i) = dfs.next(&*g) {
//         g[i].ins.retain(|i| *i != Inst::Nop);
//     }
// }

fn c_statement(
    stmt: &Statement,
    names: &mut HashMap<String, R>,
    g: &mut Cfg,
) -> (NodeIndex, NodeIndex) {

    fn c_stmt_inner(
        stmt: &Statement,
        names: &mut HashMap<String, R>,
        g: &mut Cfg,
        exit: Option<NodeIndex>,
    ) -> (NodeIndex, NodeIndex, bool) {
        match *stmt {
            Statement::VarDecl(ref sym, ref e) => {
                let r = fresh();
                names.insert(sym.to_owned(), r);
                let (node, b) = c_expr(e, g, names);
                let this = g.add_node(block(Inst::Assign(r, b)));
                g.add_edge(node, this, "assign");
                (node, this, false)
            }
            Statement::Print(ref e) => {
                let (node, r) = c_expr(e, g, names);
                let this = g.add_node(block(Inst::Call("print", vec![r])));
                g.add_edge(node, this, "print");
                (node, this, false)
            }
            Statement::Block(ref stmts) => {
                let mut prev = None;

                for stmt in stmts {
                    let (n_begin, n_end, was_broke) = c_stmt_inner(stmt, names, g, exit);

                    if let Some((_, p_end)) = prev {
                        g.add_edge(p_end, n_begin, "block_next");
                    }

                    prev = Some((prev.map(|(b, _)| b).unwrap_or(n_begin), n_end));

                    if was_broke {
                        if let Some((b, pe)) = prev {
                            return (b, pe, true);
                        }
                        unreachable!();
                    }
                }

                prev.map(|(b, pe)| (b, pe, false)).unwrap_or_else(|| {
                    let i = g.add_node(empty_block());
                    (i, i, false)
                })
            }
            Statement::Switch(Switch { ref arg, ref cases }) => {
                let (node, arg_r) = c_expr(arg, g, names);
                let exit = g.add_node(empty_block());

                let mut prev = node;
                let mut default = None;

                for case in cases {
                    match *case {
                        Case::Default(ref s) => {
                            default = Some(s);
                        }
                        Case::Case(v, ref s) => {
                            let (guard_node, guard_r) = c_expr(&Expr::I64(v), g, names);
                            let test_node = g.add_node(block(Inst::Test(arg_r, guard_r)));
                            let (begin, end, was_broke) = c_stmt_inner(s, names, g, Some(exit));
                            let c_end = g.add_node(empty_block());
                            g.add_edge(prev, guard_node, "switch_next");
                            g.add_edge(guard_node, test_node, "test");
                            g.add_edge(test_node, begin, "true");
                            g.add_edge(test_node, c_end, "false");
                            if !was_broke {
                                g.add_edge(end, c_end, "case_end");
                            }
                            prev = c_end;
                        }
                    }
                }

                if let Some(s) = default {
                    let (begin, end, _) = c_stmt_inner(s, names, g, Some(exit));
                    g.add_edge(prev, begin, "switch_next_default");
                    prev = end;
                }

                g.add_edge(prev, exit, "switch_exit");

                (node, exit, false)
            }
            Statement::Break => {
                let exit = exit.unwrap();
                let this = g.add_node(empty_block());
                g.add_edge(this, exit, "break_exit");
                (this, this, true)
            }
        }
    }

    let (a, b, _) = c_stmt_inner(stmt, names, g, None);
    (a, b)
}

fn c_expr(e: &Expr, g: &mut Cfg, names: &HashMap<String, R>) -> (NodeIndex, R) {
    fn bin_op(
        f: fn(R, R, R) -> Inst,
        a: &Expr,
        b: &Expr,
        g: &mut Cfg,
        names: &HashMap<String, R>,
    ) -> (NodeIndex, R) {
        let (node_a, r_a) = c_expr(a, g, names);
        let (node_b, r_b) = c_expr(b, g, names);
        g.add_edge(node_a, node_b, "binop1");
        let r = fresh();
        let this = g.add_node(block(f(r, r_a, r_b)));
        g.add_edge(node_b, this, "binop2");
        (this, r)
    };

    match *e {
        Expr::I64(v) => {
            let r = fresh();
            let this = g.add_node(block(Inst::Load(r, v)));
            (this, r)
        }
        Expr::Add(ref a, ref b) => bin_op(Inst::Add, a, b, g, names),
        Expr::Mult(ref a, ref b) => bin_op(Inst::Mult, a, b, g, names),
        Expr::Var(ref s) => {
            let r = fresh();
            let b = names[s];
            let this = g.add_node(block(Inst::Assign(r, b)));
            (this, r)
        }
        Expr::Read => {
            let r = fresh();
            let call_node = g.add_node(block(Inst::Call("read", vec![])));
            let node = g.add_node(block(Inst::AssignFrom(r, "rax")));
            g.add_edge(call_node, node, "read");
            (node, r)
        }
    }
}

fn block(i: Inst) -> Block {
    Block {
        label: "".into(),
        ins: vec![i],
    }
}

fn empty_block() -> Block {
    Block {
        label: "".into(),
        ins: vec![],
    }
}

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


// #[derive(Debug, PartialEq)]
// pub enum Edge {
//     DependsOn,
//     Adj,
// }

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

fn fresh() -> R {
    use std::sync::atomic::*;
    static COUNTER: AtomicUsize = ATOMIC_USIZE_INIT;

    R(COUNTER.fetch_add(1, Ordering::SeqCst) as u32)
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

// pub fn compile(p: &Program) -> Code {
//     let mut code = Code {
//         entry: NodeIndex::new(0),
//         ir: Graph::new(),
//         fns: HashMap::new(),
//         label_counter: 0,
//     };

//     let entry_node = Ir::BB(BB {
//         label: "_start:".into(),
//         ins: vec![],
//     });

//     code.entry = code.ir.add_node(entry_node);
//     code.make_fn_available("exit", compile_exit);

//     for stmt in &p.0 {
//         // compile_statement(stmt, code.entry, &mut code);
//     }

//     match code.ir[code.entry] {
//         Ir::BB(BB { ref mut ins, .. }) => {
//             ins.push(Ins::J("mp", "exit"));
//         }
//         Ir::Graph(_) => unimplemented!(),
//     }


//     // TODO: automate this as a separate pass?
//     // scan blocks starting at entry, adding deps as encountered in jmps and calls, etc.
//     code.ir.add_edge(
//         code.entry,
//         code.fns["exit"],
//         Edge::DependsOn,
//     );

//     code
// }

// fn compile_statement(s: &Statement, current_node: NodeIndex, code: &mut Code) {
//     match *s {
//         Statement::Switch(Switch { ref arg, ref cases }) => unimplemented!(),
//         Statement::Break => unimplemented!(),
//         Statement::Print(ref e) => {
//             code.make_fn_available("print", compile_print);
//             {
//                 let ins = match code.ir[current_node] {
//                     Ir::BB(BB { ref mut ins, .. }) => ins,
//                     Ir::Graph(_) => unimplemented!(),
//                 };

//                 compile_expr(e, pinned("rdi"), ins);
//                 ins.push(Ins::Call("print"));
//             }

//             code.ir.add_edge(
//                 current_node,
//                 code.fns["print"],
//                 Edge::DependsOn,
//             );
//         }
//         Statement::VarDecl(ref s, ref e) => unimplemented!(),
//         Statement::Block(ref stmts) => unimplemented!(),
//     }
// }

// fn compile_expr(e: &Expr, reg: Reg, ins: &mut Vec<Ins>) {
//     match *e {
//         Expr::I64(v) => ins.push(Ins::LoadImmI64(reg, v)),
//         Expr::Var(ref s) => unimplemented!(),
//         Expr::Read => unimplemented!(),
//         Expr::Add(ref a, ref b) => {
//             compile_expr(a, reg, ins);
//             let r = fresh();
//             compile_expr(b, r, ins);
//             ins.push(Ins::Add(reg, r));
//         }
//         Expr::Mult(ref a, ref b) => {
//             compile_expr(a, reg, ins);
//             let r = fresh();
//             compile_expr(b, r, ins);
//             ins.push(Ins::Imul(reg, r));
//         }
//     }
// }

// fn compile_print() -> Ir {
//     let bb1 = BB {
//         label: "print:".into(),
//         ins: vec![
//             Ins::Push(pinned("r12")),
//             Ins::LoadImmI64(pinned("rcx"), 10),
//             Ins::MovRegReg(pinned("rax"), pinned("rdi")),
//             Ins::MovRegReg(pinned("rsi"), pinned("rdi")),
//             Ins::ShrImm(pinned("rdi"), 63),
//         ],
//     };

//     let bb2 = BB {
//         label: "__print_loop1:".into(),
//         ins: vec![
//             Ins::Inc(pinned("rdi")),
//             Ins::Cqo,
//             Ins::Idiv(pinned("rcx")),
//             Ins::Test(pinned("rax"), pinned("rax")),
//             Ins::J("nz", "__print_loop1"),

//             Ins::Inc(pinned("rdi")),
//             Ins::MovRegReg(pinned("rax"), pinned("rsi")),
//             Ins::LoadImmI64(pinned("r8"), 8),
//             Ins::MovRegReg(pinned("r12"), pinned("rdi")),
//             Ins::AndImm(pinned("r12"), 0xFFF0),
//             Ins::Cmp(pinned("r8"), pinned("r12")),
//             Ins::Cmov("a", pinned("r12"), pinned("r8")),
//             Ins::Sub(pinned("rsp"), pinned("r12")),
//             Ins::Dec(pinned("rdi")),
//             Ins::Test(pinned("rax"), pinned("rax")),
//             Ins::J("ns", "__print_skip_neg"),
//             Ins::Store(Store::ImmImm {
//                 base: pinned("rsp"),
//                 offset: 0,
//                 val: 45,
//                 size: Size::Byte,
//             }),
//         ],
//     };

//     let bb3 = BB {
//         label: "__print_skip_neg:".into(),
//         ins: vec![Ins::MovRegReg(pinned("r9"), pinned("rdi"))],
//     };

//     let bb4 = BB {
//         label: "__print_loop2:".into(),
//         ins: vec![
//             Ins::Dec(pinned("r9")),
//             Ins::Cqo,
//             Ins::Idiv(pinned("rcx")),
//             Ins::MovRegReg(pinned("r8"), pinned("rdx")),
//             Ins::SarImm(pinned("r8b"), 7),
//             Ins::Xor(pinned("dl"), pinned("r8b")),
//             Ins::Sub(pinned("dl"), pinned("r8b")),
//             Ins::AddImmI64(pinned("dl"), 48),
//             Ins::Store(Store::RegReg {
//                 base: pinned("rsp"),
//                 offset: pinned("r9"),
//                 src: pinned("dl"),
//             }),
//             Ins::Test(pinned("rax"), pinned("rax")),
//             Ins::J("nz", "__print_loop2"),
//             Ins::Store(Store::RegImm {
//                 base: pinned("rsp"),
//                 offset: pinned("rdi"),
//                 val: 10,
//                 size: Size::Byte,
//             }),
//             Ins::Inc(pinned("rdi")),
//             Ins::MovRegReg(pinned("rdx"), pinned("rdi")),
//             Ins::LoadImmI64(pinned("rax"), 1),
//             Ins::LoadImmI64(pinned("rdi"), 1),
//             Ins::MovRegReg(pinned("rsi"), pinned("rsp")),
//             Ins::Syscall,
//             Ins::Add(pinned("rsp"), pinned("r12")),
//             Ins::Pop(pinned("r12")),
//             Ins::Ret,
//         ],
//     };

//     let mut g = Graph::new();
//     let bb1 = g.add_node(Ir::BB(bb1));
//     let bb2 = g.add_node(Ir::BB(bb2));
//     let bb3 = g.add_node(Ir::BB(bb3));
//     let bb4 = g.add_node(Ir::BB(bb4));

//     g.add_edge(bb1, bb2, Edge::Adj);
//     g.add_edge(bb2, bb3, Edge::Adj);
//     g.add_edge(bb2, bb4, Edge::Adj);

//     Ir::Graph(g)
// }

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

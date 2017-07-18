
use petgraph::{stable_graph, visit, Direction, Incoming, Undirected};
use itertools::*;
use petgraph::prelude::{NodeIndex, Graph};
use strum::IntoEnumIterator;

use mrst;

use ast::*;
use self::Edge::Debug;
use analysis;

use std::borrow::Cow;
use std::mem;
use std::sync::atomic::*;
use std::fmt::{self, Display, Formatter};
use std::collections::{HashMap, HashSet};

const JUMP_TABLE_ROW_SIZE: usize = 5;

#[derive(Debug, Clone)]
pub struct Block {
    idx: NodeIndex,
    label: String,
    pub ins: Vec<Inst>,
}

pub type Cfg = stable_graph::StableDiGraph<Block, Edge>;

#[derive(Hash, Eq, PartialEq, Copy, PartialOrd, Ord, Clone)]
pub enum R {
    R(u32),
    Pin(Pinned),
    Const(i64),
}

impl R {
    pub fn is_sym(&self) -> bool {
        match *self {
            R::R(_) => true,
            _ => false,
        }
    }
    pub fn is_const(&self) -> bool {
        match *self {
            R::Const(_) => true,
            _ => false,
        }
    }
    pub fn is_pinned(&self) -> bool {
        match *self {
            R::Pin(_) => true,
            _ => false,
        }
    }
}

fn fresh() -> R {

    static COUNTER: AtomicUsize = ATOMIC_USIZE_INIT;

    R::R(COUNTER.fetch_add(1, Ordering::SeqCst) as u32)
}

#[derive(EnumIter, Hash, Eq, PartialEq, Copy, PartialOrd, Ord, Clone)]
pub enum Pinned {
    Rax,
    Rbx,
    Rcx,
    Rdx,
    Rdi,
    Rsi,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
    Rbp,
}

#[derive(PartialEq, Debug, Copy, Clone)]
pub enum JumpType {
    Near,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Inst {
    // a == b?
    Test(R, R),
    // a = b
    Assign(R, R),
    // a = b + c
    Add(R, R, R),
    Mult(R, R, R),
    Call(&'static str, usize),
    Jmp(Cow<'static, str>, Option<JumpType>),
    //
    Cjmp(R, NodeIndex),
    Shr(R, R),
    And(R, R, R),
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Edge {
    True,
    False,
    Table(usize),
    Debug(&'static str),
}

impl Edge {
    fn is_true_false(&self) -> bool {
        match *self {
            Edge::True | Edge::False => true,
            _ => false,
        }
    }
}

pub fn compile(ast: &Program) -> Cfg {
    let mut g = Cfg::new();
    let mut names = HashMap::new();
    let mut label_exits = HashMap::new();

    let mut prev = None;

    for stmt in &ast.0 {
        let (begin, end) = c_statement(stmt, &mut names, &mut label_exits, &mut g);
        if let Some(p_end) = prev {
            g.add_edge(p_end, begin, Debug("next_top"));
        }
        prev = Some(end);
    }

    let exit = g.add_node(block(Inst::Jmp(Cow::Borrowed("exit"), None)));
    if let Some(prev) = prev {
        g.add_edge(prev, exit, Debug("top_exit"));
    }

    contract(&mut g);
    label_blocks(&mut g);
    // analysis::const_prop(&mut g);

    g
}

pub fn codegen(mut g: Cfg) -> String {
    let lv = analysis::live_variables(&g);
    let ig = interference_graph(&g, &lv);
    let ra = allocate_registers(ig);

    if cfg!(debug_assertions) {
        let mut missing = vec![];
        for r in g.node_indices()
            .flat_map(|n| &g[n].ins)
            .flat_map(|ins| vars_written(ins).into_iter().chain(vars_read(ins)))
            .filter(|r| r.is_sym())
        {
            if !ra.contains_key(&r) {
                missing.push(r);
            }
        }

        missing.sort();
        missing.dedup();

        assert_eq!(Vec::<R>::new(), missing);
    }

    // apply register allocation
    let start = g.node_indices().next().unwrap();
    let mut dfs = visit::Dfs::new(&g, start);
    while let Some(n) = dfs.next(&g) {
        for ins in g[n].ins.iter_mut() {
            match ins {
                &mut Inst::And(ref mut a, ref mut b, ref mut c) |
                &mut Inst::Mult(ref mut a, ref mut b, ref mut c) |
                &mut Inst::Add(ref mut a, ref mut b, ref mut c) => {
                    debug_assert!(!a.is_const());
                    if a.is_sym() {
                        *a = R::Pin(ra[a]);
                    }
                    if b.is_sym() {
                        *b = R::Pin(ra[b]);
                    }
                    if c.is_sym() {
                        *c = R::Pin(ra[c]);
                    }
                }
                &mut Inst::Test(ref mut a, ref mut b) => {
                    if a.is_sym() {
                        *a = R::Pin(ra[a]);
                    }
                    if b.is_sym() {
                        *b = R::Pin(ra[b]);
                    }
                }
                &mut Inst::Cjmp(ref mut a, ..) => {
                    debug_assert!(!a.is_const());
                    if a.is_sym() {
                        *a = R::Pin(ra[a]);
                    }
                }
                &mut Inst::Shr(ref mut a, ref mut b) |
                &mut Inst::Assign(ref mut a, ref mut b) => {
                    debug_assert!(!a.is_const());
                    if a.is_sym() {
                        *a = R::Pin(ra[a]);
                    }
                    if b.is_sym() {
                        *b = R::Pin(ra[b]);
                    }
                }
                &mut Inst::Call(..) |
                &mut Inst::Jmp(..) => {}
            }
        }
    }

    // emit

    let mut p = StringBuilder::new();
    p.line("bits 64");
    p.line("global _start");

    let mut data = StringBuilder::new();
    data.line("section .data");

    let mut text = StringBuilder::new();
    text.line("section .text");
    text.label("_start");

    let start = g.node_indices().next().unwrap();
    let mut stack = vec![start];
    let mut visited = HashSet::new();

    while let Some(n) = stack.pop() {
        if visited.contains(&n) {
            continue;
        }

        visited.insert(n);

        compile_bb(&g, n, &mut text);

        match g[n].ins.last() {
            Some(&Inst::Test(mut a, mut b)) => {
                assert_eq!(g.neighbors(n).count(), 2);
                assert!(g.edges(n).all(|e| e.weight().is_true_false()));
                let (mut near, mut far) = g.neighbors(n).next_tuple().unwrap();

                // could be smarter about this, e.g. with some look ahead
                // default heuristic: smaller block will be written next
                if g[near].ins.len() > g[far].ins.len() {
                    mem::swap(&mut near, &mut far);
                }

                let cc = g[g.find_edge(n, far).unwrap()] == Edge::True;

                if a == b || a.is_const() && b.is_const() {
                    // only one branch can ever be taken, so implicit branch there
                    let next = if cc == (a == b) { far } else { near };
                    stack.push(next);
                } else {
                    // only emit cmp here

                    if a.is_const() {
                        mem::swap(&mut a, &mut b);
                    }

                    debug_assert!(a.is_pinned());
                    debug_assert!(!b.is_sym());
                    text.line(format!("cmp {}, {}", a, b));

                    let jmp = if cc { "je" } else { "jne" };
                    let target = &g[far].label;
                    text.line(format!("{} {}", jmp, target));

                    stack.push(far);
                    stack.push(near);
                }
            }

            Some(&Inst::Cjmp(a, table_node)) => {
                debug_assert_eq!(g.neighbors(n).count(), 1);
                debug_assert_eq!(g.neighbors(n).next(), Some(table_node));
                assert!(a.is_pinned());

                let label = &g[table_node].label;

                text.line(format!(
                    "lea {0}, [{1} + {0} * {2}]",
                    a,
                    label,
                    JUMP_TABLE_ROW_SIZE
                ));
                text.line(format!("jmp {}", a));

                assert!(!visited.contains(&table_node));
                visited.insert(table_node);

                compile_bb(&g, table_node, &mut text);

                stack.extend(g.neighbors(table_node));
            }

            _ => {
                assert!(g.neighbors(n).count() <= 1);
                if let Some(n) = g.neighbors(n).next() {
                    stack.push(n);
                    if visited.contains(&n) {
                        // back edge, must jump
                        text.line(format!("jmp {}", g[n].label));
                    }
                }
            }
        }
    }

    link_builtins(&mut text);

    p.extend(data.into());
    p.extend(text.into());
    p.into()
}

fn compile_bb(g: &Cfg, idx: NodeIndex, asm: &mut StringBuilder) {
    let bb = &g[idx];
    asm.label(&bb.label);

    for ins in &bb.ins {
        match *ins {
            Inst::Add(a, mut b, mut c) => {
                debug_assert!(a.is_pinned());
                if b.is_const() {
                    mem::swap(&mut b, &mut c);
                }
                debug_assert!(b.is_pinned());
                debug_assert!(!c.is_sym());
                let s;

                if a == b || a == c {
                    if b == c {
                        s = format!("add {0}, {0}", a);
                    } else {
                        let r = if a == b { c } else { b };
                        s = format!("add {}, {}", a, r);
                    }
                } else {
                    s = format!("lea {}, [{} + {}]", a, b, c);
                };

                asm.line(s);
            }
            Inst::And(a, mut b, mut c) => {
                debug_assert!(a.is_pinned());
                if b.is_const() {
                    mem::swap(&mut b, &mut c);
                }
                debug_assert!(!b.is_sym());
                debug_assert!(!c.is_sym());

                if a == b || a == c {
                    if b != c {
                        let r = if a == b { c } else { b };
                        asm.line(format!("and {}, {}", a, r));
                    }
                } else {
                    unimplemented!()
                }
            }
            Inst::Shr(a, b) => {
                debug_assert!(a.is_pinned());
                debug_assert!(!b.is_sym());

                asm.line(format!("shr {}, {}", a, b));
            }
            Inst::Assign(a, b) => {
                debug_assert!(a.is_pinned());
                debug_assert!(!b.is_sym());

                if a != b {
                    asm.line(format!("mov {}, {}", a, b));
                }
            }
            Inst::Call(s, _) => asm.line(format!("call {}", s)),
            Inst::Jmp(ref s, t) => {
                asm.line(format!(
                    "jmp {} {}",
                    t.map_or("".to_owned(), |jt| jt.to_string()),
                    s
                ))
            }
            Inst::Mult(a, b, c) => {
                debug_assert!(a.is_pinned());
                debug_assert!(!b.is_sym());
                debug_assert!(!c.is_sym());
                asm.line(format!("lea {}, [{} * {}]", a, b, c));
            }
            Inst::Test(mut a, mut b) => {
                // defer
            }
            Inst::Cjmp(a, n) => {
                // defer
            }
        }
    }
}

fn contract(g: &mut Cfg) {
    let start = g.node_indices().next().unwrap();
    let mut stack = vec![start];
    let mut visited = HashSet::new();

    while let Some(i) = stack.pop() {
        visited.insert(i);

        while g.neighbors(i).filter(|n| !visited.contains(n)).count() == 1 {
            let child = g.neighbors(i)
                .filter(|n| !visited.contains(n))
                .next()
                .unwrap();
            if g.neighbors_directed(child, Direction::Incoming).count() > 1 ||
                g[child].label != ""
            {
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

    // formerly thread_jumps
    loop {
        let mut to_add = vec![];
        let mut changed = false;

        g.retain_nodes(|g, i| {
            if g[i].ins.is_empty() && g.neighbors(i).count() == 1 {
                let target = g.neighbors(i).next().unwrap();

                for n in g.neighbors_directed(i, Incoming) {
                    let w = g[g.find_edge(n, i).unwrap()];
                    to_add.push((n, target, w));
                    changed = true;
                }

                return false;
            }

            true
        });

        if !changed {
            break;
        }

        for (s, t, w) in to_add {
            g.add_edge(s, t, w);
        }
    }
}

fn label_blocks(g: &mut Cfg) {
    let start = g.node_indices().next().unwrap();
    let mut dfs = visit::Dfs::new(&*g, start);
    let mut counter = 0;

    while let Some(i) = dfs.next(&*g) {
        g[i].idx = i;
        if g[i].label == "" {
            g[i].label = format!("L{}", counter);
        }
        counter += 1;
    }
}

pub type InterferenceGraph = Graph<R, (), Undirected>;

pub fn interference_graph(g: &Cfg, lv: &analysis::LiveVariables) -> InterferenceGraph {
    let mut idx_mapping = HashMap::new();
    let mut ig = Graph::new_undirected();

    for node in g.node_indices() {
        for live_set in lv.internal_liveness(g, node).values() {
            for &r in live_set {
                idx_mapping.entry(r).or_insert_with(|| ig.add_node(r));
            }
            for (a, b) in live_set.iter().tuple_combinations() {
                ig.update_edge(idx_mapping[a], idx_mapping[b], ());
            }
        }
    }

    ig
}

pub fn allocate_registers(mut ig: InterferenceGraph) -> HashMap<R, Pinned> {
    let colors: HashSet<_> = Pinned::iter().collect();
    let mut mapping = HashMap::new();
    let k = colors.len();
    let mut stack = vec![];

    let num_sym = ig.node_indices().filter(|&n| ig[n].is_sym()).count();

    for _ in 0..num_sym {
        let node = ig.node_indices()
            .filter(|&n| ig[n].is_sym() && ig.neighbors(n).count() < k)
            .next()
            .or_else(|| ig.node_indices().filter(|&n| ig[n].is_sym()).next())
            .unwrap();

        assert!(ig[node].is_sym());

        let neighbors: Vec<R> = ig.neighbors(node).map(|n| ig[n]).collect();
        let r = ig.remove_node(node).unwrap();

        if r.is_sym() {
            stack.push((r, neighbors));
        }
    }

    while let Some((node, neighbors)) = stack.pop() {
        let neighbor_colors = neighbors
            .into_iter()
            .map(|r| match r {
                R::R(_) => mapping[&r],
                R::Pin(p) => p,
                R::Const(_) => unreachable!(),
            })
            .collect();
        let color = match node {
            R::Pin(p) => p,
            _ => {
                *colors.difference(&neighbor_colors).next().expect(
                    "failed to K-color",
                )
            }
        };
        mapping.insert(node, color);
    }

    mapping
}

pub fn vars_read(ins: &Inst) -> HashSet<R> {
    let mut set = HashSet::new();
    match *ins {
        Inst::And(_, a, b) |
        Inst::Add(_, a, b) |
        Inst::Mult(_, a, b) |
        Inst::Test(a, b) => {
            if !a.is_const() {
                set.insert(a);
            }
            if !b.is_const() {
                set.insert(b);
            }
        }
        Inst::Shr(_, b) |
        Inst::Assign(_, b) => {
            if !b.is_const() {
                set.insert(b);
            }
        }
        Inst::Call(_, num_args) => {
            use self::Pinned::*;
            // SysV: RDI, RSI, RDX, RCX
            assert!(num_args <= 4); // only supporting up to 4 args for now
            set.extend([Rdi, Rsi, Rdx, Rcx].iter().cloned().take(num_args).map(
                R::Pin,
            ));

            // caller save regs
            set.extend(
                [Rax, Rcx, Rdx, Rsi, Rdi, R8, R9, R10, R11]
                    .iter()
                    .cloned()
                    .map(R::Pin),
            );
        }
        Inst::Cjmp(a, _) => {
            if !a.is_const() {
                set.insert(a);
            }
        }
        Inst::Jmp(..) => {}
    }
    set
}

pub fn vars_written(ins: &Inst) -> HashSet<R> {
    let mut set = HashSet::new();

    match *ins {
        Inst::Shr(r, ..) |
        Inst::And(r, ..) |
        Inst::Cjmp(r, ..) |
        Inst::Add(r, ..) |
        Inst::Assign(r, _) |
        Inst::Mult(r, ..) => {
            set.insert(r);
        }
        Inst::Call(..) => {
            use self::Pinned::*;
            set.extend(
                [Rax, Rcx, Rdx, Rsi, Rdi, R8, R9, R10, R11]
                    .iter()
                    .cloned()
                    .map(R::Pin),
            );
        }
        Inst::Jmp(..) | Inst::Test(..) => {}
    }
    set
}

fn c_statement(
    stmt: &Statement,
    names: &mut HashMap<String, R>,
    label_exits: &mut HashMap<String, NodeIndex>,
    g: &mut Cfg,
) -> (NodeIndex, NodeIndex) {

    let (a, b, _) = c_stmt_inner(stmt, names, label_exits, g, None);
    (a, b)
}


fn c_stmt_inner(
    stmt: &Statement,
    names: &mut HashMap<String, R>,
    label_exits: &mut HashMap<String, NodeIndex>,
    g: &mut Cfg,
    exit: Option<NodeIndex>,
        // First node, last node, whether or not to break out
) -> (NodeIndex, NodeIndex, bool) {
    match *stmt {
        Statement::VarDecl(ref sym, ref e) => {
            let r = fresh();
            assert!(
                names.insert(sym.to_owned(), r).is_none(),
                "Cannot reuse variable name: {}",
                sym
            );
            let (node, b) = c_expr(e, g, names);
            g[node].ins.push(Inst::Assign(r, b));
            (node, node, false)
        }
        Statement::Print(ref e) => {
            let (node, r) = c_expr(e, g, names);
            g[node].ins.push(Inst::Assign(R::Pin(Pinned::Rdi), r));
            g[node].ins.push(Inst::Call("print", 1));
            (node, node, false)
        }
        Statement::Block(ref stmts) => {
            let mut prev = None;

            for stmt in stmts {
                let (n_begin, n_end, was_broke) = c_stmt_inner(stmt, names, label_exits, g, exit);

                if let Some((_, p_end)) = prev {
                    g.add_edge(p_end, n_begin, Debug("block_next"));
                }

                prev = Some((prev.map_or(n_begin, |(b, _)| b), n_end));

                if was_broke {
                    if let Some((b, pe)) = prev {
                        return (b, pe, true);
                    }
                    unreachable!();
                }
            }

            prev.map_or_else(
                || {
                    let i = g.add_node(empty_block());
                    (i, i, false)
                },
                |(b, pe)| (b, pe, false),
            )
        }
        Statement::Switch(Switch { ref arg, ref cases }) => {
            /*

                switch (x) {
                    case 0: { thing0(); break; }
                    case 1: { thing1(); break; }
                    default: blah();
                    case 2: thing2();
                    case 4: { thing4(); break; }
                }

                // branch node
                hash:
                   jmp table + R0 * 5

                table:
                    jmp L0
                    jmp L1
                    jmp L2
                    jmp exit // hole in table
                    jmp L3

                // TODO: make sure the ordering stays the same, at least for fallthrough intervals

                // leaves
                L0: thing0()
                    jmp exit
                L1: thing1()
                    jmp exit
                D0: blah()   // no break
                L2: thing2() // no break
                L3: thing3()
                    jmp exit
                exit:

                 */
            use mrst::{Tree, Marker, HashFn};
            use mrst::methods::*;

            let mut default = None;
            let cases: Vec<(usize, &Statement)> = cases
                .into_iter()
                .filter_map(|c| match *c {
                    Case::Case(v, ref st) => Some((v as usize, st.as_ref().unwrap())),
                    Case::Default(ref st) => {
                        default = Some(st);
                        None
                    }
                })
                .collect();

            let (arg_block, arg_r) = c_expr(arg, g, names);
            let exit = g.add_node(empty_block());

            let tree = Tree::new(&*cases, &[&SubLow, &ShiftMask]);

            let mut stack = vec![(&tree, (arg_block, exit))];

            while let Some(state) = stack.pop() {
                match state {
                    (&Tree::Branch {
                         ref children,
                         ref hash_fn,
                     },
                     (_, exit)) => {
                        let hash_r = fresh();

                        let mut ins = vec![Inst::Assign(hash_r, arg_r)];
                        let num_table_entries = match *hash_fn {
                            HashFn::SubLow(f) => {
                                if f.bias != 0 {
                                    ins.push(Inst::Add(hash_r, hash_r, R::Const(-(f.bias as i64))));
                                }
                                f.max
                            }
                            HashFn::ShiftMask(Window { l, r }) => {
                                let width = 1 + l - r;
                                let mask = ((1 << width) - 1) as i64;
                                if r != 0 {
                                    ins.push(Inst::Shr(hash_r, R::Const(r as i64)));
                                }
                                if mask != -1 {
                                    ins.push(Inst::And(hash_r, hash_r, R::Const(mask)));
                                }
                                1 << width
                            }
                            HashFn::ClzSub(f) => unimplemented!(),
                        };

                        let compute_jump = g.add_node(Block {
                            ins,
                            ..empty_block()
                        });

                        static CASE_ENTRY_LABEL_COUNTER: AtomicUsize = ATOMIC_USIZE_INIT;
                        static TABLE_LABEL_COUNTER: AtomicUsize = ATOMIC_USIZE_INIT;

                        let mut case_entry_blocks = vec![];
                        let mut ins = vec![];

                        for _ in 0..num_table_entries {
                            let mut block = empty_block();
                            let label_num = CASE_ENTRY_LABEL_COUNTER.fetch_add(1, Ordering::SeqCst);
                            let label = format!("C_{}", label_num);

                            ins.push(Inst::Jmp(Cow::Owned(label.clone()), Some(JumpType::Near)));

                            block.label = label;

                            case_entry_blocks.push(g.add_node(block));
                        }

                        let table_label_num = TABLE_LABEL_COUNTER.fetch_add(1, Ordering::SeqCst);

                        let table = g.add_node(Block {
                            ins,
                            label: format!("T_{}", table_label_num),
                            ..empty_block()
                        });

                        g[compute_jump].ins.push(Inst::Cjmp(hash_r, table));

                        g.add_edge(arg_block, compute_jump, Debug("arg_to_hash"));
                        g.add_edge(compute_jump, table, Debug("hash_to_table"));
                        // let mut prev = None;

                        for (i, (&entry_block, child)) in
                            case_entry_blocks.iter().zip(children).enumerate()
                        {
                            g.add_edge(table, entry_block, Edge::Table(i));
                            let end_block = g.add_node(empty_block());
                            g.add_edge(end_block, exit, Debug("case_end_to_exit"));

                            stack.push((child, ((entry_block, end_block))));

                            // prev = Some();
                        }
                    }
                    (&Tree::Leaf(ref marker), (entry_block, end_block)) => {
                        match *marker {
                            Marker::Case(val, ref st) => {
                                // guard
                                g[entry_block].ins.push(
                                    Inst::Test(arg_r, R::Const(val as i64)),
                                );
                                g.add_edge(entry_block, end_block, Edge::False);

                                let (body_begin, body_end, was_broke) =
                                    c_stmt_inner(st, names, label_exits, g, Some(exit));

                                // TODO: implemented fallthrough and remove this
                                assert!(was_broke);

                                g.add_edge(entry_block, body_begin, Edge::True);

                                // TODO: if !was_broke {
                                g.add_edge(body_end, end_block, Debug("body_end_to_case_end"));
                            }
                            Marker::Default => unimplemented!(),
                        }
                    }
                }
            }

            (arg_block, exit, false)
        }
        Statement::Break(ref s) => {
            let exit = if let Some(ref label) = *s {
                label_exits[label]
            } else {
                exit.unwrap()
            };

            let this = g.add_node(empty_block());
            // g.add_edge(this, exit, Debug("break"));
            (this, this, true)
        }
        Statement::While(ref cond, ref body, ref label) => {
            let (node, arg_r) = c_expr(cond, g, names);
            let (guard_expr, guard_r) = c_expr(&Expr::I64(0), g, names);
            let guard_ins = g.remove_node(guard_expr).unwrap().ins;
            g[node].ins.extend(guard_ins);
            g[node].ins.push(Inst::Test(arg_r, guard_r));

            let exit = g.add_node(empty_block());

            if let Some(ref label) = *label {
                assert!(label_exits.insert(label.clone(), exit).is_none());
            }

            let (begin, end, was_broke) = c_stmt_inner(body, names, label_exits, g, Some(exit));

            g.add_edge(node, begin, Edge::False);
            g.add_edge(node, exit, Edge::True);

            if !was_broke {
                g.add_edge(end, node, Debug("loop"));
            }

            (node, exit, false)
        }
        Statement::Assignment(ref s, ref e) => {
            let (node, r) = c_expr(e, g, names);
            let reg = names[s];

            g[node].ins.push(Inst::Assign(reg, r));
            (node, node, false)
        }
    }
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
        let r = fresh();
        let mut b_ins = g.remove_node(node_b).unwrap().ins;
        let a = &mut g[node_a];
        a.ins.append(&mut b_ins);
        a.ins.push(f(r, r_a, r_b));

        (node_a, r)
    };

    match *e {
        Expr::I64(v) => {
            let r = fresh();
            let this = g.add_node(block(Inst::Assign(r, R::Const(v))));
            (this, r)
        }
        Expr::Add(ref a, ref b) => bin_op(Inst::Add, a, b, g, names),
        Expr::Mult(ref a, ref b) => bin_op(Inst::Mult, a, b, g, names),
        Expr::Var(ref s) => {
            let b = names[s];
            let this = g.add_node(empty_block());
            (this, b)
        }
        Expr::Read => {
            let r = fresh();
            let mut block = block(Inst::Call("read", 0));
            block.ins.push(Inst::Assign(r, R::Pin(Pinned::Rax)));
            let node = g.add_node(block);
            (node, r)
        }
    }
}

fn block(i: Inst) -> Block {
    Block {
        ins: vec![i],
        ..empty_block()
    }
}

fn empty_block() -> Block {
    Block {
        idx: NodeIndex::end(),
        label: "".into(),
        ins: vec![],
    }
}

fn link_builtins(asm: &mut StringBuilder) {
    asm.extend(
        "print:
    sub rsp, 32
    lea rsi, [rsp + 24]
    mov rax, rdi
    mov rcx, 10
    mov byte [rsi], 10
__print_loop:
    dec rsi
    cqo
    idiv rcx
    mov r8, rdx
    sar r8b, 7
    xor dl, r8b
    sub dl, r8b
    or dl, 48
    mov [rsi], dl
    test rax, rax
    jnz __print_loop
    test rdi, rdi
    jns __print_skip_neg
    dec rsi
    mov byte [rsi], 45
__print_skip_neg:
    lea rdx, [rsp + 25]
    sub rdx, rsi
    mov rax, 1
    mov rdi, 1
    syscall
    add rsp, 32
    ret
exit:
    mov rax, 60
    xor rdi, rdi
    syscall",
    );
}

struct StringBuilder {
    string: String,
}

impl StringBuilder {
    fn into(self) -> String {
        self.string
    }
    fn new() -> Self {
        StringBuilder { string: String::new() }
    }

    fn line<S>(&mut self, s: S)
    where
        S: AsRef<str>,
    {
        self.string.push_str("    ");
        self.string.push_str(s.as_ref());
        self.string.push('\n');
    }

    fn label<S>(&mut self, s: S)
    where
        S: AsRef<str>,
    {
        self.string.push_str(s.as_ref());
        self.string.push_str(":\n");
    }

    fn extend<S>(&mut self, s: S)
    where
        S: AsRef<str>,
    {
        self.string += s.as_ref();
    }
}

impl Display for R {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match *self {
            R::Pin(p) => write!(f, "{}", p),
            R::R(n) => write!(f, "R{}", n),
            R::Const(val) => write!(f, "{}", val),
        }
    }
}

impl fmt::Debug for R {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl Display for Pinned {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use self::Pinned::*;
        let s = match *self {
            Rax => "rax",
            Rbx => "rbx",
            Rcx => "rcx",
            Rdx => "rdx",
            Rdi => "rdi",
            Rsi => "rsi",
            R8 => "r8",
            R9 => "r9",
            R10 => "r10",
            R11 => "r11",
            R12 => "r12",
            R13 => "r13",
            R14 => "r14",
            R15 => "r15",
            Rbp => "rbp",
        };
        write!(f, "{}", s)
    }
}

impl fmt::Debug for Pinned {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Display::fmt(self, f)
    }
}


impl Display for Block {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut s = String::new();
        for (i, ins) in self.ins.iter().enumerate() {
            s += &format!("\n{}:  {:?}", i, ins);
        }
        write!(f, "[{}] {}{}", self.idx.index(), self.label, s)
    }
}


impl Display for Edge {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match *self {
            Edge::True => write!(f, "true"),
            Edge::False => write!(f, "false"),
            Edge::Table(idx) => write!(f, "[{}]", idx),
            Edge::Debug(s) => write!(f, "{:?}", s),
        }
    }
}


impl Display for JumpType {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match *self {
            JumpType::Near => write!(f, "near"),
        }
    }
}

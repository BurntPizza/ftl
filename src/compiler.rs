
use petgraph::{stable_graph, visit, Direction, Incoming, Undirected};
use itertools::*;
use petgraph::prelude::{NodeIndex, Graph};
use strum::IntoEnumIterator;

use dataflow::{self, Forward, Backward, May};
use ast::*;
use self::Edge::Debug;

use std::mem;
use std::fmt::{self, Display, Formatter};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Block {
    idx: NodeIndex,
    label: String,
    ins: Vec<Inst>,
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
    use std::sync::atomic::*;
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

#[derive(PartialEq, Debug, Clone)]
pub enum Inst {
    // a == b?
    Test(R, R),
    // a = b
    Assign(R, R),
    Load(R, i64),
    // a = b + c
    Add(R, R, R),
    Mult(R, R, R),
    Call(&'static str, usize),
    Jmp(&'static str),
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Edge {
    True,
    False,
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

impl Display for Edge {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match *self {
            Edge::True => write!(f, "true"),
            Edge::False => write!(f, "false"),
            Edge::Debug(s) => write!(f, "{:?}", s),
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

    let exit = g.add_node(block(Inst::Jmp("exit")));
    if let Some(prev) = prev {
        g.add_edge(prev, exit, Debug("top_exit"));
    }

    contract(&mut g);
    label_blocks(&mut g);
    const_prop(&mut g);

    g
}

pub fn codegen(mut g: Cfg) -> String {
    let lv = LiveVariables::new(&g);
    let ig = interference_graph(&g, &lv);
    let ra = allocate_registers(ig);

    // apply register allocation
    let start = g.node_indices().next().unwrap();
    let mut dfs = visit::Dfs::new(&g, start);
    while let Some(n) = dfs.next(&g) {
        for ins in g[n].ins.iter_mut() {
            match ins {
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
                &mut Inst::Assign(ref mut a, ref mut b) => {
                    debug_assert!(!a.is_const());
                    if a.is_sym() {
                        *a = R::Pin(ra[a]);
                    }
                    if b.is_sym() {
                        *b = R::Pin(ra[b]);
                    }
                }
                &mut Inst::Mult(ref mut a, ref mut b, ref mut c) => {
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
                &mut Inst::Load(ref mut a, _) => {
                    debug_assert!(!a.is_const());
                    if a.is_sym() {
                        *a = R::Pin(ra[a]);
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

    // visit blocks in depth first order.

    while let Some(n) = stack.pop() {
        if visited.contains(&n) {
            continue;
        }

        visited.insert(n);

        compile_bb(&g[n], &mut text);

        if let Some(&Inst::Test(a, b)) = g[n].ins.last() {
            // cmp instruction was already emitted in compile_bb
            // so now emit jump instruction

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
                // only one branch can ever be taken
                if cc == (a == b) {
                    stack.push(far);
                } else {
                    stack.push(near);
                }
            } else {
                let jmp = if cc { "je" } else { "jne" };
                let target = &g[far].label;
                text.line(format!("{} {}", jmp, target));

                stack.push(far);
                stack.push(near);
            }
        } else {
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

    link_builtins(&mut text);

    let mut p = p.into();
    p += &*data.into();
    p += &*text.into();
    p
}

fn const_prop(g: &mut Cfg) -> HashMap<R, i64> {
    let (defs, uses) = defs_uses(&g);
    let mut consts = HashMap::new();
    let mut to_remove = HashSet::new();

    // TODO: transitive propagation

    for (r, def_set) in defs {
        // only 1 def site, might be constant
        if def_set.len() == 1 {
            let (n, i) = *def_set.iter().next().unwrap();
            if let Inst::Load(lr, val) = g[n].ins[i] {
                debug_assert_eq!(r, lr);
                consts.insert(r, val);
                to_remove.insert((n, i));
            }
        }
    }

    // for each use set of a constant
    for (r, use_set, &val) in
        uses.into_iter().filter_map(|(r, use_set)| {
            consts.get(&r).map(|val| (r, use_set, val))
        })
    {
        // mark as constant
        for (n, i) in use_set {
            match g[n].ins.get_mut(i).unwrap() {
                &mut Inst::Add(_, ref mut b, ref mut c) => {
                    if *b == r {
                        *b = R::Const(val);
                    }
                    if *c == r {
                        *c = R::Const(val);
                    }
                }
                &mut Inst::Test(ref mut a, ref mut b) => {
                    if *a == r {
                        *a = R::Const(val);
                    }
                    if *b == r {
                        *b = R::Const(val);
                    }
                }
                &mut Inst::Assign(_, ref mut b) => {
                    if *b == r {
                        *b = R::Const(val);
                    }
                }
                &mut Inst::Mult(_, ref mut b, ref mut c) => {
                    if *b == r {
                        *b = R::Const(val);
                    }
                    if *c == r {
                        *c = R::Const(val);
                    }
                }
                &mut Inst::Load(..) => {}
                &mut Inst::Call(..) |
                &mut Inst::Jmp(..) => {}
            }
        }
    }

    // remove the Load instructions that provided constants
    // refactor: could do this lazily (e.g. return to_remove and just skip ins at assembly time)
    // then the RD (and other passes') info wouldn't be invalidated
    let start = g.node_indices().next().unwrap();
    let mut dfs = visit::Dfs::new(&*g, start);

    while let Some(n) = dfs.next(&*g) {
        let block = &mut g[n].ins;
        for i in (0..block.len()).rev() {
            if to_remove.contains(&(n, i)) {
                block.remove(i);
            }
        }
    }

    consts
}

// TODO: rewrite RD analysis to compute exactly the required info below

fn defs_uses(g: &Cfg) -> (HashMap<R, HashSet<InsIdx>>, HashMap<R, HashSet<InsIdx>>) {
    let rd = ReachingDefs::new(&g);
    let mut ins_idx_to_r = HashMap::new();
    let mut uses = HashMap::new();
    let mut defs = HashMap::new();

    for n in g.node_indices() {
        for (i, ins) in g[n].ins.iter().enumerate() {
            let mut written: Vec<_> = vars_written(ins)
                .into_iter()
                .filter(|r| r.is_sym())
                .collect();
            assert!(written.len() <= 1);
            if let Some(r) = written.pop() {
                uses.entry(r).or_insert(HashSet::new());
                defs.entry(r).or_insert(HashSet::new()).insert((n, i));
                ins_idx_to_r.insert((n, i), r);
            }
        }
    }

    for n in g.node_indices() {
        let ia = rd.internal_analysis(g, n);
        for (i, ins) in g[n].ins.iter().enumerate() {
            for r in vars_read(ins).into_iter().filter(|r| r.is_sym()) {
                for def in &ia[&i] {
                    if r == ins_idx_to_r[def] {
                        uses.get_mut(&r).unwrap().insert((n, i));
                    }
                }
            }
        }
    }

    (defs, uses)
}

fn compile_bb(bb: &Block, asm: &mut StringBuilder) {
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
            Inst::Assign(a, b) => {
                debug_assert!(a.is_pinned());
                debug_assert!(!b.is_sym());

                if a != b {
                    asm.line(format!("mov {}, {}", a, b));
                }
            }
            Inst::Call(s, _) => asm.line(format!("call {}", s)),
            Inst::Jmp(s) => asm.line(format!("jmp {}", s)),
            Inst::Load(a, b) => {
                debug_assert!(a.is_pinned());
                asm.line(format!("mov {}, {}", a, b));
            }
            Inst::Mult(a, b, c) => {
                debug_assert!(a.is_pinned());
                debug_assert!(!b.is_sym());
                debug_assert!(!c.is_sym());
                asm.line(format!("lea {}, [{} * {}]", a, b, c));
            }
            Inst::Test(mut a, mut b) => {
                if a.is_const() {
                    if b.is_const() {
                        continue;
                    }
                    mem::swap(&mut a, &mut b);
                }
                debug_assert!(a.is_pinned());
                debug_assert!(!b.is_sym());
                asm.line(format!("cmp {}, {}", a, b));
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
            if g.neighbors_directed(child, Direction::Incoming).count() > 1 {
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
        g[i].label = format!("L{}", counter);
        counter += 1;
    }
}

pub type InterferenceGraph = Graph<R, (), Undirected>;

pub fn interference_graph(g: &Cfg, lv: &LiveVariables) -> InterferenceGraph {
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
            .unwrap_or_else(|| {
                ig.node_indices()
                    .filter(|&n| ig[n].is_sym())
                    .next()
                    .unwrap()
            });

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

type InsIdx = (NodeIndex, usize);

pub struct LiveVariables {
    out_f: HashMap<NodeIndex, HashSet<R>>,
}

fn vars_read(ins: &Inst) -> HashSet<R> {
    let mut set = HashSet::new();
    match *ins {
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
        Inst::Jmp(_) => {}
        Inst::Load(..) => {}
    }
    set
}

fn vars_written(ins: &Inst) -> HashSet<R> {
    let mut set = HashSet::new();

    match *ins {
        Inst::Add(r, ..) |
        Inst::Assign(r, _) |
        Inst::Load(r, _) |
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

impl LiveVariables {
    pub fn internal_liveness(&self, g: &Cfg, block: NodeIndex) -> HashMap<usize, HashSet<R>> {
        let ins = &g[block].ins;

        // outs for each ins i
        let mut map = HashMap::new();
        let mut current_out = self.out_f[&block].clone();
        for (i, ins) in ins.iter().enumerate().rev() {
            map.insert(i, current_out.clone());

            // kill
            for r in vars_written(ins) {
                current_out.remove(&r);
            }

            // gen
            current_out.extend(vars_read(ins));
        }

        map
    }

    pub fn new(g: &Cfg) -> Self {
        let in_f = g.node_indices().map(|n| (n, HashSet::new())).collect();
        let out_f = g.node_indices().map(|n| (n, HashSet::new())).collect();

        let mut gen = HashMap::new();
        let mut kill = HashMap::new();

        for n in g.node_indices() {
            let mut lgen = HashSet::new();
            let mut lkill = HashSet::new();

            for ins in &g[n].ins {
                // REVIEW: ordering of these
                lkill.extend(vars_written(ins));
                lgen.extend(vars_read(ins).difference(&lkill));
            }

            gen.insert(n, lgen);
            kill.insert(n, lkill);
        }

        let init = Lvstate {
            in_f,
            out_f,
            gen,
            kill,
        };

        dataflow::analyze(g, init, Backward, May)
    }
}

pub struct Lvstate {
    in_f: HashMap<NodeIndex, HashSet<R>>,
    out_f: HashMap<NodeIndex, HashSet<R>>,
    gen: HashMap<NodeIndex, HashSet<R>>,
    kill: HashMap<NodeIndex, HashSet<R>>,
}

impl dataflow::Analysis for LiveVariables {
    type State = Lvstate;

    fn from(state: Self::State) -> Self {
        LiveVariables { out_f: state.out_f }
    }
}

impl dataflow::State for Lvstate {
    type NodeIdx = NodeIndex;
    type Idx = R;
    type Set = HashSet<Self::Idx>;

    fn gen(&self, i: Self::NodeIdx) -> &Self::Set {
        &self.gen[&i]
    }

    fn kill(&self, i: Self::NodeIdx) -> &Self::Set {
        &self.kill[&i]
    }

    fn in_facts(&mut self, i: Self::NodeIdx) -> &mut Self::Set {
        self.in_f.get_mut(&i).unwrap()
    }

    fn out_facts(&mut self, i: Self::NodeIdx) -> &mut Self::Set {
        self.out_f.get_mut(&i).unwrap()
    }
}

pub struct ReachingDefs {
    reach_in: HashMap<NodeIndex, HashSet<InsIdx>>,
    defs: HashMap<R, HashSet<InsIdx>>,
}

impl ReachingDefs {
    pub fn internal_analysis(&self, g: &Cfg, bb: NodeIndex) -> HashMap<usize, HashSet<InsIdx>> {
        let ins = &g[bb].ins;

        let mut map = HashMap::new();
        let mut current_out = self.reach_in[&bb].clone();
        for (i, ins) in ins.iter().enumerate() {
            let mut gen = false;

            // kill
            for r in vars_written(ins).into_iter().filter(|r| !r.is_pinned()) {
                gen = true;
                for ins_idx in &self.defs[&r] {
                    current_out.remove(&ins_idx);
                }
            }

            // gen
            if gen {
                current_out.insert((bb, i));
            }

            map.insert(i, current_out.clone());
        }

        map
    }

    fn new(g: &Cfg) -> Self {
        let mut defs = HashMap::new();
        let mut r_in = HashMap::new();
        let mut r_out = HashMap::new();

        for n in g.node_indices() {
            r_in.insert(n, HashSet::new());
            r_out.insert(n, HashSet::new());

            for (i, ins) in g[n].ins.iter().enumerate() {
                for r in vars_written(ins) {
                    defs.entry(r).or_insert(HashSet::new()).insert((n, i));
                }
            }
        }

        let gen: HashMap<NodeIndex, HashSet<InsIdx>> = g.node_indices()
            .map(|n| {
                (
                    n,
                    g[n]
                        .ins
                        .iter()
                        .enumerate()
                        .filter(|&(_, ins)| {
                            vars_written(ins)
                                .into_iter()
                                .filter(|r| !r.is_pinned())
                                .count() != 0
                        })
                        .map(|(i, _)| (n, i))
                        .collect(),
                )
            })
            .collect();

        let kill = g.node_indices()
            .map(|n| {
                let gen = &gen[&n];
                (
                    n,
                    g[n]
                        .ins
                        .iter()
                        .filter_map(|ins| {
                            let vw: HashSet<R> = vars_written(ins)
                                .into_iter()
                                .filter(|r| !r.is_pinned())
                                .collect();
                            if vw.is_empty() { None } else { Some(vw) }
                        })
                        .flat_map(|r_set| {
                            r_set.into_iter().flat_map(|r| {
                                defs[&r].iter().cloned().filter(|def| !gen.contains(def))
                            })
                        })
                        .collect(),
                )
            })
            .collect();

        let init = Rdstate {
            gen,
            kill,
            in_f: r_in,
            out_f: r_out,
            defs,
        };

        dataflow::analyze(g, init, Forward, May)
    }
}

impl dataflow::Analysis for ReachingDefs {
    type State = Rdstate;

    fn from(state: Self::State) -> Self {
        ReachingDefs {
            reach_in: state.in_f,
            defs: state.defs,
        }
    }
}

pub struct Rdstate {
    gen: HashMap<NodeIndex, HashSet<InsIdx>>,
    kill: HashMap<NodeIndex, HashSet<InsIdx>>,
    in_f: HashMap<NodeIndex, HashSet<InsIdx>>,
    out_f: HashMap<NodeIndex, HashSet<InsIdx>>,
    defs: HashMap<R, HashSet<InsIdx>>,
}

impl dataflow::State for Rdstate {
    type NodeIdx = NodeIndex;
    type Idx = InsIdx;
    type Set = HashSet<Self::Idx>;

    fn gen(&self, i: Self::NodeIdx) -> &Self::Set {
        &self.gen[&i]
    }

    fn kill(&self, i: Self::NodeIdx) -> &Self::Set {
        &self.kill[&i]
    }

    fn in_facts(&mut self, i: Self::NodeIdx) -> &mut Self::Set {
        self.in_f.get_mut(&i).unwrap()
    }

    fn out_facts(&mut self, i: Self::NodeIdx) -> &mut Self::Set {
        self.out_f.get_mut(&i).unwrap()
    }
}

fn c_statement(
    stmt: &Statement,
    names: &mut HashMap<String, R>,
    label_exits: &mut HashMap<String, NodeIndex>,
    g: &mut Cfg,
) -> (NodeIndex, NodeIndex) {

    fn c_stmt_inner(
        stmt: &Statement,
        names: &mut HashMap<String, R>,
        label_exits: &mut HashMap<String, NodeIndex>,
        g: &mut Cfg,
        exit: Option<NodeIndex>,
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
                    let (n_begin, n_end, was_broke) =
                        c_stmt_inner(stmt, names, label_exits, g, exit);

                    if let Some((_, p_end)) = prev {
                        g.add_edge(p_end, n_begin, Debug("block_next"));
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

                let mut case_blocks = vec![];

                for case in cases {
                    match *case {
                        Case::Case(..) => case_blocks.push(g.add_node(empty_block())),
                        Case::Default(ref s) => {
                            default = Some(s);
                        }
                    }
                }

                for ((i, block), case) in case_blocks.iter().cloned().enumerate().zip(cases) {
                    match *case {
                        Case::Case(v, ref s) => {
                            let (guard_node, guard_r) = c_expr(&Expr::I64(v), g, names);
                            let c_end = g.add_node(empty_block());
                            g[guard_node].ins.push(Inst::Test(arg_r, guard_r));


                            if let Some(ref s) = *s {
                                let (begin, end, was_broke) =
                                    c_stmt_inner(s, names, label_exits, g, Some(exit));

                                g.add_edge(block, begin, Debug("case_block"));
                                if !was_broke {
                                    g.add_edge(end, c_end, Debug("case_end"));
                                }
                            } else {
                                let n = if i + 1 < cases.len() {
                                    case_blocks[i + 1]
                                } else {
                                    exit
                                };
                                g.add_edge(block, n, Debug("case_fallthrough"));
                            }

                            g.add_edge(guard_node, block, Edge::True);
                            g.add_edge(guard_node, c_end, Edge::False);
                            g.add_edge(prev, guard_node, Debug("switch_next"));

                            prev = c_end;
                        }
                        _ => {}
                    }
                }

                if let Some(s) = default {
                    let (begin, end, _) = c_stmt_inner(s, names, label_exits, g, Some(exit));
                    g.add_edge(prev, begin, Debug("switch_next_default"));
                    prev = end;
                }

                // NOTE: weirdness with add_edge, possible bug somewhere
                g.update_edge(prev, exit, Debug("switch_exit"));

                (node, exit, false)
            }
            Statement::Break(ref s) => {
                let exit = if let Some(ref label) = *s {
                    label_exits[label]
                } else {
                    exit.unwrap()
                };

                let this = g.add_node(empty_block());
                g.add_edge(this, exit, Debug("break"));
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

    let (a, b, _) = c_stmt_inner(stmt, names, label_exits, g, None);
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
            let this = g.add_node(block(Inst::Load(r, v)));
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


use std::collections::{HashMap, HashSet};

use itertools::*;
use petgraph::visit;
use petgraph::prelude::NodeIndex;

use dataflow::{self, Forward, Set};

use compiler::{Cfg, Inst, R, vars_read};

pub fn const_prop(g: &mut Cfg) {
    let mut in_f = HashMap::new();
    let mut out_f = HashMap::new();
    let mut gen = HashMap::new();
    let mut kill = HashMap::new();

    for n in g.node_indices() {
        in_f.insert(n, RSet::empty());
        out_f.insert(n, RSet::empty());
        let mut consts = HashMap::new();
        let mut lkill = HashMap::new();

        for ins in &g[n].ins {
            match *ins {
                Inst::And(a, b, c) |
                Inst::Mult(a, b, c) |
                Inst::Add(a, b, c) => {
                    if !a.is_sym() {
                        continue;
                    }
                    if let R::Const(bv) = b {
                        consts.insert(b, bv);
                    }
                    if let R::Const(cv) = c {
                        consts.insert(c, cv);
                    }
                    if let Some(&b) = consts.get(&b) {
                        if let Some(&c) = consts.get(&c) {
                            consts.insert(a, const_op(ins, &[b, c]));
                            continue;
                        }
                    }
                    consts.remove(&a);
                    lkill.insert(a, 0);
                }
                Inst::Assign(a, b) => {
                    if !a.is_sym() {
                        continue;
                    }
                    if let R::Const(bv) = b {
                        consts.insert(b, bv);
                    }
                    if let Some(&b) = consts.get(&b) {
                        consts.insert(a, const_op(ins, &[b]));
                        continue;
                    }
                    consts.remove(&a);
                    lkill.insert(a, 0);
                }
                Inst::Cjmp(a, ..) => {
                    //
                }
                Inst::Shr(a, b) => {
                    if let R::Const(bv) = b {
                        consts.insert(b, bv);
                    }
                    lkill.insert(a, 0);
                }
                Inst::Call(..) | Inst::Jmp(..) | Inst::Test(..) => {}
            }
        }

        let lgen = consts.into_iter().filter(|&(r, _)| r.is_sym()).collect();
        gen.insert(n, RSet(lgen));
        kill.insert(n, RSet(lkill));
    }

    let init = ConstProp {
        gen,
        kill,
        in_f,
        out_f,
    };

    fn join<S, Set, I>(state: &S, iter: I) -> Set
    where
        I: Iterator<Item = NodeIndex>,
        Set: dataflow::Set<(R, i64)>,
        S: dataflow::State<Fact = (R, i64), Set = Set, NodeIdx = NodeIndex>,
    {
        let inputs = iter.collect_vec();
        let union_of_inputs = inputs.iter().cloned().map(|i| state.out_facts(i)).fold(
            Set::empty(),
            Set::union,
        );

        let union_of_kills = inputs.iter().cloned().map(|i| state.kill(i)).fold(
            Set::empty(),
            Set::union,
        );

        union_of_inputs.difference(&union_of_kills)
    }

    let cp: ConstProp = dataflow::analyze_custom_join(&*g, init, Forward, join);

    let start = g.node_indices().next().unwrap();
    let mut dfs = visit::Dfs::new(&*g, start);
    let mut to_remove = HashSet::new();
    let mut written = HashMap::new();

    fn const_op(ins: &Inst, input: &[i64]) -> i64 {
        match *ins {
            Inst::Add(..) => input.into_iter().cloned().sum(),
            Inst::And(..) => input.into_iter().cloned().fold(-1, |a, b| a & b),
            Inst::Assign(..) => input.into_iter().cloned().next().unwrap(),
            Inst::Mult(..) => input.into_iter().cloned().product(),
            Inst::Shr(..) | Inst::Test(..) | Inst::Jmp(..) | Inst::Cjmp(..) | Inst::Call(..) => {
                unreachable!()
            }
        }
    }

    while let Some(n) = dfs.next(&*g) {
        let mut consts: HashMap<R, i64> = cp.in_f[&n].0.clone();

        for (i, ins) in g[n].ins.iter_mut().enumerate() {
            debug_assert!(consts.keys().all(|r| !r.is_pinned()));
            let i_clone = ins.clone();
            match ins {
                // assigning 3
                &mut Inst::And(ref mut a, ref mut b, ref mut c) |
                &mut Inst::Mult(ref mut a, ref mut b, ref mut c) |
                &mut Inst::Add(ref mut a, ref mut b, ref mut c) => {
                    if let R::Const(bv) = *b {
                        consts.insert(*b, bv);
                    } else if let Some(&bv) = consts.get(b) {
                        *b = R::Const(bv);
                    }
                    if let R::Const(cv) = *c {
                        consts.insert(*c, cv);
                    } else if let Some(&cv) = consts.get(c) {
                        *c = R::Const(cv);
                    }
                    if let R::Const(b) = *b {
                        if let R::Const(c) = *c {
                            if !a.is_pinned() {
                                consts.insert(*a, const_op(&i_clone, &[b, c]));
                                to_remove.insert((n, i));
                                written.insert((n, i), *a);
                            }
                            continue;
                        }
                    }
                    consts.remove(a);
                }
                // assigning 2
                &mut Inst::Assign(ref mut a, ref mut b) => {
                    if let R::Const(bv) = *b {
                        consts.insert(*b, bv);
                    } else if let Some(&bv) = consts.get(b) {
                        *b = R::Const(bv);
                    }
                    if let R::Const(b) = *b {
                        if !a.is_pinned() {
                            consts.insert(*a, const_op(&i_clone, &[b]));
                            to_remove.insert((n, i));
                            written.insert((n, i), *a);
                        }
                        continue;
                    }
                    consts.remove(a);
                }
                // reading 2
                &mut Inst::Test(ref mut a, ref mut b) => {
                    if let R::Const(av) = *a {
                        consts.insert(*a, av);
                    } else if let Some(&val) = consts.get(a) {
                        *a = R::Const(val);
                    }
                    if let R::Const(bv) = *b {
                        consts.insert(*b, bv);
                    } else if let Some(&val) = consts.get(b) {
                        *b = R::Const(val);
                    }
                }
                &mut Inst::Cjmp(ref mut a, ..) => {
                    //
                }
                &mut Inst::Call(..) |
                &mut Inst::Jmp(..) => {}
                &mut Inst::Shr(_, ref mut b) => {
                // &mut Inst::Cjmp(_, ref mut b, _) => {
                    if let R::Const(bv) = *b {
                        consts.insert(*b, bv);
                    } else if let Some(&bv) = consts.get(b) {
                        *b = R::Const(bv);
                    }
                }
            }
        }
    }

    let mut uses = HashMap::new();

    for n in g.node_indices() {
        for (i, ins) in g[n].ins.iter().enumerate() {
            match *ins {
                Inst::Call(..) => {}
                _ => {
                    for r in vars_read(ins).into_iter().filter(R::is_sym) {
                        uses.entry(r).or_insert_with(HashSet::new).insert((n, i));
                    }
                }
            }
        }
    }

    let to_remove: HashSet<_> = to_remove
        .iter()
        .cloned()
        .filter(|idx| {
            uses.get(&written[idx])
                .into_iter()
                .flat_map(|opt| opt)
                .all(|idx| to_remove.contains(idx))
        })
        .collect();

    // remove the dead instructions that provided constants
    // refactor: could do this lazily?
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
}

struct ConstProp {
    in_f: HashMap<NodeIndex, RSet>,
    out_f: HashMap<NodeIndex, RSet>,
    gen: HashMap<NodeIndex, RSet>,
    kill: HashMap<NodeIndex, RSet>,
}

#[derive(Debug, Clone, PartialEq)]
struct RSet(HashMap<R, i64>);

impl Set<(R, i64)> for RSet {
    fn empty() -> Self {
        RSet(HashMap::new())
    }

    // note: only compare keys
    fn difference(&self, other: &Self) -> Self {
        RSet(
            self.0
                .iter()
                .map(|(&k, &v)| (k, v))
                .filter(|&(ref k, _)| !other.0.contains_key(&k))
                .collect(),
        )
    }

    fn union(mut self, other: &Self) -> Self {
        self.0.extend(other.0.iter().map(|(&k, &v)| (k, v)));
        self
    }

    fn intersection(self, other: &Self) -> Self {
        RSet(
            self.0
                .into_iter()
                .filter(|&(k, v)| other.0.get(&k).map_or(false, |&ov| ov == v))
                .collect(),
        )
    }
}

impl dataflow::State for ConstProp {
    type NodeIdx = NodeIndex;
    type Fact = (R, i64);
    type Set = RSet;

    fn gen(&self, i: Self::NodeIdx) -> &Self::Set {
        &self.gen[&i]
    }

    fn kill(&self, i: Self::NodeIdx) -> &Self::Set {
        &self.kill[&i]
    }

    fn in_facts(&self, i: Self::NodeIdx) -> &Self::Set {
        self.in_f.get(&i).unwrap()
    }

    fn in_facts_mut(&mut self, i: Self::NodeIdx) -> &mut Self::Set {
        self.in_f.get_mut(&i).unwrap()
    }

    fn out_facts(&self, i: Self::NodeIdx) -> &Self::Set {
        self.out_f.get(&i).unwrap()
    }

    fn out_facts_mut(&mut self, i: Self::NodeIdx) -> &mut Self::Set {
        self.out_f.get_mut(&i).unwrap()
    }
}

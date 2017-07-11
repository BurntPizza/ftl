
use std::collections::{HashMap, HashSet};

use petgraph::prelude::NodeIndex;

use dataflow::{self, Backward, May};

use compiler::{Cfg, R, vars_read, vars_written};

pub fn live_variables(g: &Cfg) -> LiveVariables {
    let f = || g.node_indices().map(|n| (n, HashSet::new())).collect();

    let mut gen = HashMap::new();
    let mut kill = HashMap::new();

    for n in g.node_indices() {
        let mut lgen = HashSet::new();
        let mut lkill = HashSet::new();

        for ins in &g[n].ins {
            // REVIEW: ordering of these
            lkill.extend(vars_written(ins).into_iter().filter(|r| !r.is_const()));
            lgen.extend(vars_read(ins).difference(&lkill).filter(|r| !r.is_const()));
        }

        gen.insert(n, lgen);
        kill.insert(n, lkill);
    }

    let init = LiveVariables {
        in_f: f(),
        out_f: f(),
        gen,
        kill,
    };

    dataflow::analyze(g, init, Backward, May)
}

pub struct LiveVariables {
    in_f: HashMap<NodeIndex, HashSet<R>>,
    out_f: HashMap<NodeIndex, HashSet<R>>,
    gen: HashMap<NodeIndex, HashSet<R>>,
    kill: HashMap<NodeIndex, HashSet<R>>,
}

impl LiveVariables {
    pub fn internal_liveness(&self, g: &Cfg, block: NodeIndex) -> HashMap<usize, HashSet<R>> {
        let ins = &g[block].ins;

        // outs for each ins i
        let mut map = HashMap::new();
        let mut current_out = self.out_f[&block].clone();
        for (i, ins) in ins.iter().enumerate().rev() {
            // kill
            for r in vars_written(ins) {
                current_out.remove(&r);
            }

            // gen
            current_out.extend(vars_read(ins));

            // REVIEW: ordering
            map.insert(i, current_out.clone());
        }

        map
    }
}

impl dataflow::State for LiveVariables {
    type NodeIdx = NodeIndex;
    type Fact = R;
    type Set = HashSet<Self::Fact>;

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

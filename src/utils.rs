
use std::fmt::{Display};
use std::collections::HashMap;

use ::pg;
use pg::Graph;

pub trait AsGraph<N, E, D> {
    fn as_graph(&self) -> Graph<N, E, D>;
}

impl<N: Clone, E: Clone, D: pg::EdgeType> AsGraph<N, E, D>
    for pg::stable_graph::StableGraph<N, E, D> {
    fn as_graph(&self) -> Graph<N, E, D> {
        let mut gg: Graph<N, E, D> = Graph::new().into_edge_type();
        let mut map = HashMap::new();

        for i in self.node_indices() {
            let new_node = gg.add_node(self[i].clone());
            map.insert(i, new_node);
        }

        for i in self.node_indices() {
            for n in self.neighbors(i) {
                let e = self.find_edge(i, n).unwrap();
                let s = map[&i];
                let t = map[&n];
                gg.add_edge(s, t, self[e].clone());
            }
        }
        gg
    }
}

impl<N: Clone, E: Clone, D: pg::EdgeType> AsGraph<N, E, D> for Graph<N, E, D> {
    fn as_graph(&self) -> Graph<N, E, D> {
        (*self).clone()
    }
}

pub fn print_graph<N, E, D>(g: &AsGraph<N, E, D>)
where
    N: Display,
    E: Display,
    D: pg::EdgeType,
{
    let gg = g.as_graph();

    let dot = pg::dot::Dot::new(&gg);
    println!("{:#}", dot);
}

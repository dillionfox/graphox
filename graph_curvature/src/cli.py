import random

import networkx as nx
import pandas as pd

from curvature import GraphCurvature


def make_example_graph(n: int = 4) -> nx.Graph:
    G = nx.dorogovtsev_goltsev_mendes_graph(n)
    for (v1, v2) in G.edges():
        G[v1][v2]['weight'] = 1.0
    return G


def make_example_node_weights(G: nx.Graph, n_patients: int) -> pd.DataFrame:
    node_weight_sets = pd.DataFrame([dict(zip(list(G.nodes), random.choices(range(1, 3), k=len(G.nodes)))) for n in
                                     range(n_patients)])
    return node_weight_sets


def make_example(n: int = 4, n_patients: int = 3) -> tuple:
    G = make_example_graph(n)
    node_weight_sets = make_example_node_weights(G, n_patients)
    return G, node_weight_sets


def compute_scalar_curvature(G: nx.Graph) -> GraphCurvature:
    orc = GraphCurvature(G)
    orc.compute()
    return orc


def compute_nodal_curvatures(orc: GraphCurvature, node_weight_sets: pd.DataFrame) -> tuple:
    curvature_per_patient = dict()
    nodal_curvature = pd.DataFrame([])
    for n, column in enumerate(node_weight_sets.columns):
        curvature_per_patient[n], nodal_curvature = orc.compute_total_curvature(node_weight_sets[column])
    return curvature_per_patient, nodal_curvature


def main(G: nx.Graph, node_weight_sets: pd.DataFrame):
    pass

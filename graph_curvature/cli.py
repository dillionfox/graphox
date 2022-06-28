import random
from os.path import exists

import click
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


@click.command()
@click.option('--G', default=nx.Graph(), help='NetworkX graph object')
@click.option('--G_pickle', default='', help='Path to pickle file containing NetworkX graph')
@click.option('--node_weight_sets', default=pd.DataFrame([]), help='DataFrame containing node weight sets')
@click.option('--node_weight_sets_csv', default='', help='CSV file containing node weight sets')
def main(G: nx.Graph, G_pickle: str, node_weight_sets: pd.DataFrame, node_weight_sets_csv: str):

    # Make sure all user input has correct type
    if not isinstance(G, nx.Graph):
        raise TypeError('G must be a NetworkX graph. User supplied input of type {}'.format(type(G)))
    if not isinstance(G_pickle, str):
        raise TypeError('G_pickle must be a str. User supplied input of type {}'.format(type(G_pickle)))
    if not isinstance(node_weight_sets, pd.DataFrame):
        raise TypeError('node_weight_sets must be a pandas DataFrame. User supplied input of type {}'.format(
            type(node_weight_sets)))
    if not isinstance(node_weight_sets_csv, str):
        raise TypeError(
            'node_weight_sets_csv must be a str. User supplied input of type {}'.format(type(node_weight_sets_csv)))

    # If user does not provide input for G, try to read from pickle or make an example.
    if nx.is_empty(G):
        if G_pickle:
            if exists(G_pickle):
                G = nx.read_gpickle(G_pickle)
            else:
                raise ValueError('Path supplied for G_pickle is invalid.')
        else:
            print('No graph was provided. Generating a graph to use as an example. Please see usage.')
            G = make_example_graph(4)

    # If user does not provide DataFrame containing node weight sets, try to read them in or make an example.
    if node_weight_sets.empty:
        if node_weight_sets_csv:
            if exists(node_weight_sets_csv):
                node_weight_sets = pd.read_csv(node_weight_sets_csv)
            else:
                raise ValueError('Path supplied for node_weight_sets_csv is invalid.')
        else:
            print('No node weight sets were provided. Generating an example. Please see usage.')
            node_weight_sets = make_example_node_weights(G, 3)

    # Make sure graph and node weight sets are compatible
    if len(G.nodes) != node_weight_sets.shape[0]:
        raise NodeWeightMatchError('The number of nodes in the graph, G, must match the number of rows in the ' +
                                   'node_weight_sets DataFrame. User provided: G.nodes = {}, '.format(len(G.nodes)) +
                                   'node_weight_sets.rows = {}'.format(node_weight_sets.shape[0])
                                   )

    # Compute curvature
    scalar_curvature = compute_scalar_curvature(G)
    curvature_per_patient, nodal_curvature = compute_nodal_curvatures(scalar_curvature, node_weight_sets)

    return curvature_per_patient, nodal_curvature


class NodeWeightMatchError(Exception):
    """The number of nodes in the graph, G, must match the number of rows in the node_weight_sets DataFrame."""

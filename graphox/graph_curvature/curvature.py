"""Simple implementation of Ollivier Ricci curvature calculator
Copyright (C) 2022 Dillion Fox

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.


Note - I borrowed some really cool features from:
https://github.com/saibalmars/GraphRicciCurvature/blob/master/GraphRicciCurvature/OllivierRicci.py
Using lru_cache & heapq was taken directly from this script, and both are super performant.

"""

import functools
import heapq
import multiprocessing as mp
from os.path import exists

import networkit as nk
import networkx as nx
import numpy as np
import ot
import pandas as pd


class GraphCurvature(object):
    r"""Compute Ollivier Ricci curvature (e.g., Graph Curvature, Network Curvature)
    on a NetworkX graph object, described in
    `Ricci curvature of Markov chains on metric spaces, Yann Ollivier
    <https://www.sciencedirect.com/science/article/pii/S002212360800493X?via%3Dihub>`

    This method has been used in numerous studies for estimating patient prognosis
    from "omics" data. Examples:
        - <https://www.nature.com/articles/srep12323>
        - <https://www.nature.com/articles/s41525-021-00259-9>
        - <https://www.biorxiv.org/content/10.1101/2022.03.24.485712v2.full.pdf>
        - <https://arxiv.org/abs/1502.04512>

    Patients with *lower* total curvature are thought to have *poorer* prognoses.
    I highly recommend reading the Ollivier paper (link provided above) for a
    technical deep-dive into the method before reading the applications-focused
    papers.

    Args:
        G (networkX graph): input graph
        n_procs (int, optional): Number of processors to use. Default is all available procs.
        alpha (float, optional): Tunable parameter described in Ollivier paper. Default: 0.5.
        max_nodes_in_heap (int, optional): Optional parameter designed to avoid OOM issues.
            Default: 3000. For powerful computing resources, this number can be drastically
            increased.

    Note:
        Using this in a GCN could be super cool.
            Starting point: https://github.com/GeoX-Lab/CGNN
            Graph classification: https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing

    """

    def __init__(
            self,
            G: nx.Graph,
            n_procs: int = mp.cpu_count(),
            alpha: float = 0.5,
            max_nodes_in_heap: int = 3000,
            scalar_curvatures: pd.DataFrame = pd.DataFrame([])
    ) -> None:

        self.G = G.copy()
        self.proc = n_procs
        self.alpha = alpha
        self._check_edge_weights()
        self._remove_self_loops()
        self.all_pairs_shortest_path: np.array
        self.all_pairs_shortest_path = None
        self.scalar_curvatures: pd.DataFrame
        self.scalar_curvatures = None
        self.use_heap: bool
        self.use_heap = True
        self.max_nodes_in_heap = max_nodes_in_heap
        self.scalar_curvatures = scalar_curvatures
        self.edge_curvatures: dict
        self.edge_curvatures = dict()

    def __str__(self) -> str:
        return r"""Very basic class for computing graph curvature from a weighted NetworkX graph"""

    @classmethod
    def from_save(cls, G: nx.Graph, scalar_curvatures: pd.DataFrame, **kwargs):
        r"""Alternative entry point to instantiating the class if previous savepoint was used.

        :param G: nx.Graph
        :param scalar_curvatures: pd.DataFrame
        :param kwargs:
        :return: Instantiated object
        """
        return cls(G, scalar_curvatures=scalar_curvatures, **kwargs)

    def save_checkpoint(self, name: str = 'graph_curvature_savepoint') -> None:
        r"""Save graph and scalar curvatures to file. This can be handy when exploring different ideas, or for
        inspecting exactly what is happening at each stage of the calculation.

        :param name: base name to use for naming files
        :return: None
        """
        graph_name = '{}_G.pkl'.format(name)
        if exists(graph_name):
            raise NameError('{} exists. Refusing to overwrite. Choose another name and try again.'.format(graph_name))
        nx.write_gpickle(self.G)

        scalar_curvatures_name = '{}_scalar_curvatures.csv'.format(name)
        if exists(scalar_curvatures_name):
            raise NameError(
                '{} exists. Refusing to overwrite. Choose another name and try again.'.format(scalar_curvatures_name))
        self.scalar_curvatures.to_csv(scalar_curvatures_name)

    def compute(self) -> None:
        r"""Main function for computing graph curvature. First compute edge curvatures, then nodal curvatures.

        The compute_total_curvature function computes the sum of nodal curvatures, weighted by node weights.
        """

        # If edge_curvatures were computed using compute_edge_curvatures, don't recompute them.
        if not self.edge_curvatures:
            # Compute curvature for all edges
            self.compute_edge_curvatures()

        # Assign edge curvatures to graph, G
        nx.set_edge_attributes(self.G, self.edge_curvatures, 'edge_curvature')

        # Compute scalar curvatures
        for node in self.G.nodes():
            self.G.nodes[node]['curvature'] = sum(
                [self.G[node][neighbor]['edge_curvature'] / self.G.degree(node) for neighbor in
                 self.G.neighbors(node)])

        # Set instance attribute for convenience
        self.scalar_curvatures = pd.DataFrame(self.G._node).T
        self.scalar_curvatures.index = list(self.G.nodes)
        self.scalar_curvatures.index.name = 'gene'

    def compute_edge_curvatures(self):
        r"""Convert graph to "NetworKit" object so that all shortest paths between nodes can be computed.
        Then, for each pair of nodes, compute Ricci curvature for given edge.

         :return dict containing networkX node pairs mapped to edge curvatures
        """

        global G_nk

        # Convert Networkx object to Networkit graph
        G_nk = nk.nxadapter.nx2nk(self.G, weightAttr='weight')
        # Construct dictionaries to make it easy to translate between nx and nk
        nx2nk_ndict = {n: idx for idx, n in enumerate(self.G.nodes())}
        nk2nx_ndict = {idx: n for idx, n in enumerate(self.G.nodes())}

        # Compute "All Pairs Shortest Path" using NetworKit library
        self.all_pairs_shortest_path = np.array(nk.distance.APSP(G_nk).run().getDistances())

        # Prepare source/target tuples for Multiprocessing
        sc_tg_pairs = [(nx2nk_ndict[source], nx2nk_ndict[target]) for source, target in self.G.edges()]

        # Compute edge curvature with Multiprocessing
        with mp.get_context('fork').Pool(processes=self.proc) as pool:
            result = pool.imap_unordered(self._compute_single_edge, sc_tg_pairs,
                                         chunksize=int(np.ceil(len(sc_tg_pairs) / (self.proc * 4))))
            pool.close()
            pool.join()

        # Return curvature for all edges
        self.edge_curvatures = {(nk2nx_ndict[k[0]], nk2nx_ndict[k[1]]): rc[k] for rc in result for k in list(rc.keys())}

    def _compute_single_edge(self, sc_tg_pair: tuple) -> dict:
        r"""Compute Ricci curvature for a single edge/single pair of nodes

        :param sc_tg_pair: A pair of networkX nodes
        :return: dict of networkX nodes mapped to Ricci curvature
        """

        # If the weight of edge is too small, return 0 instead.
        if G_nk.weight(sc_tg_pair[0], sc_tg_pair[1]) < 1e-6:
            return {(sc_tg_pair[0], sc_tg_pair[1]): 0}

        # Distribute densities for source and source's neighbors as x
        x, source_top_neighbors = self._get_single_node_neighbors_distributions(sc_tg_pair[0])
        y, target_top_neighbors = self._get_single_node_neighbors_distributions(sc_tg_pair[1])

        # Compute transportation matrix / Loss matrix
        d = self.all_pairs_shortest_path[np.ix_(source_top_neighbors, target_top_neighbors)]

        # Return curvature for given edge: k = 1 - m_{x,y} / d(x,y)
        return {(sc_tg_pair[0], sc_tg_pair[1]): 1 - (ot.emd2(x, y, d) / G_nk.weight(sc_tg_pair[0], sc_tg_pair[1]))}

    @functools.lru_cache
    def _get_single_node_neighbors_distributions(self, node: str) -> tuple:
        r"""Compute weighted distribution of nodes and lookup neighbors

        :param node: NetworkX node
        :return: tuple containing weighted distribution and neighbors (networkX nodes)
        """

        # No neighbor, all mass stay at node
        if not list(G_nk.iterNeighbors(node)):
            return [1], [node]

        if not self.use_heap:
            # Get sum of distributions from x's all neighbors
            weight_node_pair = [(np.e ** (-G_nk.weight(node, nbr) ** 2), nbr) for nbr in
                                list(G_nk.iterNeighbors(node))]

            # Compute sum of weights from neighbors
            nbr_edge_weight_sum = sum([x[0] for x in weight_node_pair])

        else:
            weight_node_pair = []
            for nbr in list(G_nk.iterNeighbors(node)):
                weight = np.e ** (-G_nk.weight(node, nbr) ** 2)
                if len(weight_node_pair) < self.max_nodes_in_heap:
                    heapq.heappush(weight_node_pair, (weight, nbr))
                else:
                    heapq.heappushpop(weight_node_pair, (weight, nbr))
            nbr_edge_weight_sum = sum([x[0] for x in weight_node_pair])

        # Compute weighted distribution of pairs
        distributions = [(1.0 - self.alpha) * w / nbr_edge_weight_sum for w, _ in
                         weight_node_pair] if nbr_edge_weight_sum > 1e-6 else [(1.0 - self.alpha) / len(
            weight_node_pair)] * len(weight_node_pair)

        # Return distribution and list of neighbors
        return distributions + [self.alpha], [x[1] for x in weight_node_pair] + [node]

    def _make_neighbor_weights_df(self, node_weights: dict) -> pd.DataFrame:
        r"""Construct dataframe containing scalar curvature for each gene (computed from edge contractions)
        and gene weights from patient omics data.
        """

        # Create symmetrized (redundant) list of edges
        links = pd.concat([pd.DataFrame(self.G.edges, columns=['gene1', 'gene2']),
                           pd.DataFrame(self.G.edges, columns=['gene2', 'gene1'])]).reset_index(drop=True)

        # This should not drop anything
        links.drop_duplicates(subset=['gene1', 'gene2'], inplace=True)

        # Groupby on 'gene1' to get all neighbors for all genes. Because edge list was symmetrized,
        # this catches all possible neighbors for each gene. Replace gene names with associated
        # patient omics weights.
        df_nei_nws = links.groupby('gene1')['gene2'].apply(list).apply(
            lambda x: [node_weights[_] for _ in x]).reset_index()

        # Assign column names
        df_nei_nws.columns = ['gene', 'neighbor_weights']

        # Merge neighboring patient-omics weights with scalar curvature
        df_nodes = self.scalar_curvatures.merge(df_nei_nws, right_on='gene', left_on='gene')

        # Lookup patient omics value for each node (gene)
        df_nodes['weight'] = df_nodes['gene'].apply(lambda x: node_weights[x])
        return df_nodes

    def compute_total_curvature(self, node_weights: dict) -> tuple:
        r"""Compute nodal curvatures and sum them to compute total curvature"""
        # Make sure the user passed in a dictionary with the correct number of keys (one per gene)
        self._check_node_weights(node_weights)

        # Construct node-centric dataframe containing patient omics (node weights) data and 
        # scalar curvatures
        df_nodes = self._make_neighbor_weights_df(node_weights)

        # Scale the node weight (omics) by the sum of the weights of the neighbors
        df_nodes['pi'] = df_nodes.apply(lambda x: x['weight'] * sum(x['neighbor_weights']), axis=1)

        # Normalization according to Equation 10 -- changed "sum" to "mean"
        df_nodes['pi'] = df_nodes['pi'] / (df_nodes['pi'].sum())

        # Scale the scalar curvatures by pi
        df_nodes['nodal_curvature'] = df_nodes.apply(lambda x: 100 * x['pi'] * x['curvature'], axis=1)

        # Compute total curvature: sum of all nodal curvatures for the network
        total_curvature = df_nodes['nodal_curvature'].sum()
        return total_curvature, df_nodes['nodal_curvature']

    def _check_edge_weights(self) -> None:
        r"""If edge weights aren't defined, set them all to 1 """
        if not nx.get_edge_attributes(self.G, 'weight'):
            print('Edge weights are not defined! Setting them all equal to 1.')
            for (v1, v2) in self.G.edges():
                self.G[v1][v2]['weight'] = 1.0

    def _check_node_weights(self, node_weights: dict) -> None:
        r"""If edge weights aren't defined, set them all to 1 """
        if node_weights is None or not isinstance(node_weights, dict):
            print('Node weights are not defined! They must be passed in as a dict. Assigning 1 for all weights.')
            node_weights = dict(zip(list(self.G.nodes), [1] * len(self.G.nodes)))

        if not len(node_weights.keys()) == len(self.G.nodes):
            raise ValueError('Node weights dict was passed in with the wrong number of keys. Exiting.')

    def _remove_self_loops(self) -> None:
        r"""Remove self-loops. We can't work with these """
        self_loop_edges = list(nx.selfloop_edges(self.G))
        if self_loop_edges:
            self.G.remove_edges_from(self_loop_edges)
            print('Removing self-loops! Check your graph to see what went wrong: nx.selfloop_edges(G)')

    def curvature_per_pat(self, omics_df: pd.DataFrame, rec: bool = False) -> tuple:
        r"""This function needs to be rewritten. The goal is to compute curvature for a set of
        node weights (e.g., patients) without recomputing the edge and scalar curvatures.

        :param omics_df: Dataframe containing omics data. First column = 'gene',
                                        subsequent columns represent patients/samples
        :param rec: This option is specific to one of my use cases. It needs
                                        to be refactored out.
        :return (tuple): pd.DataFrame, pd.DataFrame
        """

        if not omics_df.columns[0] == 'gene':
            print('First column of omics_df must be "gene". Exiting.')
            return ()

        pat_curves = dict()
        nodal_curvature_list = []
        for pat in omics_df.columns[1:]:
            node_weights = dict(zip(omics_df['gene'], omics_df[pat]))
            pat_curves[pat], nodal_curvatures = self.compute_total_curvature(node_weights)
            nodal_curvature_list.append(nodal_curvatures)

        curvatures_df = pd.DataFrame(list(pat_curves.items()), columns=['subject', 'curvature']).sort_values(
            by='curvature',
            ascending=False)

        # Discard mismatched ids
        common_indices = list(set(omics_df.columns).intersection(curvatures_df['subject']))
        omics_df = omics_df[['gene'] + common_indices]

        nodal_curvatures = pd.concat(nodal_curvature_list, axis=1)
        nodal_curvatures.columns = omics_df.drop('gene', axis=1).columns
        nodal_curvatures.set_index(omics_df['gene'], inplace=True)
        return curvatures_df, nodal_curvatures


def compute_scalar_curvature(G: nx.Graph) -> GraphCurvature:
    orc = GraphCurvature(G)
    orc.compute()
    return orc


def compute_nodal_curvatures(orc: GraphCurvature, node_weight_sets: pd.DataFrame) -> tuple:
    curvature_per_patient = pd.DataFrame([])
    nodal_curvature = pd.DataFrame([])
    for n, column in enumerate(node_weight_sets.drop(columns=['gene']).columns):
        weights_dict = dict(zip(node_weight_sets['gene'].tolist(), node_weight_sets[column].tolist()))
        curvature_per_patient[n], nodal_curvature = orc.compute_total_curvature(weights_dict)
    return curvature_per_patient.T, nodal_curvature

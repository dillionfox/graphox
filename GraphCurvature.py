import functools
import math

import multiprocessing as mp
import networkit as nk
import networkx as nx
import numpy as np
import pandas as pd
import ot


class GraphCurvature:
    def __init__(self, G: nx.Graph, n_procs=mp.cpu_count(), alpha=0.5):
        self.G = G.copy()
        self.proc = n_procs
        self.alpha = alpha
        self._check_edge_weights()
        self._remove_self_loops()
        self.all_pairs_shortest_path = None
        self.nodal_curvatures = None

    def __str__(self):
        return """Very basic class for computing graph curvature from a weighted NetworkX graph"""

    def compute(self):
        """
        Main function for computing graph curvature. First compute edge curvatures, then nodal curvatures.

        """
        # Compute curvature for all edges
        edge_curvatures = self._compute_edge_curvatures()
        # Assign edge curvatures to graph, G
        nx.set_edge_attributes(self.G, edge_curvatures, 'edge_curvature')
        # Compute nodal curvatures
        for node in self.G.nodes():
            self.G.nodes[node]['curvature'] = sum(
                [self.G[node][neighbor]['edge_curvature'] / self.G.degree(node) for neighbor in
                 self.G.neighbors(node)])
        self.nodal_curvatures = pd.DataFrame(self.G._node).T

    def _compute_edge_curvatures(self):
        """
        Convert graph to "NetworKit" object so that all shortest paths between nodes can be computed.
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
        return {(nk2nx_ndict[k[0]], nk2nx_ndict[k[1]]): rc[k] for rc in result for k in list(rc.keys())}

    def _compute_single_edge(self, sc_tg_pair):
        """
        Compute Ricci curvature for a single edge/single pair of nodes

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

    @functools.lru_cache(1000000)
    def _get_single_node_neighbors_distributions(self, node):
        """
        Compute weighted distribution of nodes and lookup neighbors

        :param node: NetworkX node
        :return: tuple containing weighted distribution and neighbors (networkX nodes)
        """

        # No neighbor, all mass stay at node
        if not list(G_nk.iterNeighbors(node)):
            return [1], [node]

        # Get sum of distributions from x's all neighbors
        weight_node_pair = [(math.e ** (-G_nk.weight(node, nbr) ** 2), nbr) for nbr in list(G_nk.iterNeighbors(node))]

        # Compute sum of weights from neighbors
        nbr_edge_weight_sum = sum([x[0] for x in weight_node_pair])

        # Compute weighted distribution of pairs
        distributions = [(1.0 - self.alpha) * w / nbr_edge_weight_sum for w, _ in
                         weight_node_pair] if nbr_edge_weight_sum > 1e-6 else [(1.0 - self.alpha) / len(
                            weight_node_pair)] * len(weight_node_pair)

        # Return distribution and list of neighbors
        return distributions + [self.alpha], [x[1] for x in weight_node_pair] + [node]

    def _make_neighbor_weights_df(self, node_weights):
        links = pd.DataFrame(self.G.edges, columns=['gene1', 'gene2'])
        df_nei_nws = links.groupby('gene1')['gene2'].apply(list).apply(
            lambda x: [node_weights[_] for _ in x]).reset_index()
        df_nei_nws.columns = ['gene', 'neighbor_weights']
        df_nodes = self.nodal_curvatures.merge(df_nei_nws, right_on='gene', left_index=True)
        df_nodes['weight'] = df_nodes['gene'].apply(lambda x: node_weights[x])
        return df_nodes

    def compute_total_curvature(self, node_weights):
        self._check_node_weights(node_weights)
        df_nodes = self._make_neighbor_weights_df(node_weights)

        df_nodes['pi'] = df_nodes.apply(lambda x: x['weight'] * sum(x['neighbor_weights']), axis=1)
        # Normalization according to Equation 10
        df_nodes['pi'] = df_nodes['pi'] / (df_nodes['pi'].sum())

        curv_sums = df_nodes.groupby('gene')['curvature'].apply(sum)
        df_nodes['scalar_curvature'] = df_nodes.apply(lambda x: x['pi'] * curv_sums.loc[x['gene']], axis=1)
        total_curvature = df_nodes['scalar_curvature'].sum()
        return total_curvature

    def _check_edge_weights(self):
        """
        If edge weights aren't defined, set them all to 1.

        """
        if not nx.get_edge_attributes(self.G, 'weight'):
            print('Edge weights are not defined! Setting them all equal to 1.')
            for (v1, v2) in self.G.edges():
                self.G[v1][v2]['weight'] = 1.0

    def _check_node_weights(self, node_weights):
        """
        If edge weights aren't defined, set them all to 1.

        """
        if node_weights is None or not isinstance(node_weights, dict):
            print('Node weights are not defined! They must be passed in as a dict. Assigning 1 for all weights.')
            node_weights = dict(zip(list(self.G.nodes), [1]*len(self.G.nodes)))
        if not len(node_weights.keys()) == len(self.G.nodes):
            print('Node weights dict was passed in with the wrong number of keys. Exiting.')
            exit()

    def _remove_self_loops(self):
        """
        Remove self-loops. We can't work with these.

        """
        self_loop_edges = list(nx.selfloop_edges(self.G))
        if self_loop_edges:
            self.G.remove_edges_from(self_loop_edges)
            print('Removing self-loops! Check your graph to see what went wrong: nx.selfloop_edges(G)')

"""Base class for building graphs for graph machine learning
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

"""

from abc import ABC, abstractmethod
from pathlib import Path

import dask.dataframe as dd
import networkx as nx
import numpy as np
import pandas as pd
from torch_geometric.utils.convert import from_networkx

from graphox.graph_curvature.curvature import GraphCurvature


class BaseGraphBuilder(ABC):

    def __init__(self,
                 omics_data_file: str,
                 omics_annotation_file: str,
                 string_aliases_file: str,
                 string_edges_file: str,
                 confidence_level: int = 900,
                 output_dir: str = 'output',
                 graph_file_name: str = 'G.pkl',
                 curvature_file_name: str = 'curvatures.csv',
                 n_procs: int = 4,
                 make_pytorch_graphs: bool = True
                 ):
        self.omics_data_file = omics_data_file
        self.omics_data: pd.DataFrame = pd.DataFrame([])
        self.omics_annotation_file = omics_annotation_file
        self.omics_annotation: pd.DataFrame = pd.DataFrame([])
        self.string_aliases_file = string_aliases_file
        self.string_aliases: pd.DataFrame = pd.DataFrame([])
        self.string_edges_file = string_edges_file
        self.string_edges: pd.DataFrame = pd.DataFrame([])
        self.confidence_level = confidence_level
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.graph_file_name = self.output_dir.joinpath(graph_file_name)
        self.curvature_file_name = self.output_dir.joinpath(curvature_file_name)
        self.n_procs = n_procs
        self.make_pytorch_graphs = make_pytorch_graphs
        self.edges_df: pd.DataFrame = pd.DataFrame([])
        self.G: nx.Graph
        self.G = nx.Graph()
        self.edge_curvatures: pd.DataFrame
        self.edge_curvatures = pd.DataFrame([])
        self.edge_curvatures_dict = dict()
        self.orc: GraphCurvature
        self.orc = None
        self.gene_to_pt_ind = dict()
        self.edge_curvatures_file_path = self.output_dir.joinpath('pt_graphs').joinpath('pt_edge_curvatures.csv')
        self.pt_graphs_path = self.output_dir.joinpath('pt_graphs')

    def __str__(self):
        return """Base class for building graphs from STRING database and 'omics' datasets.
        """

    def execute(self):
        r"""Main function for executing class"""

        print("Converting gene symbols...")
        self._convert_gene_symbols()

        print("Constructing NetworkX graph...")
        self._construct_networkx_graph()

        print("Computing edge curvatures...")
        self.compute_edge_curvatures()

        if self.make_pytorch_graphs:
            print("Converting to PyTorch Geometric and writing individual graphs...")
            self._pytorch_preprocess()
            self._convert_to_pytorch()
            self._save_edge_curvatures_pytorch_format()

    def _convert_gene_symbols(self) -> None:
        r"""STRING database reports protein-protein interactions (PPIs) using their own ID's.
        They also provide an enormous table containing conversions between various gene
        symbols/IDs. Use this table to convert PPIs to gene symbols.

        :return: None
        """
        # Read in "omics" data and make sure it fits the expected format
        omics_data = dd.read_csv(self.omics_data_file)
        if 'gene' not in omics_data.columns:
            print('\tExpected omics_data file format:')
            print('\tgene\tsample1\tsample2\t...')
            print('\t"gene" column not detected. Searching for "symbol" column to use instead.')
            omics_data['gene'] = omics_data['symbol']

        # Prepare omics data for computation
        drop_columns = ['EntrezID', 'symbol', 'gene_name']
        omics_data = omics_data.drop(columns=drop_columns, errors='ignore')

        # Prepare STRING database file for computation
        string_aliases = dd.read_csv(self.string_aliases_file, sep='\t')
        string_aliases = string_aliases[['#string_protein_id', 'alias']]
        aliases = string_aliases.drop_duplicates().compute()
        aliases_trim = aliases.merge(omics_data[['gene']].compute(), left_on='alias', right_on='gene').drop_duplicates(
            subset='#string_protein_id').drop(columns='alias')

        # Prepare nodes and edges for graph
        links = dd.read_csv(self.string_edges_file, sep=' ')
        merges = links.merge(aliases_trim, left_on='protein1', right_on='#string_protein_id').merge(
            aliases_trim,
            left_on='protein2',
            right_on='#string_protein_id'
        )
        edges_df = merges.compute()[['gene_x', 'gene_y', 'combined_score']].rename(
            columns={'gene_x': 'gene_1', 'gene_y': 'gene_2'})
        edges_df = edges_df[edges_df['combined_score'] > self.confidence_level]
        nodes = pd.concat([edges_df, edges_df.rename({'gene_1': 'gene_2', 'gene_2': 'gene_1'})])[
            ['gene_1']].drop_duplicates()

        omics_data = omics_data.merge(nodes, left_on='gene', right_on='gene_1', how='inner').drop('gene_1')

        # Make instance attribute pandas type. It's easier this way.
        self.omics_data = pd.DataFrame(omics_data, columns=omics_data.columns)
        self.edges_df = edges_df

    def _construct_networkx_graph(self) -> None:
        r"""Construct NetworkX graph using the PPIs from STRING. Make sure to only
        grab the genes reported in the omics file. No need to add nodes we don't have
        information for.

        :return:
        """
        # Prepare edges for NetworkX graph
        edges_df = self.edges_df
        edges_array = edges_df.to_numpy()
        edges_array[:, 2] = np.array([{'weight': w} for w in edges_array[:, 2]])
        edges = [tuple(_) for _ in edges_array]

        # Prepare nodes for NetworkX graph
        nodes = list(set(edges_df['gene_1'].unique().tolist() + edges_df['gene_2'].unique().tolist()))

        # Remove any nodes that we don't have omics data for. This shouldn't happen, but just in case.

        extra_nodes = set(nodes) - set(self.omics_data['gene'].tolist())
        if n_extra_nodes := len(extra_nodes) > 0:
            print('Removing {} nodes from graph'.format(n_extra_nodes))
            nodes = set(nodes) - extra_nodes

        # Instantiate empty graph
        G = nx.Graph()

        # Populate graph with edges and nodes
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        # Remove self-loops. STRING database shouldn't lead to them, but just to be safe
        G.remove_edges_from(nx.selfloop_edges(G))

        # Find the largest connected component and drop the rest
        G_cc = [
            G.subgraph(c).copy() for c in nx.connected_components(G) if c == max(nx.connected_components(G), key=len)
        ][0]

        self.G = G_cc

        # Save graph as pickle
        nx.write_gpickle(self.G, self.graph_file_name)

    def compute_edge_curvatures(self) -> None:
        r"""Compute edge curvatures *only*. The full curvature calculation, including nodal curvatures
        and total curvature, is not necessary here. The edge curvatures are computed so that they can be
        passed into the RGCN in the message passing layer.

        :return: None
        """
        # Instantiate GraphCurvature class with NetworkX graph
        orc = GraphCurvature(self.G, n_procs=self.n_procs)

        # Compute edge curvatures from NetworkX graph
        orc.compute_edge_curvatures()

        # Store the edge curvature results on disk and as an instance attribute
        edge_curvatures = orc.edge_curvatures

        # Quick-and-dirty dictionary-to-dataframe conversion. This works better than the
        # pandas built-in.
        edge_curvatures_df = pd.DataFrame(list(edge_curvatures.keys()))
        edge_curvatures_df['curvature'] = edge_curvatures.values()
        edge_curvatures_df.rename(columns={0: 'gene_1', 1: 'gene_2'})

        # Save to disk
        edge_curvatures_df.to_csv(self.curvature_file_name)

        # Set instance attribute to dataframe. It's more convenient than the dictionary.
        self.edge_curvatures = edge_curvatures_df
        self.edge_curvatures_dict = edge_curvatures
        self.orc = orc

    def compute_nodal_curvatures(self) -> None:
        r"""Finish graph curvature calculation after edge curvatures are computed

        :return:
        """
        # If edge_curvatures were not previously computed, compute them.
        if not self.edge_curvatures.empty:
            self.compute_edge_curvatures()

        # Compute nodal_curvatures
        self.orc.compute()

    def _pytorch_preprocess(self) -> None:
        r"""Preprocessing step. Goal is to convert from a base NetworkX graph to a set of
        pytorch graphs that encode omics and response data. Prepare omics data for the
        expected pyg format. Convert NetworkX graph to pyg.

        :return: None
        """
        # Make dataframe containing all nodes from graph
        G_df = pd.DataFrame(self.G.nodes, columns=['gene']).reset_index()

        # Merge omics data into node dataframe. This puts the omics data in the order expected by pytorch
        self.tpm_df = G_df.merge(self.omics_data, left_on='gene', right_on='gene').dropna().drop_duplicates(
            subset=['gene'])

        # Create conversion dictionary between gene symbols and indices
        ind_to_gene = pd.DataFrame(self.G.nodes, columns=['gene']).reset_index()[['gene', 'index']].to_dict()[
            'gene']
        self.gene_to_pt_ind = {v: k for k, v in ind_to_gene.items()}

        # Create a pytorch graph from the base graph
        self.Gp = from_networkx(self.G)

    def _save_edge_curvatures_pytorch_format(self) -> None:
        r"""Save edge curvatures using the format that matches pyg. Currently, edge curvatures
        are stored non-redundantly. Pyg stores all permutations of edges.

        E.g.,
            NetworkX                   pyg
        (node1, node2): x12 --> (node1, node2): x12
                                (node2, node1): x12

        :return: None
        """
        edge_curvatures_1 = self.edge_curvatures
        edge_curvatures = pd.concat([
            edge_curvatures_1,
            edge_curvatures_1.rename(columns={'gene_1': 'gene_2', 'gene_2': 'gene_1'})
        ])
        edge_curvatures.columns = ['gene_1', 'gene_2', 'curvature']
        edge_curvatures['ind1'] = edge_curvatures['gene_1'].apply(
            lambda x: self.gene_to_pt_ind[x] if x in self.gene_to_pt_ind else np.nan)
        edge_curvatures['ind2'] = edge_curvatures['gene_2'].apply(
            lambda x: self.gene_to_pt_ind[x] if x in self.gene_to_pt_ind else np.nan)
        edge_curvatures.dropna(inplace=True)

        edge_curvatures.sort_values(by=['ind1', 'ind2'])[['ind1', 'ind2', 'curvature']].to_csv(
            self.output_dir.joinpath('pt_graphs').joinpath('pt_edge_curvatures.csv'), index=False, header=False)

    @abstractmethod
    def _convert_to_pytorch(self):
        """Each dataset is a little different. Use one of the boilerplate classes in
        'graph_builder.py' to see an example of what should happen in this class.
        """
        return

import os
from pathlib import Path

import dask.dataframe as dd
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils.convert import from_networkx

from graphox.graph_curvature.curvature import GraphCurvature


class GraphBuilder(object):

    def __init__(self,
                 omics_data_file: str,
                 omics_annotation_file: str,
                 string_aliases_file: str,
                 string_edges_file: str,
                 confidence_level: int = 900,
                 output_dir: str = 'output',
                 graph_file_name: str = 'G.pkl',
                 curvature_file_name: str = 'curvatures.csv',
                 n_procs: int = 4
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
        self.edges_df: pd.DataFrame = pd.DataFrame([])
        self.G = None
        self.edge_curvatures = None

    def execute(self):
        print("Converting gene symbols...")
        self.convert_gene_symbols()
        print("Constructing NetworkX graph...")
        self.construct_networkx_graph()
        print("Computing edge curvatures...")
        self.compute_edge_curvatures()
        print("Converting to PyTorch Geometric and writing individual graphs...")
        self.convert_to_pytorch()

    def convert_gene_symbols(self):
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

        self.omics_data = omics_data
        self.edges_df = edges_df

    def construct_networkx_graph(self):
        # Prepare edges for NetworkX graph
        edges_df = self.edges_df
        edges_df = edges_df[edges_df['combined_score'] > self.confidence_level]
        edges_array = edges_df.to_numpy()
        edges_array[:, 2] = np.array([{'weight': w} for w in edges_array[:, 2]])
        edges = [tuple(_) for _ in edges_array]

        # Prepare nodes for NetworkX graph
        nodes = list(set(edges_df['gene_1'].unique().tolist() + edges_df['gene_2'].unique().tolist()))

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

        # Save graph as pickle
        nx.write_gpickle(G_cc, self.graph_file_name)
        self.G = G_cc

    def compute_edge_curvatures(self):
        orc = GraphCurvature(self.G, n_procs=self.n_procs)
        orc.compute_edge_curvatures()
        edge_curvatures = orc.edge_curvatures
        edge_curvatures_df = pd.DataFrame(list(edge_curvatures.keys()))
        edge_curvatures_df['curvature'] = edge_curvatures.values()
        edge_curvatures_df.rename(columns={0: 'gene_1', 1: 'gene_2'})
        edge_curvatures_df.to_csv(self.curvature_file_name)
        self.edge_curvatures = edge_curvatures_df

    def convert_to_pytorch(self):
        omics_data = pd.Dataframe(self.omics_data)
        extra_nodes = set(self.G.nodes) - set(omics_data['gene'].tolist())
        self.G.remove_nodes_from(extra_nodes)
        if n_extra_nodes := len(extra_nodes) > 0:
            print('Removing {} nodes from graph'.format(n_extra_nodes))

        G_df = pd.DataFrame(self.G.nodes, columns=['gene']).reset_index()
        tpm_df = G_df.merge(omics_data, left_on='gene', right_on='symbol').dropna().drop_duplicates(
            subset=['gene'])

        ind_to_gene = pd.DataFrame(self.G.nodes, columns=['gene']).reset_index()[['gene', 'index']].to_dict()[
            'gene']
        gene_to_ind = {v: k for k, v in ind_to_gene.items()}

        Gp = from_networkx(self.G)
        Gp_df = pd.DataFrame(Gp.to_dict()['edge_index'].T, columns=['gene_1', 'gene_2'])
        Gp_df['gene_1'] = Gp_df['gene_1'].apply(lambda x: ind_to_gene[x])
        Gp_df['gene_2'] = Gp_df['gene_2'].apply(lambda x: ind_to_gene[x])
        Gp_df.sort_values(by=['gene_1', 'gene_2'])
        df_anno = pd.read_csv(self.omics_annotation_file, index_col=0)

        df_anno['response'] = df_anno['OBJECTIVE_RESPONSE'].apply(lambda x: 1.0 if x in ['CR', 'PR'] else 0.0)

        self.output_dir.joinpath('pt_graphs').mkdir(parents=True, exist_ok=True)
        for col in tpm_df.columns[4:]:
            G_i = None
            Gt = None
            G_i = Gp
            x = torch.tensor(np.array([tpm_df[col].tolist()]).T, dtype=torch.float)
            G_i['x'] = x
            y = torch.tensor(df_anno[df_anno['RNASEQ_SAMPLE_ID'] == col]['response'].to_numpy(), dtype=torch.float)
            G_i['y'] = y
            Gt = G_i
            torch.save(Gt, self.output_dir.joinpath('pt_graphs').joinpath('G_{}.pt'.format(col)))

        edge_curvs_1 = self.edge_curvatures
        edge_curvs = pd.concat([
            edge_curvs_1,
            edge_curvs_1.rename(columns={'gene_1': 'gene_2', 'gene_2': 'gene_1'})
        ])
        edge_curvs['ind1'] = edge_curvs['gene_1'].apply(lambda x: gene_to_ind[x] if x in gene_to_ind else np.nan)
        edge_curvs['ind2'] = edge_curvs['gene_2'].apply(lambda x: gene_to_ind[x] if x in gene_to_ind else np.nan)
        edge_curvs.dropna(inplace=True)

        edge_curvs.sort_values(by=['ind1', 'ind2'])[['ind1', 'ind2', 'curvature']].to_csv(
            self.output_dir.joinpath('pt_graphs').joinpath('pt_edge_curvatures.csv'), index=False, header=False)


if __name__ == "__main__":
    root_dir = 'data/raw/'
    omics_data_ = os.path.join(root_dir, 'full_data_expr_G.csv')
    omics_anno_ = os.path.join(root_dir, 'full_data_anno.csv')
    string_aliases_file_ = os.path.join(root_dir, '9606.protein.aliases.v11.5.txt')
    string_edges_file_ = os.path.join(root_dir, '9606.protein.links.v11.5.txt')

    builder = GraphBuilder(omics_data_, omics_anno_, string_aliases_file_, string_edges_file_)
    builder.execute()

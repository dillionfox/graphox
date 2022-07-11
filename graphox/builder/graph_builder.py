from graphox.graph_curvature.curvature import GraphCurvature
import pandas as pd
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils.convert import from_networkx


class GraphBuilder(object):

    def __init__(self,
                 omics_data_file: str,
                 omics_annotation_file: str,
                 string_aliases_file: str,
                 string_edges_file: str,
                 confidence_level: int = 900,
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
        self.graph_file_name = graph_file_name
        self.curvature_file_name = curvature_file_name
        self.n_procs = n_procs
        self.G = None
        self.edge_curvatures = None

    def execute(self):
        print("Converting gene symbols...")
        self.convert_gene_symbols()
        print("Constructing NetworkX graph...")
        self.construct_networkx_graph()
        print("Computing edge curvatures")
        self.compute_edge_curvatures()

    def convert_gene_symbols(self):
        # Read in "omics" data and make sure it fits the expected format
        omics_data = pd.read_csv(self.omics_data_file)
        if 'gene' not in omics_data.columns:
            print('Expected omics_data file format:')
            print('gene\tsample1\tsample2\t...')
            print('"gene" column not detected. Searching for "symbol" column to use instead.')
            omics_data['gene'] = omics_data['symbol']

        drop_columns = ['EntrezID', 'symbol', 'gene_name']
        omics_data.drop(columns=drop_columns, errors='ignore', inplace=True)
        omics_data_converted = pd.DataFrame(omics_data['gene'].drop_duplicates())

        # Read in STRING database aliases
        self.string_aliases = pd.read_csv(self.string_aliases_file, sep='\t')

        # Convert IDs
        omics_data_converted['convs'] = omics_data_converted['gene'].apply(
            lambda x: self.string_aliases[
                self.string_aliases['alias'] == x
                ]['#string_protein_id'].tolist()[0] if x in self.string_aliases['alias'].tolist() else np.nan)
        omics_data_converted.dropna(inplace=True)

        self.omics_data = omics_data_converted

    def construct_networkx_graph(self):
        links = pd.read_csv(self.string_edges_file, sep=' ')
        links = links[links['combined_score'] > self.confidence_level]
        string_to_gene = self.omics_data.set_index('convs').to_dict()['gene']
        links['gene1'] = links['protein1'].apply(lambda x: string_to_gene[x] if x in string_to_gene.keys() else np.nan)
        links['gene2'] = links['protein2'].apply(lambda x: string_to_gene[x] if x in string_to_gene.keys() else np.nan)
        links.dropna(inplace=True)
        links = links[['gene1', 'gene2', 'combined_score']].sort_values(by='combined_score').drop_duplicates(
            subset=['gene1', 'gene2'], keep='last')
        # Construct list of tuples to define edges
        edges = [(_[0], _[1], {'weight': _[2]}) for _ in
                 list(zip(links['gene1'].tolist(), links['gene2'].tolist(), links['combined_score']))]
        # Construct NetworkX graph
        G = nx.Graph()
        # Set nodes
        G.add_nodes_from(self.omics_data['gene'].tolist())
        # Define edges
        G.add_edges_from(edges)
        # Pull out largest connected component
        G_cc = [G.subgraph(c).copy() for c in nx.connected_components(G) if c == max(nx.connected_components(G),
                                                                                     key=len)][0]
        # Save graph as pickle
        nx.write_gpickle(G_cc, self.graph_file_name)
        self.G = G_cc

    def compute_edge_curvatures(self):
        orc = GraphCurvature(self.G, n_procs=self.n_procs)
        orc.compute_edge_curvatures()
        self.edge_curvatures = orc.edge_curvatures
        self.edge_curvatures.to_csv(self.curvature_file_name)

    def convert_to_pytorch(self):
        try_pytorch = False
        if try_pytorch:
            tpm_df0 = pd.read_csv('/Users/dfox/data/immotion/data/full_data_expr_G.csv', index_col=0).drop_duplicates(
                subset=['symbol'])

            self.G = nx.read_gpickle('/Users/dfox/code/graphox/notebooks/high_conf_G.pkl')
            extra_nodes = set(self.G.nodes) - set(tpm_df0['symbol'].tolist())
            self.G.remove_nodes_from(extra_nodes)

            G_df = pd.DataFrame(self.G.nodes, columns=['gene']).reset_index()
            tpm_df = G_df.merge(tpm_df0, left_on='gene', right_on='symbol').dropna().drop_duplicates(subset=['gene'])

            ind_to_gene = pd.DataFrame(self.G.nodes, columns=['gene']).reset_index()[['gene', 'index']].to_dict()[
                'gene']
            gene_to_ind = {v: k for k, v in ind_to_gene.items()}

            Gp = from_networkx(self.G)
            Gp_df = pd.DataFrame(Gp.to_dict()['edge_index'].T, columns=['gene1', 'gene2'])
            Gp_df['gene1'] = Gp_df['gene1'].apply(lambda x: ind_to_gene[x])
            Gp_df['gene2'] = Gp_df['gene2'].apply(lambda x: ind_to_gene[x])
            Gp_df.sort_values(by=['gene1', 'gene2'])
            df_anno = pd.read_csv('/Users/dfox/data/immotion/data/full_data_anno.csv', index_col=0)

            df_anno['response'] = df_anno['OBJECTIVE_RESPONSE'].apply(lambda x: 1.0 if x in ['CR', 'PR'] else 0.0)

            for col in tpm_df.columns[4:]:
                G_i = None
                Gt = None
                G_i = Gp
                x = torch.tensor(np.array([tpm_df[col].tolist()]).T, dtype=torch.float)
                G_i['x'] = x
                y = torch.tensor(df_anno[df_anno['RNASEQ_SAMPLE_ID'] == col]['response'].to_numpy(), dtype=torch.float)
                G_i['y'] = y
                Gt = G_i
                torch.save(Gt, 'data/G_{}.pt'.format(col))

            edge_curvs_1 = pd.read_csv('../data/high_conf_edge_curvatures.csv', index_col=0)
            edge_curvs = pd.concat([edge_curvs_1, edge_curvs_1.rename(columns={'gene1': 'gene2', 'gene2': 'gene1'})])
            edge_curvs['ind1'] = edge_curvs['gene1'].apply(lambda x: gene_to_ind[x] if x in gene_to_ind else np.nan)
            edge_curvs['ind2'] = edge_curvs['gene2'].apply(lambda x: gene_to_ind[x] if x in gene_to_ind else np.nan)
            edge_curvs.dropna(inplace=True)

            edge_curvs.sort_values(by=['ind1', 'ind2'])[['ind1', 'ind2', 'curvature']].to_csv(
                'immotion_edge_curvatures_high.csv', index=False, header=False)
            edge_curvs[edge_curvs['curvature'].abs() > 0.2].sort_values(by=['ind1', 'ind2'])[
                ['ind1', 'ind2', 'curvature']].to_csv('immotion_edge_curvatures_high_strong.csv', index=False,
                                                      header=False)
            edge_curvs[edge_curvs['curvature'].abs() > 0.3].sort_values(by=['ind1', 'ind2'])[
                ['ind1', 'ind2', 'curvature']].to_csv('immotion_edge_curvatures_high_stronger.csv', index=False,
                                                      header=False)

        else:
            raise NotImplementedError()


if __name__ == "__main__":
    omics_data_ = '/Users/dfox/code/graphox/data/raw/full_data_expr_G.csv'
    omics_anno_ = '/Users/dfox/code/graphox/data/raw/full_data_anno.csv'
    string_aliases_file_ = '/Users/dfox/code/graphox/data/raw/9606.protein.aliases.v11.5.txt'
    string_edges_file_ = '/Users/dfox/code/graphox/data/raw/sample_links.txt'

    builder = GraphBuilder(omics_data_, omics_anno_, string_aliases_file_, string_edges_file_)
    builder.execute()

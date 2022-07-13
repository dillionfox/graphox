import os

import click
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils.convert import from_networkx

from graphox.builder.base import BaseGraphBuilder


class ImMotionGraphBuilder(BaseGraphBuilder):
    def __init__(
            self,
            omics_data_file: str,
            omics_annotation_file: str,
            string_aliases_file: str,
            string_edges_file: str,
            make_pytorch_graphs: bool = False,
            n_procs: int = 4,
    ):
        super().__init__(omics_data_file, omics_annotation_file,
                         string_aliases_file, string_edges_file,
                         make_pytorch_graphs=make_pytorch_graphs,
                         n_procs=n_procs)

    def convert_to_pytorch(self):
        omics_data = pd.DataFrame(self.omics_data, columns=self.omics_data.columns)
        extra_nodes = set(self.G.nodes) - set(omics_data['gene'].tolist())
        self.G.remove_nodes_from(extra_nodes)
        if n_extra_nodes := len(extra_nodes) > 0:
            print('Removing {} nodes from graph'.format(n_extra_nodes))

        G_df = pd.DataFrame(self.G.nodes, columns=['gene']).reset_index()
        tpm_df = G_df.merge(omics_data, left_on='gene', right_on='gene').dropna().drop_duplicates(
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
            G_i = Gp
            x = torch.tensor(np.array([tpm_df[col].tolist()]).T, dtype=torch.float)
            G_i['x'] = x
            y = torch.tensor(df_anno[df_anno['RNASEQ_SAMPLE_ID'] == col]['response'].to_numpy(), dtype=torch.float)
            G_i['y'] = y
            Gt = G_i
            torch.save(Gt, self.output_dir.joinpath('pt_graphs').joinpath('G_{}.pt'.format(col)))

        edge_curvatures_1 = self.edge_curvatures
        edge_curvatures = pd.concat([
            edge_curvatures_1,
            edge_curvatures_1.rename(columns={'gene_1': 'gene_2', 'gene_2': 'gene_1'})
        ])
        edge_curvatures.columns = ['gene_1', 'gene_2', 'curvature']
        edge_curvatures['ind1'] = edge_curvatures['gene_1'].apply(
            lambda x: gene_to_ind[x] if x in gene_to_ind else np.nan)
        edge_curvatures['ind2'] = edge_curvatures['gene_2'].apply(
            lambda x: gene_to_ind[x] if x in gene_to_ind else np.nan)
        edge_curvatures.dropna(inplace=True)

        edge_curvatures.sort_values(by=['ind1', 'ind2'])[['ind1', 'ind2', 'curvature']].to_csv(
            self.output_dir.joinpath('pt_graphs').joinpath('pt_edge_curvatures.csv'), index=False, header=False)


class TCatGraphBuilder(BaseGraphBuilder):
    def __init__(
            self,
            omics_data_file: str,
            omics_annotation_file: str,
            string_aliases_file: str,
            string_edges_file: str,
            make_pytorch_graphs: bool = False,
            n_procs: int = 4,
    ):
        super().__init__(omics_data_file, omics_annotation_file,
                         string_aliases_file, string_edges_file,
                         make_pytorch_graphs=make_pytorch_graphs,
                         n_procs=n_procs)

    def convert_to_pytorch(self):
        omics_data = pd.DataFrame(self.omics_data, columns=self.omics_data.columns)
        extra_nodes = set(self.G.nodes) - set(omics_data['gene'].tolist())
        self.G.remove_nodes_from(extra_nodes)
        if n_extra_nodes := len(extra_nodes) > 0:
            print('Removing {} nodes from graph'.format(n_extra_nodes))

        G_df = pd.DataFrame(self.G.nodes, columns=['gene']).reset_index()
        tpm_df = G_df.merge(omics_data, left_on='gene', right_on='gene').dropna().drop_duplicates(
            subset=['gene'])

        ind_to_gene = pd.DataFrame(self.G.nodes, columns=['gene']).reset_index()[['gene', 'index']].to_dict()[
            'gene']
        gene_to_ind = {v: k for k, v in ind_to_gene.items()}

        Gp = from_networkx(self.G)
        Gp_df = pd.DataFrame(Gp.to_dict()['edge_index'].T, columns=['gene_1', 'gene_2'])
        Gp_df['gene_1'] = Gp_df['gene_1'].apply(lambda x: ind_to_gene[x])
        Gp_df['gene_2'] = Gp_df['gene_2'].apply(lambda x: ind_to_gene[x])
        Gp_df.sort_values(by=['gene_1', 'gene_2'])

        self.output_dir.joinpath('pt_graphs').mkdir(parents=True, exist_ok=True)
        for col in tpm_df.columns[4:]:
            G_i = Gp
            x = torch.tensor(np.array([tpm_df[col].tolist()]).T, dtype=torch.float)
            G_i['x'] = x
            G_i['y'] = torch.tensor(1.0, dtype=torch.float)
            Gt = G_i
            torch.save(Gt, self.output_dir.joinpath('pt_graphs').joinpath('G_{}.pt'.format(col)))

        edge_curvatures_1 = self.edge_curvatures
        edge_curvatures = pd.concat([
            edge_curvatures_1,
            edge_curvatures_1.rename(columns={'gene_1': 'gene_2', 'gene_2': 'gene_1'})
        ])
        edge_curvatures.columns = ['gene_1', 'gene_2', 'curvature']
        edge_curvatures['ind1'] = edge_curvatures['gene_1'].apply(
            lambda x: gene_to_ind[x] if x in gene_to_ind else np.nan)
        edge_curvatures['ind2'] = edge_curvatures['gene_2'].apply(
            lambda x: gene_to_ind[x] if x in gene_to_ind else np.nan)
        edge_curvatures.dropna(inplace=True)

        edge_curvatures.sort_values(by=['ind1', 'ind2'])[['ind1', 'ind2', 'curvature']].to_csv(
            self.output_dir.joinpath('pt_graphs').joinpath('pt_edge_curvatures.csv'), index=False, header=False)


@click.command()
@click.option('--dataset', default='immotion', help='Pre-defined test dataset')
@click.option('--n_procs', default=4, help='Pre-defined test dataset')
def main(dataset: str, n_procs: int):
    root_dir = 'data/raw/'

    if dataset not in ['immotion', 'tcat']:
        print('Dataset not recognized')
        exit()

    omics_data_ = os.path.join(root_dir, '{}/counts.csv'.format(dataset))
    omics_anno_ = os.path.join(root_dir, '{}/anno.csv'.format(dataset))
    string_aliases_file_ = os.path.join(root_dir, 'string/9606.protein.aliases.v11.5.txt')
    string_edges_file_ = os.path.join(root_dir, 'string/9606.protein.links.v11.5.txt')

    if dataset == 'immotion':
        builder = ImMotionGraphBuilder(omics_data_, omics_anno_, string_aliases_file_, string_edges_file_,
                                       make_pytorch_graphs=True, n_procs=n_procs)
    elif dataset == 'tcat':
        builder = TCatGraphBuilder(omics_data_, omics_anno_, string_aliases_file_, string_edges_file_,
                                   make_pytorch_graphs=False, n_procs=n_procs)
    else:
        raise NotImplementedError()

    builder.execute()


if __name__ == "__main__":
    main()

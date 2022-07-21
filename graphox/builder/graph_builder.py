"""Classes for building graphs for graph machine learning
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

import os

import click
import numpy as np
import pandas as pd
import torch

from graphox.builder.base import CurvatureGraphBuilder


class ImMotionCurvatureGraphBuilder(CurvatureGraphBuilder):
    def __init__(
            self,
            omics_data_file: str,
            omics_annotation_file: str,
            string_aliases_file: str,
            string_edges_file: str,
            confidence_level: int = 900,
            make_pytorch_graphs: bool = False,
            n_procs: int = 4,
    ):
        super().__init__(omics_data_file, omics_annotation_file,
                         string_aliases_file, string_edges_file,
                         confidence_level=confidence_level,
                         make_pytorch_graphs=make_pytorch_graphs,
                         n_procs=n_procs)

    def __str__(self):
        return r"""Class designed to build a graph for the 'ImMotion' dataset
        """

    def _create_patient_graphs(self) -> None:
        r"""Final step of converting base NetworkX graph to a set of pyg graphs.
        Store patient/sample omics data and response as graph attribute. Save
        graph for each patient/sample.

        :return: None
        """
        # Read in omics annotation information, including responses
        df_anno = pd.read_csv(self.omics_annotation_file, index_col=0)
        df_anno['response'] = df_anno['OBJECTIVE_RESPONSE'].apply(lambda x: 1.0 if x in ['CR', 'PR'] else 0.0)

        # Create a new pyg graph for each patient and store omics + response as attributes
        self.pt_graphs_path.mkdir(parents=True, exist_ok=True)
        for col in self.tpm_df.drop(columns=['gene']).columns:
            # Make a copy
            G_i = self.Gp
            # Convert 1d omics counts to pytorch tensor
            x = torch.tensor(np.array([self.tpm_df[col].tolist()]).T, dtype=torch.float)
            # Assign omics values to graph attribute
            G_i['x'] = x
            # Convert response to pytorch tensor
            y = torch.tensor(df_anno[df_anno['RNASEQ_SAMPLE_ID'] == col]['response'].to_numpy(), dtype=torch.float)
            # Store patient/sample response/label as graph attribute
            G_i['y'] = y
            if len(y) == 0:
                print('wtf??', y)
                continue
            # Save patient/sample graph
            torch.save(G_i, self.pt_graphs_path.joinpath('G_{}.pt'.format(col)))


class TCatCurvatureGraphBuilder(CurvatureGraphBuilder):
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

    def __str__(self):
        return r"""Class designed to build a graph for the 'tcat' dataset
        """

    def _create_patient_graphs(self):
        r"""Final step of converting base NetworkX graph to a set of pyg graphs.
        Store patient/sample omics data and response as graph attribute. Save
        graph for each patient/sample. Tcat dataset doesn't really have response
        data, so mock it by assigning '1' for all samples. This dataset shouldn't
        be used for classification, anyway.

        :return: None
        """
        # Create a new pyg graph for each patient and store omics + response as attributes
        self.pt_graphs_path.mkdir(parents=True, exist_ok=True)
        for col in self.tpm_df.drop(columns=['gene']).columns:
            # Make a copy
            G_i = self.Gp
            # Convert 1d omics counts to pytorch tensor
            x = torch.tensor(np.array([self.tpm_df[col].tolist()]).T, dtype=torch.float)
            # Assign omics values to graph attribute
            G_i['x'] = x
            # Store patient/sample response/label as graph attribute
            G_i['y'] = torch.tensor(1.0, dtype=torch.float)
            # Save patient/sample graph
            torch.save(G_i, self.pt_graphs_path.joinpath('G_{}.pt'.format(col)))


@click.command()
@click.option('--dataset', default='immotion', help='Pre-defined test dataset')
@click.option('--n_procs', default=4, help='Pre-defined test dataset')
def main(dataset: str, n_procs: int) -> None:
    r"""Basic command-line interface for building graphs.

    :param dataset: str, special keyword to determine which dataset is used.
    :param n_procs: int, number of processors to use in edge curvature calculation
    :return: None
    """

    # Set root directory for finding files. This is specific to how I set up my directories.
    root_dir = 'data/raw/'

    # Only the following datasets are supported by this function.
    if dataset not in ['immotion', 'tcat']:
        print('Dataset not recognized')
        exit()

    # Read in omics data, response data, and STRING database.
    omics_data_ = os.path.join(root_dir, '{}/counts.csv'.format(dataset))
    omics_anno_ = os.path.join(root_dir, '{}/anno.csv'.format(dataset))
    string_aliases_file_ = os.path.join(root_dir, 'string/9606.protein.aliases.v11.5.txt')
    string_edges_file_ = os.path.join(root_dir, 'string/9606.protein.links.v11.5.txt')

    # Instantiate builder class
    if dataset == 'immotion':
        builder = ImMotionCurvatureGraphBuilder(omics_data_, omics_anno_, string_aliases_file_, string_edges_file_,
                                                make_pytorch_graphs=True, n_procs=n_procs)
    elif dataset == 'tcat':
        builder = TCatCurvatureGraphBuilder(omics_data_, omics_anno_, string_aliases_file_, string_edges_file_,
                                            make_pytorch_graphs=False, n_procs=n_procs)
    else:
        raise NotImplementedError()

    # Build the graphs
    builder.execute()


if __name__ == "__main__":
    main()

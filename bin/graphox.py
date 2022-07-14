import os

import click

from ..graphox.builder.graph_builder import ImMotionGraphBuilder
from ..graphox.graph_curvature.curvature import compute_nodal_curvatures
from ..graphox.rgcn.rgcn import rgcn_trainer


def _run_rgcn(dataset: str, n_procs: int):
    # Set paths, choose project
    root_dir = 'data/raw/'

    # Make sure predefined project is chosen
    if dataset not in ['immotion', 'tcat']:
        print('Dataset not recognized')
        exit()

    # Set paths to omics data and STRING database files
    omics_data_ = os.path.join(root_dir, '{}/counts.csv'.format(dataset))
    omics_anno_ = os.path.join(root_dir, '{}/anno.csv'.format(dataset))
    string_aliases_file_ = os.path.join(root_dir, 'string/9606.protein.aliases.v11.5.txt')
    string_edges_file_ = os.path.join(root_dir, 'string/9606.protein.links.v11.5.txt')

    # Instantiate builder object to build graphs
    builder = ImMotionGraphBuilder(omics_data_, omics_anno_, string_aliases_file_, string_edges_file_,
                                   make_pytorch_graphs=True, n_procs=n_procs)
    # Build graphs and write output to disk
    builder.execute()

    # Run RGCN model
    rgcn_trainer(builder.pt_graphs_path, 2, builder.edge_curvatures_file_path)


def _compute_curvature(dataset: str, n_procs: int):
    # Set paths, choose project
    root_dir = 'data/raw/'

    # Make sure predefined project is chosen
    if dataset not in ['immotion', 'tcat']:
        print('Dataset not recognized')
        exit()

    # Set paths to omics data and STRING database files
    omics_data_ = os.path.join(root_dir, '{}/counts.csv'.format(dataset))
    omics_anno_ = os.path.join(root_dir, '{}/anno.csv'.format(dataset))
    string_aliases_file_ = os.path.join(root_dir, 'string/9606.protein.aliases.v11.5.txt')
    string_edges_file_ = os.path.join(root_dir, 'string/9606.protein.links.v11.5.txt')

    # Instantiate builder object to build graphs
    builder = ImMotionGraphBuilder(omics_data_, omics_anno_, string_aliases_file_, string_edges_file_,
                                   make_pytorch_graphs=False, n_procs=n_procs)

    # Build graphs and write output to disk
    builder.execute()

    # Compute nodal curvatures
    builder.compute_nodal_curvatures()

    # Compute curvature per patient and total curvature
    curvature_per_patient, nodal_curvature = compute_nodal_curvatures(builder.orc, builder.omics_data)
    print(curvature_per_patient)


@click.command()
@click.option('--dataset', default='immotion', help='Pre-defined test dataset')
@click.option('--n_procs', default=4, help='Pre-defined test dataset')
@click.option('--compute_curvature', is_flag=True)
@click.option('--run_rgcn', is_flag=True)
def main(dataset: str, n_procs: int, compute_curvature, run_rgcn):
    # Make sure predefined project is chosen
    if dataset not in ['immotion', 'tcat']:
        print('Dataset not recognized')
        exit()

    if compute_curvature:
        _compute_curvature(dataset, n_procs)

    if run_rgcn:
        _run_rgcn(dataset, n_procs)


if __name__ == "__main__":
    main()

import os

import click

from graphox.builder.graph_builder import ImMotionCurvatureGraphBuilder
from graphox.graph_curvature.curvature import compute_nodal_curvatures
from graphox.rgcn.rgcn import train_rgcn
from pathlib import Path


output_dir: Path = Path('output')


def _run_rgcn(dataset: str, conf: int, n_procs: int, config: dict):
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
    builder = ImMotionCurvatureGraphBuilder(omics_data_, omics_anno_, string_aliases_file_, string_edges_file_,
                                            confidence_level=conf, make_pytorch_graphs=True, n_procs=n_procs)

    # Run RGCN model
    config['graphs_path'] = builder.pt_graphs_path
    config['curvature_file'] = builder.edge_curvatures_file_path
    train_rgcn(config)


def _compute_curvature(dataset: str, conf: int, n_procs: int):
    # Set paths, choose project
    root_dir = 'data/raw/'
    total_curvature_output = output_dir.joinpath('{}_total_curvatures.csv'.format(dataset))
    nodal_curvatures_output = output_dir.joinpath('{}_nodal_curvatures.csv'.format(dataset))

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
    builder = ImMotionCurvatureGraphBuilder(omics_data_, omics_anno_, string_aliases_file_, string_edges_file_,
                                            confidence_level=conf, make_pytorch_graphs=False, n_procs=n_procs)

    # Build graphs and write output to disk
    builder.execute()

    # Compute nodal curvatures
    builder.compute_nodal_curvatures()

    # Compute curvature per patient and total curvature
    builder.compute_curvature_per_sample()
    print(builder.total_curvature_per_sample)
    print(builder.nodal_curvatures_per_sample)
    builder.total_curvature_per_sample.to_csv(total_curvature_output)
    builder.nodal_curvatures_per_sample.to_csv(nodal_curvatures_output)


@click.command()
@click.option('--dataset', default='immotion', help='Pre-defined test dataset')
@click.option('--n_procs', default=4, help='Pre-defined test dataset')
@click.option('--conf', default=900, help='Pre-defined test dataset')
@click.option('--compute_curvature', is_flag=True)
@click.option('--run_rgcn', is_flag=True)
@click.option('--epochs', default=100, help="Number of epochs")
@click.option('--lr', default=0.05, help="Learning rate for optimizer")
@click.option('--weight_decay', default=0.025, help="Weight decay for optimizer")
@click.option('--momentum', default=0.1, help="Momentum for optimizer")
@click.option('--d_hidden', default=32, help="Dimension for hidden layers")
@click.option('--p', default=0.6, help="Dropout rate in hidden layers")
@click.option('--version', default='v0', help="Version of neural network architecture")
def main(
        dataset: str,
        n_procs: int,
        conf: int,
        compute_curvature,
        run_rgcn,
        epochs,
        lr,
        weight_decay,
        momentum,
        d_hidden,
        p,
        version,
):
    # Make sure predefined project is chosen
    if dataset not in ['immotion', 'tcat']:
        print('Dataset not recognized')
        exit()

    if compute_curvature:
        _compute_curvature(dataset, conf, n_procs)

    if run_rgcn:
        config = {
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "d_hidden": d_hidden,
            "p": p,
            "version": version
        }
        _run_rgcn(dataset, conf, n_procs, config)


if __name__ == "__main__":
    main()

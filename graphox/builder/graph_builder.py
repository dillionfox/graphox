from graphox.builder.base import BaseGraphBuilder
import os
import click


class ImMotionGraphBuilder(BaseGraphBuilder):
    def __init__(self, omics_data_file: str, omics_annotation_file: str, string_aliases_file: str,
                 string_edges_file: str):
        super().__init__(omics_data_file, omics_annotation_file, string_aliases_file, string_edges_file)


class TCatGraphBuilder(BaseGraphBuilder):
    def __init__(self, omics_data_file: str, omics_annotation_file: str, string_aliases_file: str,
                 string_edges_file: str):
        super().__init__(omics_data_file, omics_annotation_file, string_aliases_file, string_edges_file)


@click.command()
@click.option('--dataset', default='immotion', help='Pre-defined test dataset')
def main(dataset: str):
    root_dir = 'data/raw/'

    if dataset not in ['immotion', 'tcat']:
        print('Dataset not recognized')
        exit()

    omics_data_ = os.path.join(root_dir, '{}/counts.csv'.format(dataset))
    omics_anno_ = os.path.join(root_dir, '{}/anno.csv'.format(dataset))
    string_aliases_file_ = os.path.join(root_dir, 'string/9606.protein.aliases.v11.5.txt')
    string_edges_file_ = os.path.join(root_dir, 'string/9606.protein.links.v11.5.txt')

    builder = ImMotionGraphBuilder(omics_data_, omics_anno_, string_aliases_file_, string_edges_file_)
    builder.execute()


if __name__ == "__main__":
    main()

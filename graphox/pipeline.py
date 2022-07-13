from graphox.builder.graph_builder import ImMotionGraphBuilder
from graphox.graph_curvature.curvature import GraphCurvature
from graphox.rgcn.rgcn import rgcn_trainer
import os

# Set paths, choose project
root_dir = 'data/raw/'
dataset = 'immotion'

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
                               make_pytorch_graphs=True)
# Build graphs and write output to disk
builder.execute()

# Run RGCN model
rgcn_trainer(builder.pt_graphs_path, 2, builder.edge_curvatures_file_path)

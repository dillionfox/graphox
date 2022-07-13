from graphox.builder.graph_builder import BaseGraphBuilder
from graphox.graph_curvature.curvature import GraphCurvature
from graphox.rgcn.rgcn import main
import os

root_dir = 'data/raw/'
omics_data_ = os.path.join(root_dir, 'full_data_expr_G.csv')
omics_anno_ = os.path.join(root_dir, 'full_data_anno.csv')
string_aliases_file_ = os.path.join(root_dir, '9606.protein.aliases.v11.5.txt')
string_edges_file_ = os.path.join(root_dir, '9606.protein.links.v11.5.txt')

builder = BaseGraphBuilder(omics_data_, omics_anno_, string_aliases_file_, string_edges_file_)
builder.execute()
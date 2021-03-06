# Graphox: Graph curvature toolkit 
## Graph Curvature / Network Curvature / Ollivier Ricci Curvature
## Curvature Graph Convolutional Neural Networks (coming soon)

Compute Ollivier Ricci curvature (e.g., Graph Curvature, Network Curvature)
on a NetworkX graph object, described in
*Ricci curvature of Markov chains on metric spaces, Yann Ollivier*
https://www.sciencedirect.com/science/article/pii/S002212360800493X?via%3Dihub

This method has been used in numerous studies for estimating patient prognosis
from "omics" data. Examples:

    - https://www.nature.com/articles/srep12323
    - https://www.nature.com/articles/s41525-021-00259-9
    - https://www.biorxiv.org/content/10.1101/2022.03.24.485712v2.full.pdf
    - https://arxiv.org/abs/1502.04512

# Installation
```bash
git clone https://github.com/dillionfox/graphox.git
conda env create -f environment.yml
pip install .
```

# Usage
Please see notebooks/example.ipynb for example usage.

import os
import pathlib

import torch
from torch_geometric.data import InMemoryDataset


class ImMotionDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.root)

    @property
    def processed_file_names(self):
        return os.listdir(self.root)

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = [torch.load(filepath.absolute()) for filepath in pathlib.Path(self.root).glob('*.pt')]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
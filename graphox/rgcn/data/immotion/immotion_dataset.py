"""Classes for building pyg datasets
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
import pathlib
from abc import ABC

import torch
from torch_geometric.data import Dataset


class ImMotionDataset(Dataset, ABC):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __str__(self):
        return r"""Class for building pyg dataset for ImMotion data
        """

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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            data_list = [torch.load(filepath.absolute(), map_location=lambda storage, loc: storage.cuda(0)) for filepath in
                         pathlib.Path(self.root).glob('*.pt')]
        else:
            data_list = [torch.load(filepath.absolute()) for filepath in pathlib.Path(self.root).glob('*.pt')]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

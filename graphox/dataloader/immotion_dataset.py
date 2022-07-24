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
from pathlib import Path
from abc import ABC

import torch
from torch_geometric.data import Dataset, InMemoryDataset
from sklearn.model_selection import train_test_split
import pandas as pd


class ImMotionDataset(Dataset, ABC):
    def __init__(
            self,
            root,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            subset='all',
            test_size: float = 0.2,
            override_split: bool = False
    ):
        self.subset = subset
        self.test_size = test_size
        self.override_split = override_split
        self.file_list = None
        self.train_files = Path(root).joinpath('train_files.csv')
        self.test_files = Path(root).joinpath('test_files.csv')
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        if self.file_list is None:
            self._set_file_list()
        return self.file_list

    @property
    def processed_file_names(self):
        if self.file_list is None:
            self._set_file_list()
        return self.file_list

    def _check_test_train_files(self):
        # Check to see if the test/train files need to be made or remade
        if (not self.train_files.exists() or not self.test_files.exists()) or self.override_split:
            files = list(Path(self.root).glob('G_EA*.pt'))
            y = [torch.load(f).y.numpy()[0] for f in files]
            X_train, X_test, y_train, y_test = train_test_split(files, y, stratify=y, test_size=self.test_size)
            pd.DataFrame(X_train, columns=['filename']).to_csv(self.train_files, columns=['filename'], index=False)
            pd.DataFrame(X_test, columns=['filename']).to_csv(self.test_files, columns=['filename'], index=False)

    def _set_file_list(self):
        if self.subset == 'train':
            train_files = pd.read_csv(self.train_files)['filename'].tolist()
            self.file_list = [_ for _ in Path(self.root).glob('G_EA*.pt') if str(_) in train_files]
        elif self.subset == 'test':
            test_files = pd.read_csv(self.test_files)['filename'].tolist()
            self.file_list = [_ for _ in Path(self.root).glob('G_EA*.pt') if str(_) in test_files]
        else:
            self.file_list = list(Path(self.root).glob('G_EA*.pt'))

    def process(self):
        if self.subset in ['train', 'test']:
            self._check_test_train_files()
        if self.file_list is None:
            self._set_file_list()
        idx = 0
        for raw_path in self.file_list:
            # Read data from `raw_path`.
            data = torch.load(raw_path)
            data.to(device='cuda' if torch.cuda.is_available() else 'cpu')
            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data.to(device='cuda' if torch.cuda.is_available() else 'cpu')


class ImMotionDatasetInMemory(InMemoryDataset, ABC):
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

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    def process(self):
        # Read data into huge `Data` list.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            data_list = [torch.load(filepath.absolute(), map_location=lambda storage, loc: storage.cuda(0)) for filepath
                         in Path(self.root).glob('*.pt')]
        else:
            data_list = [torch.load(filepath.absolute()) for filepath in Path(self.root).glob('*.pt')]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

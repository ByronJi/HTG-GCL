import os
import torch
import pickle
import numpy as np
from definitions import ROOT_DIR

from data.tu_utils import load_data, S2V_to_PyG, get_fold_indices
from data.utils import convert_graph_dataset_with_gudhi, convert_graph_dataset_with_rings
from data.datasets import InMemoryComplexDataset

# get func
from itertools import repeat
import copy
from data.complex import Complex, Cochain


class TUDataset(InMemoryComplexDataset):
    """A dataset of complexes obtained by lifting graphs from TUDatasets."""

    def __init__(self, root, name, max_dim=2, num_classes=2, degree_as_tag=False,
                 init_method='sum', seed=0, include_down_adj=False, max_ring_sizes=None):

        self.name = name
        self.degree_as_tag = degree_as_tag

        # assert max_ring_size is None or max_ring_size >= 3
        # self._max_ring_size = max_ring_size
        # cellular = (max_ring_size is not None)
        # if cellular:
        #     assert max_dim == 2

        if isinstance(max_ring_sizes, int):
            max_ring_sizes = [max_ring_sizes]
        self._max_ring_sizes = max_ring_sizes or [None]
        self._cellular = any(r is not None for r in self._max_ring_sizes)
        cellular = self._cellular


        super(TUDataset, self).__init__(root, max_dim=max_dim, num_classes=num_classes,
            init_method=init_method, include_down_adj=include_down_adj, cellular=cellular)

        # Load processed data (now the entire dataset is loaded)
        self.data, self.slices = torch.load(self.processed_paths[0])

        # Remove the K-fold related code: No fold, train_ids, val_ids
        self.test_ids = None  # No test ids as well

        # 加载并缓存 pyg 格式图
        with open(self.raw_paths[0], 'rb') as handle:
            self.graph_list = pickle.load(handle)  # PyG 的图列表
        self.pyg_graphs = self.graph_list

    @property
    def processed_dir(self):
        """This is overwritten, so the cellular complex data is placed in another folder"""
        directory = super(TUDataset, self).processed_dir
        suffix = f"_{self._max_ring_sizes}rings" if self._cellular else ""
        suffix += f"_down_adj" if self.include_down_adj else ""
        return directory + suffix

    # @property
    # def processed_file_names(self):
    #     return ['{}_complex_list.pt'.format(self.name)]
    @property
    def processed_file_names(self):
        return ['{}_complex_list_{}.pt'.format(self.name, r if r is not None else 'gudhi')
                for r in self._max_ring_sizes]


    @property
    def raw_file_names(self):
        # The processed graph files are our raw files.
        # They are obtained when running the initial data conversion S2V_to_PyG.
        return ['{}_graph_list_degree_as_tag_{}.pkl'.format(self.name, self.degree_as_tag)]

    def download(self):
        # This will process the raw data into a list of PyG Data objs.
        data, num_classes = load_data(self.raw_dir, self.name, self.degree_as_tag)
        self._num_classes = num_classes
        print('Converting graph data into PyG format...')
        graph_list = [S2V_to_PyG(datum) for datum in data]
        with open(self.raw_paths[0], 'wb') as handle:
            pickle.dump(graph_list, handle)
        self.pyg_graphs = self.graph_list

    def process(self):
        with open(self.raw_paths[0], 'rb') as handle:
            graph_list = pickle.load(handle)

        for r in self._max_ring_sizes:
            if r is not None:
                print(f"Converting with max_ring_size={r}")
                complexes, _, _ = convert_graph_dataset_with_rings(
                    graph_list, max_ring_size=r,
                    include_down_adj=self.include_down_adj,
                    init_method=self._init_method,
                    init_edges=True, init_rings=True)
            else:
                print("Converting with gudhi...")
                complexes, _, _ = convert_graph_dataset_with_gudhi(
                    graph_list, expansion_dim=self.max_dim,
                    include_down_adj=self.include_down_adj,
                    init_method=self._init_method)

            save_path = os.path.join(self.processed_dir,
                                     f"{self.name}_complex_list_{r if r is not None else 'gudhi'}.pt")
            torch.save(self.collate(complexes, self.max_dim), save_path)


    def get(self, idx):
        # 初始化缓存
        if not hasattr(self, '__data_list__') or self.__data_list__ is None:
            self.__data_list__ = [None] * self.len()

        cached = self.__data_list__[idx]
        if cached is not None:
            pyg_data, complexes = cached
            return pyg_data, complexes

        # 获取目标
        targets = self.data['labels']
        target = targets[idx] if not torch.is_tensor(targets) else targets[idx:idx + 1]

        pyg_data = self.pyg_graphs[idx]
        complexes = []

        # 一个 ring size 只加载一次，之后复用
        if not hasattr(self, '__complex_cache__'):
            self.__complex_cache__ = {}

        for r in self._max_ring_sizes:
            suffix = f"{r if r is not None else 'gudhi'}"
            if suffix not in self.__complex_cache__:
                complex_file = os.path.join(self.processed_dir, f"{self.name}_complex_list_{suffix}.pt")
                complex_data, complex_slices = torch.load(complex_file)
                self.__complex_cache__[suffix] = (complex_data, complex_slices)

            complex_data, complex_slices = self.__complex_cache__[suffix]

            retrieved = [
                self._get_cochain_from(complex_data, complex_slices, dim, idx)
                for dim in range(0, self.max_dim + 1)
            ]
            cochains = [r[0] for r in retrieved if not r[1]]
            complexes.append(Complex(*cochains, y=target))

        # 缓存的是引用，不是深拷贝，极限省内存
        self.__data_list__[idx] = (pyg_data, complexes)

        return pyg_data, complexes

    def get_tune_idx_split(self):
        raise NotImplementedError('Not implemented yet')


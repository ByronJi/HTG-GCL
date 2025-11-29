import torch
import os.path as osp
import os

from data.utils import convert_graph_dataset_with_rings, convert_graph_dataset_with_gudhi
from data.datasets import InMemoryComplexDataset
from ogb.graphproppred import PygGraphPropPredDataset

from data.complex import Complex


class OGBDataset(InMemoryComplexDataset):
    """This is OGB graph-property prediction. This are graph-wise classification tasks."""

    def __init__(self, root, name, max_ring_sizes, use_edge_features=False, transform=None,
                 pre_transform=None, pre_filter=None, init_method='sum', 
                 include_down_adj=False, simple=False, n_jobs=2):
        self.name = name

        if isinstance(max_ring_sizes, int):
            max_ring_sizes = [max_ring_sizes]
        self._max_ring_sizes = max_ring_sizes or [None]
        self._cellular = any(r is not None for r in self._max_ring_sizes)
        cellular = self._cellular

        self._use_edge_features = use_edge_features
        self._simple = simple
        self._n_jobs = n_jobs
        super(OGBDataset, self).__init__(root, transform, pre_transform, pre_filter,
                                         max_dim=2, init_method=init_method, 
                                         include_down_adj=include_down_adj, cellular=cellular)
        self.data, self.slices, idx, self.num_tasks = self.load_dataset()
        self.train_ids = idx['train']
        self.val_ids = idx['valid']
        self.test_ids = idx['test']

        self.pyg_graphs = PygGraphPropPredDataset(self.name, self.raw_dir)
        self.task_type = self.pyg_graphs.task_type
        
    @property
    def raw_file_names(self):
        name = self.name.replace('-', '_')  # Replacing is to follow OGB folder naming convention
        # The processed graph files are our raw files.
        return [f'{name}/processed/geometric_data_processed.pt']

    # @property
    # def processed_file_names(self):
    #     return [f'{self.name}_complex.pt', f'{self.name}_idx.pt', f'{self.name}_tasks.pt']
    @property
    def processed_file_names(self):
        complex_files = [
            '{}_complex_list_{}.pt'.format(self.name, r if r is not None else 'gudhi')
            for r in self._max_ring_sizes
        ]
        return complex_files + [f'{self.name}_idx.pt', f'{self.name}_tasks.pt']
    
    @property
    def processed_dir(self):
        """Overwrite to change name based on edge and simple feats"""
        directory = super(OGBDataset, self).processed_dir
        suffix1 = f"_{self._max_ring_sizes}rings" if self._cellular else ""
        suffix2 = "-E" if self._use_edge_features else ""
        suffix3 = "-S" if self._simple else ""
        return directory + suffix1 + suffix2 + suffix3

    def download(self):
        # Instantiating this will download and process the graph dataset.
        dataset = PygGraphPropPredDataset(self.name, self.raw_dir)

    def load_dataset(self):
        """Load the dataset from here and process it if it doesn't exist"""
        print("Loading dataset from disk...")
        data, slices = torch.load(self.processed_paths[0]) # 主要是为了保证一些check函数
        idx = torch.load(self.processed_paths[-2])
        tasks = torch.load(self.processed_paths[-1])
        return data, slices, idx, tasks

    def process(self):
        
        # At this stage, the graph dataset is already downloaded and processed
        dataset = PygGraphPropPredDataset(self.name, self.raw_dir)
        split_idx = dataset.get_idx_split()
        if self._simple:  # Only retain the top two node/edge features
            print('Using simple features')
            dataset.data.x = dataset.data.x[:,:2]
            dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

        # NB: the init method would basically have no effect if 
        # we use edge features and do not initialize rings. 
        print(f"Converting the {self.name} dataset to a cell complex...")
        for r in self._max_ring_sizes:
            if r is not None:
                print(f"Converting with max_ring_size={r}")
                complexes, _, _ = convert_graph_dataset_with_rings(
                    dataset, max_ring_size=r,
                    include_down_adj=self.include_down_adj,
                    init_method=self._init_method,
                    init_edges=self._use_edge_features,
                    init_rings=False,
                    n_jobs=self._n_jobs)

            else:
                print("Converting with gudhi...")
                complexes, _, _ = convert_graph_dataset_with_gudhi(
                    dataset, expansion_dim=self.max_dim,
                    include_down_adj=self.include_down_adj,
                    init_method=self._init_method)

            save_path = os.path.join(self.processed_dir,
                                     f"{self.name}_complex_list_{r if r is not None else 'gudhi'}.pt")
            torch.save(self.collate(complexes, self.max_dim), save_path)

        
        print(f'Saving idx in {self.processed_paths[-2]}...')
        torch.save(split_idx, self.processed_paths[-2])

        print(f'Saving num_tasks in {self.processed_paths[-1]}...')
        torch.save(dataset.num_tasks, self.processed_paths[-1])

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

        pyg_data = self.pyg_graphs.get(idx)
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

def load_ogb_graph_dataset(root, name):
    raw_dir = osp.join(root, 'raw')
    dataset = PygGraphPropPredDataset(name, raw_dir)
    idx = dataset.get_idx_split()

    return dataset, idx['train'], idx['valid'], idx['test']

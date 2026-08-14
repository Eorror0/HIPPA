import torch_geometric
from typing import List

import torch
from torch import Tensor
from torch_sparse import SparseTensor, cat
import torch_geometric
from torch_geometric.data import Data

class BatchWSI(torch_geometric.data.Batch):
    def __init__(self):
        super(BatchWSI, self).__init__()
        pass

    @classmethod
    def from_data_list(cls, data_list, follow_batch=[], exclude_keys=[], update_cat_dims={}):

            keys = list(set(data_list[0].keys) - set(exclude_keys))

            assert 'batch' not in keys and 'ptr' not in keys

            batch = cls()

            for key in data_list[0].__dict__.keys():
                if key[:2] != '__' and key[-2:] != '__':
                    batch[key] = None

            batch.__num_graphs__ = len(data_list)
            batch.__data_class__ = data_list[0].__class__
            for key in keys + ['batch']:
                batch[key] = []
            batch['ptr'] = [0]
            cat_dims = {}
            device = None
            slices = {key: [0] for key in keys}
            cumsum = {key: [0] for key in keys}
            num_nodes_list = []
            for i, data in enumerate(data_list):
                for key in keys:
                    item = data[key]

                    cum = cumsum[key][-1]
                    if isinstance(item, Tensor) and item.dtype != torch.bool:
                        if not isinstance(cum, int) or cum != 0:
                            item = item + cum
                    elif isinstance(item, SparseTensor):
                        value = item.storage.value()
                        if value is not None and value.dtype != torch.bool:
                            if not isinstance(cum, int) or cum != 0:
                                value = value + cum
                            item = item.set_value(value, layout='coo')
                    elif isinstance(item, (int, float)):
                        item = item + cum

                    size = 1

                    if key in update_cat_dims.keys():
                        cat_dim = update_cat_dims[key]
                    else:
                        cat_dim = data.__cat_dim__(key, data[key])

                        if isinstance(item, Tensor) and item.dim() == 0:
                            cat_dim = None

                    cat_dims[key] = cat_dim

                    if isinstance(item, Tensor) and cat_dim is None:
                        cat_dim = 0
                        item = item.unsqueeze(0)
                        device = item.device
                    elif isinstance(item, Tensor):
                        size = item.size(cat_dim)
                        device = item.device
                    elif isinstance(item, SparseTensor):
                        size = torch.tensor(item.sizes())[torch.tensor(cat_dim)]
                        device = item.device()

                    batch[key].append(item)

                    slices[key].append(size + slices[key][-1])
                    inc = data.__inc__(key, item)
                    if isinstance(inc, (tuple, list)):
                        inc = torch.tensor(inc)
                    cumsum[key].append(inc + cumsum[key][-1])

                    if key in follow_batch:
                        if isinstance(size, Tensor):
                            for j, size in enumerate(size.tolist()):
                                tmp = f'{key}_{j}_batch'
                                batch[tmp] = [] if i == 0 else batch[tmp]
                                batch[tmp].append(
                                    torch.full((size, ), i, dtype=torch.long,
                                            device=device))
                        else:
                            tmp = f'{key}_batch'
                            batch[tmp] = [] if i == 0 else batch[tmp]
                            batch[tmp].append(
                                torch.full((size, ), i, dtype=torch.long,
                                        device=device))

                if hasattr(data, '__num_nodes__'):
                    num_nodes_list.append(data.__num_nodes__)
                else:
                    num_nodes_list.append(None)

                num_nodes = data.num_nodes
                if num_nodes is not None:
                    item = torch.full((num_nodes, ), i, dtype=torch.long,
                                    device=device)
                    batch.batch.append(item)
                    batch.ptr.append(batch.ptr[-1] + num_nodes)

            batch.batch = None if len(batch.batch) == 0 else batch.batch
            batch.ptr = None if len(batch.ptr) == 1 else batch.ptr
            batch.__slices__ = slices
            batch.__cumsum__ = cumsum
            batch.__cat_dims__ = cat_dims
            batch.__num_nodes_list__ = num_nodes_list

            ref_data = data_list[0]
            for key in batch.keys:
                items = batch[key]
                item = items[0]

                if key in update_cat_dims.keys():
                    cat_dim = update_cat_dims[key]
                else:
                    cat_dim = ref_data.__cat_dim__(key, item)
                    cat_dim = 0 if cat_dim is None else cat_dim

                if isinstance(item, Tensor):
                    batch[key] = torch.cat(items, cat_dim)
                elif isinstance(item, SparseTensor):
                    batch[key] = cat(items, cat_dim)
                elif isinstance(item, (int, float)):
                    batch[key] = torch.tensor(items)

            if torch_geometric.is_debug_enabled():
                batch.debug()

            return batch.contiguous()

"""
Code based on https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py

Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
Copyright (c) 2021 The CWN Project Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d as BN, ReLU
from torch_geometric.nn import GINConv, JumpingKnowledge, GINEConv, global_add_pool
from mp.nn import get_nonlinearity, get_pooling_fn

# modified version, delete num_classes, no need in unsupervised setting
class GIN0(torch.nn.Module):
    def __init__(self, num_features, num_layers, hidden, readout='sum',
                 dropout_rate=0.5, nonlinearity='relu'):
        super(GIN0, self).__init__()
        self.pooling_fn = get_pooling_fn(readout) 
        self.nonlinearity = nonlinearity
        self.dropout_rate = dropout_rate
        conv_nonlinearity = get_nonlinearity(nonlinearity, return_module=True)
        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden),
                BN(hidden),
                conv_nonlinearity(),
                Linear(hidden, hidden),
                BN(hidden),
                conv_nonlinearity(),
            ), train_eps=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        BN(hidden),
                        conv_nonlinearity(),
                        Linear(hidden, hidden),
                        BN(hidden),
                        conv_nonlinearity(),
                    ), train_eps=False))
        self.lin1 = Linear(hidden, hidden*2)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, data):
        # model_nonlinearity = get_nonlinearity(self.nonlinearity, return_module=False)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = self.pooling_fn(x, batch)
        # x = model_nonlinearity(self.lin1(x))
        x = self.lin1(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x

    def __repr__(self):
        return self.__class__.__name__


class GIN0WithJK(torch.nn.Module):
    def __init__(self, num_features, num_layers, hidden, num_classes, mode='cat', readout='sum',
                 dropout_rate=0.5, nonlinearity='relu'):
        super(GIN0WithJK, self).__init__()
        self.pooling_fn = get_pooling_fn(readout) 
        self.dropout_rate = dropout_rate
        self.nonlinearity = nonlinearity
        conv_nonlinearity = get_nonlinearity(nonlinearity, return_module=True)
        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden),
                BN(hidden),
                conv_nonlinearity(),
                Linear(hidden, hidden),
                BN(hidden),
                conv_nonlinearity(),
            ), train_eps=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        BN(hidden),
                        conv_nonlinearity(),
                        Linear(hidden, hidden),
                        BN(hidden),
                        conv_nonlinearity(),
                    ), train_eps=False))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        model_nonlinearity = get_nonlinearity(self.nonlinearity, return_module=False)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        x = self.jump(xs)
        x = self.pooling_fn(x, batch)
        x = model_nonlinearity(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__


class GIN(torch.nn.Module):
    def __init__(self, num_features, num_layers, hidden, num_classes, readout='sum',
                 dropout_rate=0.5, nonlinearity='relu'):
        super(GIN, self).__init__()
        self.pooling_fn = get_pooling_fn(readout) 
        self.dropout_rate = dropout_rate
        self.nonlinearity = nonlinearity
        conv_nonlinearity = get_nonlinearity(nonlinearity, return_module=True)
        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden),
                BN(hidden),
                conv_nonlinearity(),
                Linear(hidden, hidden),
                BN(hidden),
                conv_nonlinearity(),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        BN(hidden),
                        conv_nonlinearity(),
                        Linear(hidden, hidden),
                        BN(hidden),
                        conv_nonlinearity(),
                    ), train_eps=True))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        model_nonlinearity = get_nonlinearity(self.nonlinearity, return_module=False)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = self.pooling_fn(x, batch)
        x = model_nonlinearity(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__


class GINWithJK(torch.nn.Module):
    def __init__(self, num_features, num_layers, hidden, num_classes, mode='cat', readout='sum',
                 dropout_rate=0.5, nonlinearity='relu'):
        super(GINWithJK, self).__init__()
        self.pooling_fn = get_pooling_fn(readout) 
        self.dropout_rate = dropout_rate
        self.nonlinearity = nonlinearity
        conv_nonlinearity = get_nonlinearity(nonlinearity, return_module=True)
        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden),
                BN(hidden),
                conv_nonlinearity(),
                Linear(hidden, hidden),
                BN(hidden),
                conv_nonlinearity(),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        BN(hidden),
                        conv_nonlinearity(),
                        Linear(hidden, hidden),
                        BN(hidden),
                        conv_nonlinearity(),
                    ), train_eps=True))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        model_nonlinearity = get_nonlinearity(self.nonlinearity, return_module=False)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        x = self.jump(xs)
        x = self.pooling_fn(x, batch)
        x = model_nonlinearity(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__


class ZINCEncoder(torch.nn.Module):
	def __init__(self, num_atom_type, num_bond_type, emb_dim=100, num_gc_layers=5, drop_ratio=0.0, pooling_type="standard"):
		super(ZINCEncoder, self).__init__()
		self.pooling_type = pooling_type
		self.emb_dim = emb_dim
		self.num_gc_layers = num_gc_layers
		self.drop_ratio = drop_ratio

		self.out_node_dim = self.emb_dim
		if self.pooling_type == "standard":
			self.out_graph_dim = self.emb_dim
		elif self.pooling_type == "layerwise":
			self.out_graph_dim = self.emb_dim * self.num_gc_layers
		else:
			raise NotImplementedError

		self.atom_embedding = torch.nn.Embedding(num_atom_type, emb_dim)
		self.bond_embedding = torch.nn.Embedding(num_bond_type, emb_dim)
		self.convs = torch.nn.ModuleList()
		self.bns = torch.nn.ModuleList()
		for i in range(num_gc_layers):
			nn = Sequential(Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim), ReLU(),
			                Linear(2 * emb_dim, emb_dim))
			conv = GINEConv(nn)
			bn = torch.nn.BatchNorm1d(emb_dim)
			self.convs.append(conv)
			self.bns.append(bn)

		self.init_emb()

	def init_emb(self):
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, batch, x, edge_index, edge_attr, edge_weight=None):
		x = self.atom_embedding(x).squeeze(1)
		edge_attr = self.bond_embedding(edge_attr)

		# compute node embeddings using GNN
		xs = []
		for i in range(self.num_gc_layers):
			x = self.convs[i](x, edge_index, edge_attr, edge_weight)
			x = self.bns[i](x)
			if i == self.num_gc_layers - 1:
				# remove relu for the last layer
				x = F.dropout(x, self.drop_ratio, training=self.training)
			else:
				x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
			xs.append(x)

		# compute graph embedding using pooling
		if self.pooling_type == "standard":
			xpool = global_add_pool(x, batch)
			return xpool, x

		# elif self.pooling_type == "layerwise":
		# 	xpool = [global_add_pool(x, batch) for x in xs]
		# 	xpool = torch.cat(xpool, 1)
		# 	if self.is_infograph:
		# 		return xpool, torch.cat(xs, 1)
		# 	else:
		# 		return xpool, x
		else:
			raise NotImplementedError

	# def get_embeddings(self, loader, device, is_rand_label=False):
	# 	ret = []
	# 	y = []
	# 	with torch.no_grad():
	# 		for data in loader:
	# 			data = data.to(device)
	# 			batch, x, edge_index, edge_attr = data.batch, data.x, data.edge_index, data.edge_attr
	# 			edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None
    #
	# 			if x is None:
	# 				x = torch.ones((batch.shape[0], 1)).to(device)
	# 			x, _ = self.forward(batch, x, edge_index, edge_attr, edge_weight)
    #
	# 			ret.append(x.cpu().numpy())
	# 			if is_rand_label:
	# 				y.append(data.rand_label.cpu().numpy())
	# 			else:
	# 				y.append(data.y.cpu().numpy())
	# 	ret = np.concatenate(ret, 0)
	# 	y = np.concatenate(y, 0)
	# 	return ret, y
import torch
import torch.nn.functional as F

from torch.nn import Linear, Embedding, Sequential, BatchNorm1d as BN
from torch_geometric.nn import JumpingKnowledge, GINEConv
from mp.layers import InitReduceConv, EmbedVEWithReduce, OGBEmbedVEWithReduce, SparseCINConv, CINppConv
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from data.complex import ComplexBatch
from mp.nn import pool_complex, get_pooling_fn, get_nonlinearity, get_graph_norm

from torch.nn import ReLU
from torch_geometric.nn import global_add_pool
import numpy as np

class EmbedSparseCIN(torch.nn.Module):
	"""
	A cellular version of GIN with some tailoring to nimbly work on molecules from the ZINC database.

	This model is based on
	https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
	"""

	def __init__(self, atom_types, bond_types, out_size, num_layers, hidden,
				 dropout_rate: float = 0.5, max_dim: int = 2, jump_mode=None, nonlinearity='relu',
				 readout='sum', train_eps=False, final_hidden_multiplier: int = 2,
				 readout_dims=(0, 1, 2), final_readout='sum', apply_dropout_before='lin2',
				 init_reduce='sum', embed_edge=False, embed_dim=None, use_coboundaries=False,
				 graph_norm='bn'):
		super(EmbedSparseCIN, self).__init__()

		self.max_dim = max_dim
		if readout_dims is not None:
			self.readout_dims = tuple([dim for dim in readout_dims if dim <= max_dim])
		else:
			self.readout_dims = list(range(max_dim+1))

		if embed_dim is None:
			embed_dim = hidden
		self.v_embed_init = Embedding(atom_types, embed_dim)

		self.e_embed_init = None
		if embed_edge:
			self.e_embed_init = Embedding(bond_types, embed_dim)
		self.reduce_init = InitReduceConv(reduce=init_reduce)
		self.init_conv = EmbedVEWithReduce(self.v_embed_init, self.e_embed_init, self.reduce_init)

		self.final_readout = final_readout
		self.dropout_rate = dropout_rate
		self.apply_dropout_before = apply_dropout_before
		self.jump_mode = jump_mode
		self.convs = torch.nn.ModuleList()
		self.nonlinearity = nonlinearity
		self.readout = readout
		self.graph_norm = get_graph_norm(graph_norm)
		act_module = get_nonlinearity(nonlinearity, return_module=True)
		for i in range(num_layers):
			layer_dim = embed_dim if i == 0 else hidden
			self.convs.append(
				SparseCINConv(up_msg_size=layer_dim, down_msg_size=layer_dim,
					boundary_msg_size=layer_dim, passed_msg_boundaries_nn=None,
					passed_msg_up_nn=None, passed_update_up_nn=None,
					passed_update_boundaries_nn=None, train_eps=train_eps, max_dim=self.max_dim,
					hidden=hidden, act_module=act_module, layer_dim=layer_dim,
					graph_norm=self.graph_norm, use_coboundaries=use_coboundaries))
		self.jump = JumpingKnowledge(jump_mode) if jump_mode is not None else None
		self.lin1s = torch.nn.ModuleList()
		for _ in range(max_dim + 1):
			if jump_mode == 'cat':
				# These layers don't use a bias. Then, in case a level is not present the output
				# is just zero and it is not given by the biases.
				self.lin1s.append(Linear(num_layers * hidden, final_hidden_multiplier * hidden,
					bias=False))
			else:
				self.lin1s.append(Linear(hidden, final_hidden_multiplier * hidden))
		# self.lin2 = Linear(final_hidden_multiplier * hidden, out_size)

	def reset_parameters(self):
		for conv in self.convs:
			conv.reset_parameters()
		if self.jump_mode is not None:
			self.jump.reset_parameters()
		self.init_conv.reset_parameters()
		self.lin1s.reset_parameters()
		# self.lin2.reset_parameters()

	def jump_complex(self, jump_xs):
		# Perform JumpingKnowledge at each level of the complex
		xs = []
		for jumpx in jump_xs:
			xs += [self.jump(jumpx)]
		return xs

	def forward(self, data: ComplexBatch, include_partial=False):
		act = get_nonlinearity(self.nonlinearity, return_module=False)
		xs, jump_xs = None, None
		res = {}

		# Check input node/edge features are scalars.
		assert data.cochains[0].x.size(-1) == 1
		if 1 in data.cochains and data.cochains[1].x is not None:
			assert data.cochains[1].x.size(-1) == 1

		# Embed and populate higher-levels
		params = data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=False)
		xs = list(self.init_conv(*params))

		# Apply dropout on the input features like all models do on ZINC.
		for i, x in enumerate(xs):
			xs[i] = F.dropout(xs[i], p=self.dropout_rate, training=self.training)

		data.set_xs(xs)

		for c, conv in enumerate(self.convs):
			params = data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=False)
			start_to_process = 0
			xs = conv(*params, start_to_process=start_to_process)
			data.set_xs(xs)

			if include_partial:
				for k in range(len(xs)):
					res[f"layer{c}_{k}"] = xs[k]

			if self.jump_mode is not None:
				if jump_xs is None:
					jump_xs = [[] for _ in xs]
				for i, x in enumerate(xs):
					jump_xs[i] += [x]

		if self.jump_mode is not None:
			xs = self.jump_complex(jump_xs)

		xs = pool_complex(xs, data, self.max_dim, self.readout)
		# Select the dimensions we want at the end.
		xs = [xs[i] for i in self.readout_dims]

		if include_partial:
			for k in range(len(xs)):
				res[f"pool_{k}"] = xs[k]

		new_xs = []
		for i, x in enumerate(xs):
			if self.apply_dropout_before == 'lin1':
				x = F.dropout(x, p=self.dropout_rate, training=self.training)
			new_xs.append(act(self.lin1s[self.readout_dims[i]](x)))

		x = torch.stack(new_xs, dim=0)

		if self.apply_dropout_before == 'final_readout':
			x = F.dropout(x, p=self.dropout_rate, training=self.training)
		if self.final_readout == 'mean':
			x = x.mean(0)
		elif self.final_readout == 'sum':
			x = x.sum(0)
		else:
			raise NotImplementedError
		if self.apply_dropout_before not in ['lin1', 'final_readout']:
			x = F.dropout(x, p=self.dropout_rate, training=self.training)

		# x = self.lin2(x)

		if include_partial:
			res['out'] = x
			return x, res
		return x

	def __repr__(self):
		return self.__class__.__name__


class EmbedCINpp(EmbedSparseCIN):
	"""
	Inherit from EmbedSparseCIN and add messages from lower adj cells
	"""

	def __init__(self, atom_types, bond_types, out_size, num_layers, hidden,
				dropout_rate: float = 0.5, max_dim: int = 2, jump_mode=None,
				nonlinearity='relu', readout='sum', train_eps=False,
				final_hidden_multiplier: int = 2, readout_dims=(0, 1, 2),
				final_readout='sum', apply_dropout_before='lin2', init_reduce='sum',
				embed_edge=False, embed_dim=None, use_coboundaries=False, graph_norm='bn'):
		super(EmbedCINpp, self).__init__(atom_types, bond_types, out_size, num_layers,
										 hidden, dropout_rate, max_dim, jump_mode,
										 nonlinearity, readout, train_eps,
										 final_hidden_multiplier, readout_dims,
										 final_readout, apply_dropout_before,
										 init_reduce, embed_edge, embed_dim,
										 use_coboundaries, graph_norm)
		self.convs = torch.nn.ModuleList() #reset convs to use CINppConv instead of SparseCINConv
		act_module = get_nonlinearity(nonlinearity, return_module=True)

		if embed_dim is None:
			embed_dim = hidden

		for i in range(num_layers):
			layer_dim = embed_dim if i == 0 else hidden
			self.convs.append(
				CINppConv(up_msg_size=layer_dim, down_msg_size=layer_dim,
					boundary_msg_size=layer_dim, passed_msg_boundaries_nn=None,
					passed_msg_up_nn=None, passed_msg_down_nn=None, passed_update_up_nn=None,
					passed_update_down_nn=None, passed_update_boundaries_nn=None, train_eps=train_eps,
					max_dim=self.max_dim, hidden=hidden, act_module=act_module, layer_dim=layer_dim,
					graph_norm=self.graph_norm, use_coboundaries=use_coboundaries))

class OGBEmbedSparseCIN(torch.nn.Module):
	"""
	A cellular version of GIN with some tailoring to nimbly work on molecules from the ogbg-mol* dataset.
	It uses OGB atom and bond encoders.

	This model is based on
	https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
	"""

	def __init__(self, out_size, num_layers, hidden, dropout_rate: float = 0.5,
				 indropout_rate: float = 0.0, max_dim: int = 2, jump_mode=None,
				 nonlinearity='relu', readout='sum', train_eps=False, final_hidden_multiplier: int = 1,
				 readout_dims=(0, 1, 2), final_readout='sum', apply_dropout_before='lin2',
				 init_reduce='sum', embed_edge=False, embed_dim=None, use_coboundaries=False,
				 graph_norm='bn'):
		super(OGBEmbedSparseCIN, self).__init__()

		self.max_dim = max_dim
		if readout_dims is not None:
			self.readout_dims = tuple([dim for dim in readout_dims if dim <= max_dim])
		else:
			self.readout_dims = list(range(max_dim+1))

		if embed_dim is None:
			embed_dim = hidden
		self.v_embed_init = AtomEncoder(embed_dim)

		self.e_embed_init = None
		if embed_edge:
			self.e_embed_init = BondEncoder(embed_dim)
		self.reduce_init = InitReduceConv(reduce=init_reduce)
		self.init_conv = OGBEmbedVEWithReduce(self.v_embed_init, self.e_embed_init, self.reduce_init)

		self.final_readout = final_readout
		self.dropout_rate = dropout_rate
		self.in_dropout_rate = indropout_rate
		self.apply_dropout_before = apply_dropout_before
		self.jump_mode = jump_mode
		self.convs = torch.nn.ModuleList()
		self.nonlinearity = nonlinearity
		self.readout = readout
		act_module = get_nonlinearity(nonlinearity, return_module=True)
		self.graph_norm = get_graph_norm(graph_norm)
		for i in range(num_layers):
			layer_dim = embed_dim if i == 0 else hidden
			self.convs.append(
				SparseCINConv(up_msg_size=layer_dim, down_msg_size=layer_dim,
					boundary_msg_size=layer_dim, passed_msg_boundaries_nn=None,
					passed_msg_up_nn=None, passed_update_up_nn=None,
					passed_update_boundaries_nn=None, train_eps=train_eps, max_dim=self.max_dim,
					hidden=hidden, act_module=act_module, layer_dim=layer_dim,
					graph_norm=self.graph_norm, use_coboundaries=use_coboundaries))
		self.jump = JumpingKnowledge(jump_mode) if jump_mode is not None else None
		self.lin1s = torch.nn.ModuleList()
		for _ in range(max_dim + 1):
			if jump_mode == 'cat':
				# These layers don't use a bias. Then, in case a level is not present the output
				# is just zero and it is not given by the biases.
				self.lin1s.append(Linear(num_layers * hidden, final_hidden_multiplier * hidden,
					bias=False))
			else:
				self.lin1s.append(Linear(hidden, final_hidden_multiplier * hidden))
		# self.lin2 = Linear(final_hidden_multiplier * hidden, out_size)
		# self.lin2 = Linear(final_hidden_multiplier * hidden, final_hidden_multiplier * hidden)
		self.output_dim = final_hidden_multiplier * hidden

	def reset_parameters(self):
		for conv in self.convs:
			conv.reset_parameters()
		if self.jump_mode is not None:
			self.jump.reset_parameters()
		self.init_conv.reset_parameters()
		self.lin1s.reset_parameters()
		# self.lin2.reset_parameters()

	def jump_complex(self, jump_xs):
		# Perform JumpingKnowledge at each level of the complex
		xs = []
		for jumpx in jump_xs:
			xs += [self.jump(jumpx)]
		return xs

	def forward(self, data: ComplexBatch, include_partial=False):
		act = get_nonlinearity(self.nonlinearity, return_module=False)
		xs, jump_xs = None, None
		res = {}

		# Embed and populate higher-levels
		params = data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=False)
		xs = list(self.init_conv(*params))

		# Apply dropout on the input features
		for i, x in enumerate(xs):
			xs[i] = F.dropout(xs[i], p=self.in_dropout_rate, training=self.training)

		data.set_xs(xs)

		for c, conv in enumerate(self.convs):
			params = data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=False)
			start_to_process = 0
			xs = conv(*params, start_to_process=start_to_process)
			# Apply dropout on the output of the conv layer
			for i, x in enumerate(xs):
				xs[i] = F.dropout(xs[i], p=self.dropout_rate, training=self.training)
			data.set_xs(xs)

			if include_partial:
				for k in range(len(xs)):
					res[f"layer{c}_{k}"] = xs[k]

			if self.jump_mode is not None:
				if jump_xs is None:
					jump_xs = [[] for _ in xs]
				for i, x in enumerate(xs):
					jump_xs[i] += [x]

		if self.jump_mode is not None:
			xs = self.jump_complex(jump_xs)

		xs = pool_complex(xs, data, self.max_dim, self.readout)
		# Select the dimensions we want at the end.
		xs = [xs[i] for i in self.readout_dims]

		if include_partial:
			for k in range(len(xs)):
				res[f"pool_{k}"] = xs[k]

		new_xs = []
		for i, x in enumerate(xs):
			if self.apply_dropout_before == 'lin1':
				x = F.dropout(x, p=self.dropout_rate, training=self.training)
			new_xs.append(act(self.lin1s[self.readout_dims[i]](x)))

		x = torch.stack(new_xs, dim=0)

		if self.apply_dropout_before == 'final_readout':
			x = F.dropout(x, p=self.dropout_rate, training=self.training)
		if self.final_readout == 'mean':
			x = x.mean(0)
		elif self.final_readout == 'sum':
			x = x.sum(0)
		else:
			raise NotImplementedError
		if self.apply_dropout_before not in ['lin1', 'final_readout']:
			x = F.dropout(x, p=self.dropout_rate, training=self.training)

		# x = self.lin2(x)

		if include_partial:
			res['out'] = x
			return x, res
		return x

	def __repr__(self):
		return self.__class__.__name__

class OGBEmbedCINpp(OGBEmbedSparseCIN):
	"""
	Inherit from EmbedSparseCIN and add messages from lower adj cells
	"""
	def __init__(self, out_size, num_layers, hidden, dropout_rate: float = 0.5,
				 indropout_rate: float = 0, max_dim: int = 2, jump_mode=None,
				 nonlinearity='relu', readout='sum', train_eps=False,
				 final_hidden_multiplier: int = 2, readout_dims=(0, 1, 2),
				 final_readout='sum', apply_dropout_before='lin2', init_reduce='sum',
				 embed_edge=False, embed_dim=None, use_coboundaries=False, graph_norm='bn'):
		super().__init__(out_size, num_layers, hidden, dropout_rate, indropout_rate,
						 max_dim, jump_mode, nonlinearity, readout, train_eps,
						 final_hidden_multiplier, readout_dims, final_readout,
						 apply_dropout_before, init_reduce, embed_edge, embed_dim,
						 use_coboundaries, graph_norm)
		self.convs = torch.nn.ModuleList() #reset convs to use CINppConv instead of SparseCINConv
		act_module = get_nonlinearity(nonlinearity, return_module=True)

		if embed_dim is None:
			embed_dim = hidden

		for i in range(num_layers):
			layer_dim = embed_dim if i == 0 else hidden
			self.convs.append(
				CINppConv(up_msg_size=layer_dim, down_msg_size=layer_dim,
					boundary_msg_size=layer_dim, passed_msg_boundaries_nn=None,
					passed_msg_up_nn=None, passed_msg_down_nn=None, passed_update_up_nn=None,
					passed_update_down_nn=None, passed_update_boundaries_nn=None, train_eps=train_eps,
					max_dim=self.max_dim, hidden=hidden, act_module=act_module, layer_dim=layer_dim,
					graph_norm=self.graph_norm, use_coboundaries=use_coboundaries))

class EmbedSparseCINNoRings(torch.nn.Module):
	"""
	CIN model on cell complexes up to dimension 1 (edges). It does not make use of rings.

	This model is based on
	https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
	"""

	def __init__(self, atom_types, bond_types, out_size, num_layers, hidden,
				 dropout_rate: float = 0.5, nonlinearity='relu',
				 readout='sum', train_eps=False, final_hidden_multiplier: int = 2,
				 final_readout='sum', apply_dropout_before='lin2',
				 init_reduce='sum', embed_edge=False, embed_dim=None, use_coboundaries=False,
				 graph_norm='bn'):
		super(EmbedSparseCINNoRings, self).__init__()

		self.max_dim = 1
		self.readout_dims = [0, 1]

		if embed_dim is None:
			embed_dim = hidden
		self.v_embed_init = Embedding(atom_types, embed_dim)

		self.e_embed_init = None
		if embed_edge:
			self.e_embed_init = Embedding(bond_types, embed_dim)
		self.reduce_init = InitReduceConv(reduce=init_reduce)
		self.init_conv = EmbedVEWithReduce(self.v_embed_init, self.e_embed_init, self.reduce_init)

		self.final_readout = final_readout
		self.dropout_rate = dropout_rate
		self.apply_dropout_before = apply_dropout_before
		self.convs = torch.nn.ModuleList()
		self.nonlinearity = nonlinearity
		self.readout = readout
		self.graph_norm = get_graph_norm(graph_norm)

		act_module = get_nonlinearity(nonlinearity, return_module=True)
		for i in range(num_layers):
			layer_dim = embed_dim if i == 0 else hidden
			self.convs.append(
				SparseCINConv(up_msg_size=layer_dim, down_msg_size=layer_dim,
							  boundary_msg_size=layer_dim, passed_msg_boundaries_nn=None,
							  passed_msg_up_nn=None, passed_update_up_nn=None,
							  passed_update_boundaries_nn=None, train_eps=train_eps, max_dim=self.max_dim,
							  hidden=hidden, act_module=act_module, layer_dim=layer_dim,
							  graph_norm=self.graph_norm, use_coboundaries=use_coboundaries))
		self.lin1s = torch.nn.ModuleList()
		for _ in range(self.max_dim + 1):
			self.lin1s.append(Linear(hidden, final_hidden_multiplier * hidden))
		self.lin2 = Linear(final_hidden_multiplier * hidden, out_size)

	def reset_parameters(self):
		for conv in self.convs:
			conv.reset_parameters()
		self.init_conv.reset_parameters()
		self.lin1s.reset_parameters()
		self.lin2.reset_parameters()

	def forward(self, data: ComplexBatch):
		act = get_nonlinearity(self.nonlinearity, return_module=False)

		# Check input node/edge features are scalars.
		assert data.cochains[0].x.size(-1) == 1
		if 1 in data.cochains and data.cochains[1].x is not None:
			assert data.cochains[1].x.size(-1) == 1

		# Extract node and edge params
		params = data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=False)
		# Make the upper index of the edges None to ignore the rings. Even though max_dim = 1
		# our current code does extract upper adjacencies for edges if rings are present.
		if len(params) > 1:
			params[1].up_index = None
		# Embed the node and edge features
		xs = list(self.init_conv(*params))

		# Apply dropout on the input features
		for i, x in enumerate(xs):
			xs[i] = F.dropout(xs[i], p=self.dropout_rate, training=self.training)

		data.set_xs(xs)

		for c, conv in enumerate(self.convs):
			params = data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=False)
			if len(params) > 1:
				params[1].up_index = None

			xs = conv(*params)
			data.set_xs(xs)

		xs = pool_complex(xs, data, self.max_dim, self.readout)
		# Select the dimensions we want at the end.
		xs = [xs[i] for i in self.readout_dims]

		new_xs = []
		for i, x in enumerate(xs):
			if self.apply_dropout_before == 'lin1':
				x = F.dropout(x, p=self.dropout_rate, training=self.training)
			new_xs.append(act(self.lin1s[self.readout_dims[i]](x)))

		x = torch.stack(new_xs, dim=0)

		if self.apply_dropout_before == 'final_readout':
			x = F.dropout(x, p=self.dropout_rate, training=self.training)
		if self.final_readout == 'mean':
			x = x.mean(0)
		elif self.final_readout == 'sum':
			x = x.sum(0)
		else:
			raise NotImplementedError
		if self.apply_dropout_before not in ['lin1', 'final_readout']:
			x = F.dropout(x, p=self.dropout_rate, training=self.training)

		x = self.lin2(x)
		return x

	def __repr__(self):
		return self.__class__.__name__


class EmbedGIN(torch.nn.Module):
	"""
	GIN with cell complex inputs to test our pipeline.

	This model is based on
	https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
	"""

	def __init__(self, atom_types, bond_types, out_size, num_layers, hidden,
				 dropout_rate: float = 0.5, nonlinearity='relu',
				 readout='sum', train_eps=False, apply_dropout_before='lin2',
				 init_reduce='sum', embed_edge=False, embed_dim=None):
		super(EmbedGIN, self).__init__()

		self.max_dim = 1

		if embed_dim is None:
			embed_dim = hidden
		self.v_embed_init = Embedding(atom_types, embed_dim)

		self.e_embed_init = None
		if embed_edge:
			self.e_embed_init = Embedding(bond_types, embed_dim)
		self.reduce_init = InitReduceConv(reduce=init_reduce)
		self.init_conv = EmbedVEWithReduce(self.v_embed_init, self.e_embed_init, self.reduce_init)

		self.dropout_rate = dropout_rate
		self.apply_dropout_before = apply_dropout_before
		self.convs = torch.nn.ModuleList()
		self.nonlinearity = nonlinearity
		self.pooling_fn = get_pooling_fn(readout)
		act_module = get_nonlinearity(nonlinearity, return_module=True)
		for i in range(num_layers):
			layer_dim = embed_dim if i == 0 else hidden
			self.convs.append(
					GINEConv(
						# Here we instantiate and pass the MLP performing the `update` function.
						Sequential(
							Linear(layer_dim, hidden),
							BN(hidden),
							act_module(),
							Linear(hidden, hidden),
							BN(hidden),
							act_module(),
						), train_eps=train_eps))
		self.lin1 = Linear(hidden, hidden)
		self.lin2 = Linear(hidden, out_size)

	def reset_parameters(self):
		for conv in self.convs:
			conv.reset_parameters()
		self.init_conv.reset_parameters()
		self.lin1.reset_parameters()
		self.lin2.reset_parameters()

	def forward(self, data: ComplexBatch):
		act = get_nonlinearity(self.nonlinearity, return_module=False)

		# Check input node/edge features are scalars.
		assert data.cochains[0].x.size(-1) == 1
		if 1 in data.cochains and data.cochains[1].x is not None:
			assert data.cochains[1].x.size(-1) == 1

		# Extract node and edge params
		params = data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=False)
		# Embed the node and edge features
		xs = list(self.init_conv(*params))
		# Apply dropout on the input node features
		xs[0] = F.dropout(xs[0], p=self.dropout_rate, training=self.training)
		data.set_xs(xs)

		# We fetch input parameters only at dimension 0 (nodes)
		params = data.get_all_cochain_params(max_dim=0, include_down_features=False)[0]
		x = params.x
		edge_index = params.up_index
		edge_attr = params.kwargs['up_attr']

		# For the edge case when no edges are present.
		if edge_index is None:
			edge_index = torch.LongTensor([[], []])
			edge_attr = torch.FloatTensor([[0]*x.size(-1)])


		for c, conv in enumerate(self.convs):
			x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr)

		# Pool only over nodes
		batch_size = data.cochains[0].batch.max() + 1
		x = self.pooling_fn(x, data.nodes.batch, size=batch_size)

		if self.apply_dropout_before == 'lin1':
			x = F.dropout(x, p=self.dropout_rate, training=self.training)
		x = act(self.lin1(x))

		if self.apply_dropout_before in ['final_readout', 'lin2']:
			x = F.dropout(x, p=self.dropout_rate, training=self.training)

		x = self.lin2(x)
		return x

	def __repr__(self):
		return self.__class__.__name__


class MoleculeEncoder(torch.nn.Module):
	def __init__(self, emb_dim=300, num_gc_layers=5, drop_ratio=0.0, pooling_type="standard"):
		super(MoleculeEncoder, self).__init__()

		self.pooling_type = pooling_type
		self.emb_dim = emb_dim
		self.num_gc_layers = num_gc_layers
		self.drop_ratio = drop_ratio

		self.out_node_dim = self.emb_dim
		if self.pooling_type == "standard":
			self.out_graph_dim = self.emb_dim
		# elif self.pooling_type == "layerwise":
		# 	self.out_graph_dim = self.emb_dim * self.num_gc_layers
		else:
			raise NotImplementedError

		self.convs = torch.nn.ModuleList()
		self.bns = torch.nn.ModuleList()

		self.atom_encoder = AtomEncoder(emb_dim)
		self.bond_encoder = BondEncoder(emb_dim)

		for i in range(num_gc_layers):
			nn = Sequential(Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), ReLU(), Linear(2*emb_dim, emb_dim))
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

	def forward(self, data):
		batch, x, edge_index, edge_attr = data.batch, data.x, data.edge_index, data.edge_attr
		edge_weight = None
		x = self.atom_encoder(x)
		edge_attr = self.bond_encoder(edge_attr)

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
			return xpool

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
	# 			if isinstance(data, list):
	# 				data = data[0].to(device)
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

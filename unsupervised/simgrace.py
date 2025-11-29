import os
import numpy as np
import copy
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import random
import logging
import copy

import sys


from data.complex import ComplexBatch
from data.data_loading import DataLoader, load_dataset
from torch_geometric.data import DataLoader as PyGDataLoader

from data.datasets import TUDataset

from unsupervised.parser import get_parser
from mp.models import SparseCIN
from mp.graph_models import GIN0

from unsupervised.evaluate_embedding import evaluate_embedding
from unsupervised.utils import *

class simclr(nn.Module):
    def __init__(self, dataset, args):
        super(simclr, self).__init__()
        use_coboundaries = args.use_coboundaries.lower() == 'true'
        readout_dims = tuple(sorted(args.readout_dims))
        num_features = dataset.graph_list[0].num_features
        self.encoder = SparseCIN(num_features,  # num_input_features  dataset.num_features_in_dim(0)
                          dataset.num_classes,  # num_classes
                          args.num_layers,  # num_layers
                          args.emb_dim,  # hidden
                          dropout_rate=args.drop_rate,  # dropout rate
                          max_dim=dataset.max_dim,  # max_dim
                          jump_mode=args.jump_mode,  # jump mode
                          nonlinearity=args.nonlinearity,  # nonlinearity
                          readout=args.readout,  # readout
                          final_readout=args.final_readout,  # final readout
                          apply_dropout_before=args.drop_position,  # where to apply dropout
                          use_coboundaries=use_coboundaries,  # whether to use coboundaries in up-msg
                          graph_norm=args.graph_norm,  # normalization layer
                          readout_dims=readout_dims  # readout_dims (0,1,2)
                          )
        self.gin_encoder = GIN0(num_features,                            # num_input_features
                     args.num_layers,                         # num_layers
                     args.emb_dim,                            # hidden
                     dropout_rate=args.drop_rate,             # dropout rate
                     nonlinearity=args.nonlinearity,          # nonlinearity
                     readout=args.readout,                    # readout
        )
        # test projection head
        self.embedding_dim = args.max_dim * args.emb_dim
        self.gin_proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        self.cwn_proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                           nn.Linear(self.embedding_dim, self.embedding_dim))
        self.cwn_proj_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                          nn.Linear(self.embedding_dim, self.embedding_dim)),  # for y1
            nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                          nn.Linear(self.embedding_dim, self.embedding_dim)),  # for y2
            nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                          nn.Linear(self.embedding_dim, self.embedding_dim))  # for y3
        ])


    def forward(self, complexBatch):
        x = self.encoder(complexBatch)
        x = self.cwn_proj_head(x)
        return x

    def forward_gin(self, dataBatch):
        x = self.gin_encoder(dataBatch)
        x = self.gin_proj_head(x)
        return x

    def forward_cwn(self, complexBatch):
        x = self.encoder(complexBatch)
        x = self.cwn_proj_head(x)
        return x

    @staticmethod
    def simclr_loss(x, x_aug):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss


def get_embeddings(model, loader, device):
    ret = []
    y = []
    with torch.no_grad():
        for data in loader:
            data.to(device)

            x = model.forward(data)

            ret.append(x.cpu().numpy())
            y.append(data.y.cpu().numpy())
    ret = np.concatenate(ret, 0)
    y = np.concatenate(y, 0)
    return ret, y

def get_all_embeddings(model, loader, device):
    ret = []
    ret_c1 = []
    ret_c2 = []
    ret_c3 = []
    y = []
    with torch.no_grad():
        for data, complex_data_list in loader:
            data.to(device)
            for complex_data in complex_data_list:
                complex_data.to(device)

            x = model.gin_encoder(data)
            x_c1 = model.encoder(complex_data_list[0])
            x_c2 = model.encoder(complex_data_list[1])
            x_c3 = model.encoder(complex_data_list[2])
            ret.append(x.cpu().numpy())
            ret_c1.append(x_c1.cpu().numpy())
            ret_c2.append(x_c2.cpu().numpy())
            ret_c3.append(x_c3.cpu().numpy())
            y.append(data.y.cpu().numpy())

    ret = np.concatenate(ret, 0)
    ret_c1 = np.concatenate(ret_c1, 0)
    ret_c2 = np.concatenate(ret_c2, 0)
    ret_c3 = np.concatenate(ret_c3, 0)
    y = np.concatenate(y, 0)
    return ret, ret_c1, ret_c2, ret_c3, y

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    """The common training and evaluation script used by all the experiments."""
    # set device
    device = torch.device(
        "cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")


    print("==========================================================")
    print("Using device", str(device))
    print(f"Seed: {args.seed}")
    print("======================== Args ===========================")
    print(args)
    print("===================================================")

    # Set the seed for everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create results folder
    result_folder = os.path.join(
        args.result_folder, f'{args.dataset}-{args.exp_name}', f'seed-{args.seed}')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    filename = os.path.join(result_folder, 'results.txt')

    # Data loading
    dataset = load_dataset(args.dataset, max_dim=args.max_dim,
                           init_method=args.init_method, emb_dim=args.emb_dim,
                           max_ring_sizes=args.max_ring_sizes,
                           use_edge_features=args.use_edge_features,
                           include_down_adj=args.include_down_adj,
                           simple_features=args.simple_features, n_jobs=args.preproc_jobs)

    dataset_eval = load_dataset(args.dataset, max_dim=args.max_dim,
                           init_method=args.init_method, emb_dim=args.emb_dim,
                           max_ring_sizes=args.max_ring_sizes,
                           use_edge_features=args.use_edge_features,
                           include_down_adj=args.include_down_adj,
                           simple_features=args.simple_features, n_jobs=args.preproc_jobs)

    # Instantiate data loaders
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, max_dim=dataset.max_dim)
    dataloader_eval = DataLoader(dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, max_dim=dataset.max_dim)


    model = simclr(dataset, args).to(device)


    print("============= Model Parameters =================")
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.size())
            trainable_params += param.numel()
        total_params += param.numel()
    print("============= Params stats ==================")
    print(f"Trainable params: {trainable_params}")
    print(f"Total params    : {total_params}")

    # instantiate optimiser
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # (!) start training/evaluation
    best_val_epoch = 0
    valid_curve = []
    test_curve = []
    train_curve = []
    train_loss_curve = []
    params = []

    accuracies = {'val_cat': [], 'val_add': [], 'val_mean': [], 'test_cat': [], 'test_add': [], 'test_mean': []}

    print('Training...')

    for epoch in range(1, args.epochs + 1):

        # perform one epoch
        print("=====Epoch {}".format(epoch))


        loss_all = 0

        model.train()
        num_skips = 0
        for data_batch, complex_batch_list in dataloader:
            optimizer.zero_grad()
            data_batch = data_batch.to(device)
            for complex_batch in complex_batch_list:
                complex_batch = complex_batch.to(device)

            x = model.forward_gin(data_batch)
            y1 = model.forward_cwn(complex_batch_list[0])
            y2 = model.forward_cwn(complex_batch_list[1])
            y3 = model.forward_cwn(complex_batch_list[2])

            loss1 = simclr.simclr_loss(x, y1)
            loss2 = simclr.simclr_loss(x, y2)
            loss3 = simclr.simclr_loss(x, y3)
            loss = loss1 + loss2 + loss3
            loss_all += loss.item()
            loss.backward()
            optimizer.step()
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))

        if epoch % 1 == 0:
            model.eval()

            emb, emb_c1, emb_c2, emb_c3, y = get_all_embeddings(model, dataloader_eval, device)

            emb_cat = np.concatenate([emb, emb_c1, emb_c2, emb_c3], axis=1)
            emb_mean = (emb + emb_c1 + emb_c2 + emb_c3)/4

            acc_cat_val, acc_cat = evaluate_embedding(emb_cat, y)
            acc_mean_val, acc_mean  = evaluate_embedding(emb_mean, y)

            accuracies['val_cat'].append(acc_cat_val)
            accuracies['test_cat'].append(acc_cat)
            accuracies['val_mean'].append(acc_mean_val)
            accuracies['test_mean'].append(acc_mean)

    msg = (
        f'========Params========\n'
        f'args: {args}\n'
        f'========Result=======\n'
        f'Dataset: {args.dataset}\n'
        f"accuracies[val_cat]: {accuracies['val_cat']}\n"
        f"accuracies[test_cat]: {accuracies['test_cat']}\n"
        f"Test cat max:\n{max(accuracies['test_cat'])}\n"
        f"accuracies[val_mean]: {accuracies['val_mean']}\n"
        f"accuracies[test_mean]: {accuracies['test_mean']}\n"
        f"Test mean max:\n{max(accuracies['test_mean'])}\n"
    )

    print(msg)

    with open(filename, 'a') as f:
        f.write(msg)
import os
import numpy as np
import copy
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import random

from data.complex import ComplexBatch
from data.data_loading import DataLoader, load_dataset
from torch_geometric.data import DataLoader as PyGDataLoader

from data.datasets import TUDataset
from unsupervised.parser import get_parser
from unsupervised.model import SimCLRWithTMDC

from unsupervised.evaluate_embedding import evaluate_embedding


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

    model = SimCLRWithTMDC(dataset, args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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

        total_loss = 0

        model.train()
        num_skips = 0
        for data_batch, complex_batch_list in dataloader:
            optimizer.zero_grad()
            data_batch = data_batch.to(device)
            complex_batch_list = [cb.to(device) for cb in complex_batch_list]

            # raw features: dict with keys 'view0', 'view1', 'view2', 'view3'
            raw_feats = model.forward_encodings(data_batch, complex_batch_list)

            loss = model.compute_total_loss(raw_feats, alpha=args.alpha, beta=args.beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print('Epoch {}, Loss {}'.format(epoch, total_loss / len(dataloader)))

        if epoch % 1 == 0:
            model.eval()

            emb_cat, y = model.get_embeddings(dataloader_eval, device, fusion='concat')
            emb_mean, y = model.get_embeddings(dataloader_eval, device, fusion='mean')

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
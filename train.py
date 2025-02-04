import argparse
import torch.nn.functional as F
import torch
import random
from torch import tensor
from networks import CPFGNN
import numpy as np
from utils import partition_index, tuned_params
from dataset import load_nc_dataset
from PARAMETERS import PARAMS
# import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--r_train', type=float, default=0.6)
    parser.add_argument('--r_val', type=float, default=0.2)

    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--K', type=int, default=10)

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--prop_g_lr', type=float, default=0.01)
    parser.add_argument('--prop_f_lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--prop_g_wd', type=float, default=0.0005)
    parser.add_argument('--prop_f_wd', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--dprate', type=float, default=0.5)

    parser.add_argument('--coarsening_ratio', type=float, default=0.5)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')

    parser.add_argument('--finetuned', action='store_true')

    args = parser.parse_args()
    # path = "params/"
    # if not os.path.isdir(path):
    #     os.mkdir(path)
    if args.finetuned:
        tuned_params(args, PARAMS)

    # print(
    # args.dataset.lower(),
    # args.lr ,
    # args.prop_g_lr,
    # args.prop_f_lr,
    # args.weight_decay,
    # args.prop_g_wd,
    # args.prop_f_wd,
    # args.dropout,
    # args.dprate,
    # args.r_train,
    # args.r_val
    # )

    # define # runs number of seeds, for reproducibility
    SEED = [i for i in range(1, args.runs + 1)]

    device = 'cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu'
    print('Model will be executed on ' + device)

    dataset = load_nc_dataset(args.dataset)
    labels = dataset.label.to(device)

    # print("3")


################################################################################
# NOTE: major code
################################################################################


    print('--Graph partition precomputing.')

    data = partition_index(dataset, 1-args.coarsening_ratio, 
                       args.coarsening_method, device)
    args.num_features, args.num_classes = dataset.num_features, dataset.num_classes

    print('--Preprocessing completed. GNN training begins.')
    
    all_acc = []

    for seed in SEED:
        # NOTE: for consistent data splits, see data_utils.rand_train_test_idx
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        splits = dataset.get_idx_split(train_prop=args.r_train, valid_prop=args.r_val)
        
        # original data
        train_idx = splits['train'].to(device)
        val_idx = splits['valid'].to(device)
        test_idx = splits['test'].to(device)

        model = CPFGNN(args, data).to(device)

        optimizer = torch.optim.Adam([
            {
            'params': model.prop_g.parameters(), 
            'weight_decay': args.prop_g_wd, 'lr': args.prop_g_lr
        },
            {
            'params': model.prop_f.parameters(), 
            'weight_decay': args.prop_f_wd, 'lr': args.prop_f_lr
        },
            {
            'params': model.lin1.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.lin2.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        }
        ])

        best_val_acc = float(0)
        stop = 0

        for epoch in range(args.epochs):

            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[train_idx], labels[train_idx])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                pred = model(data).max(1)[1]
                val_acc = int(pred[val_idx].eq(labels[val_idx]).sum().item()) / int(val_idx.shape[0])

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    stop = 0
                    # torch.save(model.state_dict(), path + 'checkpoint-best-acc.pkl')
                    test_acc = int(pred[test_idx].eq(labels[test_idx]).sum().item()) / int(test_idx.shape[0])

                stop += 1

            if stop > args.early_stopping and args.early_stopping > 0:
                break

        # model.load_state_dict(torch.load(path + 'checkpoint-best-acc.pkl'))
        
        print(test_acc)
        all_acc.append(test_acc)

    print('ave_acc: {:.4f}'.format(np.mean(all_acc)))
    print('std: {:.4f}'.format(np.std(all_acc)))
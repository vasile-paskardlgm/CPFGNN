import argparse
import torch.nn.functional as F
import torch
import random
from torch import tensor
from networks import CPFGNN
import numpy as np
from utils import load_data, coarsening, IndexSubgraph, MatrixCTC
from dataset import load_nc_dataset
# import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--gpu', default=True)
    parser.add_argument('--r_train', type=float, default=0.6)
    parser.add_argument('--r_val', type=float, default=0.2)

    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--Init', type=str,
                        default='Mine_PPR')
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--rank', type=int, default=3)

    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--prop_lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--prop_wd', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--dprate', type=float, default=0.5)

    parser.add_argument('--coarsening_ratio', type=float, default=0.1)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')

    args = parser.parse_args()
    # path = "params/"
    # if not os.path.isdir(path):
    #     os.mkdir(path)

    # define # runs number of seeds, for reproducibility
    SEED = [i for i in range(1, args.runs + 1)]

    device = torch.device('cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu')

    dataset = load_nc_dataset(args.dataset)
    feature = dataset.graph['node_feat'].to(device)
    feature = F.normalize(feature, p=1)
    edges = dataset.graph['edge_index'].to(device)
    labels = dataset.label.to(device)


################################################################################
# NOTE: major code
################################################################################


    print('--Graph partition precomputing.')

    args.num_features, args.num_classes, candidate, C_list, _ = coarsening(dataset,
                                                                                     1-args.coarsening_ratio,
                                                                                     args.coarsening_method)
    CTC = MatrixCTC(candidate, C_list).to(device)

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

        model = CPFGNN(args).to(device)

        optimizer = torch.optim.Adam([{
            'params': model.lin1.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.lin2.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.prop1.parameters(),
            'weight_decay': args.prop_wd, 'lr': args.prop_lr
        }
        ],
            lr=args.lr)

        best_val_acc = float(0)
        stop = 0

        for epoch in range(args.epochs):

            model.train()
            optimizer.zero_grad()
            out = model(feature, edges, CTC)
            loss = F.nll_loss(out[train_idx], labels[train_idx])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                pred = model(feature, edges, CTC).max(1)[1]
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
# from torch_geometric.datasets import Planetoid
# from torch_geometric.datasets import Coauthor
# from torch_geometric.datasets import CitationFull
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from graph_partition.partition_utils import *
from math import sqrt


def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]

def extract_components(H):
        if H.A.shape[0] != H.A.shape[1]:
            H.logger.error('Inconsistent shape to extract components. '
                           'Square matrix required.')
            return None

        if H.is_directed():
            raise NotImplementedError('Directed graphs not supported yet.')

        graphs = []

        visited = np.zeros(H.A.shape[0], dtype=bool)

        while not visited.all():
            stack = set([np.nonzero(~visited)[0][0]])
            comp = []

            while len(stack):
                v = stack.pop()
                if not visited[v]:
                    comp.append(v)
                    visited[v] = True

                    stack.update(set([idx for idx in H.A[v, :].nonzero()[1]
                                      if not visited[idx]]))

            comp = sorted(comp)
            G = H.subgraph(comp)
            G.info = {'orig_idx': comp}
            graphs.append(G)

        return graphs

def coarsening(dataset, coarsening_ratio, coarsening_method):
    G = gsp.graphs.Graph(W=to_dense_adj(dataset.graph['edge_index'])[0])
    components = extract_components(G)
    # print('the number of subgraphs is', len(components))
    candidate = sorted(components, key=lambda x: len(x.info['orig_idx']), reverse=True)
    number = 0
    C_list=[]

    while number < len(candidate):
        H = candidate[number]
        if len(H.info['orig_idx']) > 10:
            C = coarsen(H, r=coarsening_ratio, method=coarsening_method)[0]
            C_list.append(C)
        number += 1
    return dataset.num_features, dataset.num_classes, candidate, C_list


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def indexsubgraph(candidate, C_list):
  '''
  (subgraphs/partitions/supernodes, nodes) indice pairs from C_list
  no weight is considered
  '''
  sub_g = None ## subgraphs/partitions/supernodes' indices
  nodes = None ## nodes' indices

  number = 0
  # col match keep
  while number < len(candidate):
    # matching indices to original graph
    H = candidate[number]
    keep = H.info['orig_idx']

    if len(H.info['orig_idx']) > 10: # if the graph is coarsened
      C = C_list[number].tocoo()

      if (sub_g is None) and (nodes is None):
        sub_g = np.array(C.row)
        nodes = np.array(keep)
      else:
        sub_g = np.concatenate([sub_g, C.row + sub_g.max() + 1])
        nodes = np.concatenate([nodes, keep])
    else: # if the graph is not coarsened
      sub_g = np.concatenate([sub_g, np.arange(stop=len(keep)) + sub_g.max() + 1])
      nodes = np.concatenate([nodes, keep])
    number += 1
  
  nodes = torch.from_numpy(nodes)
  sub_g = torch.from_numpy(sub_g)

  return torch.stack([nodes, sub_g], dim=0) # matching: Subgraph No. <- Original nodes No.


def partition_index(dataset, coarsening_ratio, coarsening_method, device):

    if coarsening_method in ('variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron'):
        coarsen_f = coarsening
    else:
        raise ValueError('Invalid dataname')
    
    dataset.num_features, dataset.num_classes, candidate, C_list = coarsen_f(dataset, coarsening_ratio, coarsening_method)

    indices = indexsubgraph(candidate,C_list)
    values = torch.ones(indices.shape[1])

    for i in range(indices[1,:].max() + 1):
        idx = (indices[1,:] == i)
        values[idx] = 1 / sqrt(idx.sum())
    
    partition = torch.sparse_coo_tensor(indices=indices, values=values).to(device).coalesce() # n x n'

    del indices
    del values

    feature = dataset.graph['node_feat'].to(device)
    feature = F.normalize(feature, p=1)
    edges = dataset.graph['edge_index'].to(device)
    
    return feature, edges, partition


def tuned_params(args, params: dict):
    if args.dataset.lower() not in params.keys():
        raise ValueError(f'This dataset {args.dataset.lower()} has not been finetuned')
    
    hyperparameters = params[args.dataset.lower()]
    args.lr = hyperparameters['lr']
    args.prop_g_lr = hyperparameters['prop_g_lr']
    args.prop_f_lr = hyperparameters['prop_f_lr']
    args.weight_decay = hyperparameters['weight_decay']
    args.prop_g_wd = hyperparameters['prop_g_wd']
    args.prop_f_wd = hyperparameters['prop_f_wd']
    args.dropout = hyperparameters['dropout']
    args.dprate = hyperparameters['dprate']
    args.r_train = hyperparameters['r_train']
    args.r_val = hyperparameters['r_val']
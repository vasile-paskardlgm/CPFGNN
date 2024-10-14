# from torch_geometric.datasets import Planetoid
# from torch_geometric.datasets import Coauthor
# from torch_geometric.datasets import CitationFull
import torch
from torch_geometric.utils import to_dense_adj
from graph_partition.partition_utils import *

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
    Gc_list=[]
    while number < len(candidate):
        H = candidate[number]
        if len(H.info['orig_idx']) > 10:
            C, Gc, Call, Gall = coarsen(H, r=coarsening_ratio, method=coarsening_method)
            C_list.append(C)
            Gc_list.append(Gc)
        number += 1
    return dataset.num_features, dataset.num_classes, candidate, C_list, Gc_list

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def load_data(dataset, candidate, C_list, Gc_list):
    splits = dataset.get_idx_split() # here the (random) data splits is obtained and need to be hold
    train_mask = index_to_mask(splits['train'], size=dataset.num_nodes)
    val_mask = index_to_mask(splits['valid'], size=dataset.num_nodes)
    labels = dataset.label
    features = dataset.graph['node_feat']
    n_classes = len(set(np.array(labels)))

    coarsen_node = 0
    number = 0
    coarsen_row = None
    coarsen_col = None
    coarsen_features = torch.Tensor([])
    coarsen_train_labels = torch.Tensor([])
    coarsen_train_mask = torch.Tensor([]).bool()
    coarsen_val_labels = torch.Tensor([])
    coarsen_val_mask = torch.Tensor([]).bool()

    while number < len(candidate):
        H = candidate[number]
        keep = H.info['orig_idx']
        H_features = features[keep]
        H_labels = labels[keep]
        H_train_mask = train_mask[keep]
        H_val_mask = val_mask[keep]
        if len(H.info['orig_idx']) > 10 and torch.sum(H_train_mask)+torch.sum(H_val_mask) > 0:
            train_labels = one_hot(H_labels, n_classes)
            train_labels[~H_train_mask] = torch.Tensor([0 for _ in range(n_classes)])
            val_labels = one_hot(H_labels, n_classes)
            val_labels[~H_val_mask] = torch.Tensor([0 for _ in range(n_classes)])
            C = C_list[number]
            Gc = Gc_list[number]

            new_train_mask = torch.BoolTensor(np.sum(C.dot(train_labels), axis=1))
            mix_label = torch.FloatTensor(C.dot(train_labels))
            mix_label[mix_label > 0] = 1
            mix_mask = torch.sum(mix_label, dim=1)
            new_train_mask[mix_mask > 1] = False

            new_val_mask = torch.BoolTensor(np.sum(C.dot(val_labels), axis=1))
            mix_label = torch.FloatTensor(C.dot(val_labels))
            mix_label[mix_label > 0] = 1
            mix_mask = torch.sum(mix_label, dim=1)
            new_val_mask[mix_mask > 1] = False

            coarsen_features = torch.cat([coarsen_features, torch.FloatTensor(C.dot(H_features))], dim=0)
            coarsen_train_labels = torch.cat([coarsen_train_labels, torch.argmax(torch.FloatTensor(C.dot(train_labels)), dim=1).float()], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, new_train_mask], dim=0)
            coarsen_val_labels = torch.cat([coarsen_val_labels, torch.argmax(torch.FloatTensor(C.dot(val_labels)), dim=1).float()], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, new_val_mask], dim=0)

            if coarsen_row is None:
                coarsen_row = Gc.W.tocoo().row
                coarsen_col = Gc.W.tocoo().col
            else:
                current_row = Gc.W.tocoo().row + coarsen_node
                current_col = Gc.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += Gc.W.shape[0]

        elif torch.sum(H_train_mask)+torch.sum(H_val_mask)>0:

            coarsen_features = torch.cat([coarsen_features, H_features], dim=0)
            coarsen_train_labels = torch.cat([coarsen_train_labels, H_labels.float()], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, H_train_mask], dim=0)
            coarsen_val_labels = torch.cat([coarsen_val_labels, H_labels.float()], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, H_val_mask], dim=0)

            if coarsen_row is None:
                raise Exception('The graph does not need coarsening.')
            else:
                current_row = H.W.tocoo().row + coarsen_node
                current_col = H.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += H.W.shape[0]
        number += 1

    print('the size of coarsen graph features:', coarsen_features.shape)

    coarsen_edge = torch.LongTensor([coarsen_row, coarsen_col])
    coarsen_train_labels = coarsen_train_labels.long()
    coarsen_val_labels = coarsen_val_labels.long()

    return splits, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge

def IndexSubgraph(candidate, C_list):
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

def MatrixCTC(candidate,C_list):
    indices = IndexSubgraph(candidate,C_list)
    values = torch.ones(indices.shape[1])

    for i in range(indices[1,:].max() + 1):
        idx = (indices[1,:] == i)
        values[idx] = 1 / idx.sum()
    
    CTC = torch.sparse_coo_tensor(indices=indices, values=values)
    
    return torch.sparse.mm(CTC, CTC.T)
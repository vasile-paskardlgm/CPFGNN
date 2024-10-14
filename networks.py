from typing import Optional
from torch_geometric.typing import OptTensor

import torch
import torch.nn.functional as F
import numpy as np

from torch.nn import Parameter,Linear, ModuleList
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian, remove_self_loops


class SGP_prop(MessagePassing):
    '''
    propagation class for CPFGNN
    '''

    def __init__(self, K, alpha, Init, num_classes, rank=3, Gamma=None, bias=True, **kwargs):
        super(SGP_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha


        if Init == 'Mine':
            TEMP = []
            para = torch.ones([rank, K+1])
            TEMP = torch.nn.init.xavier_normal_(para)
        elif Init == 'Mine_PPR':
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
            TEMP = torch.tensor(np.array([TEMP] * rank))
            
        self.gamma = Parameter(TEMP.float())
        
#         self.proj = Linear(num_classes, rank)
        proj_list = []
        for _ in range(K + 1):
            proj_list.append(Linear(num_classes, rank))
        self.proj_list = ModuleList(proj_list)
        self.rank = rank
        
    def __norm__(self, edge_index, num_nodes: Optional[int],
                 edge_weight: OptTensor, normalization: Optional[str] = "sym",
                 lambda_max: OptTensor = None, dtype: Optional[int] = None,
                 batch: OptTensor = None):

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)

        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()
        elif not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=dtype,
                                      device=edge_index.device)
        assert lambda_max is not None

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 fill_value=-1.,
                                                 num_nodes=num_nodes)
        assert edge_weight is not None

        return edge_index, edge_weight

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, gamma={})'.format(self.__class__.__name__, self.K,
                                          self.gamma)

    def forward(self, CTC, x, edge_index, edge_weight=None):
        edge_index, norm = self.__norm__(edge_index, x.size(0), edge_weight, dtype=x.dtype)
        # x_list, eta_list = [], []
        Tx_0 = x
        Tx_1 = self.propagate(edge_index, x=Tx_0, norm=norm)

        ## partition sharing;
        h_0 = torch.tanh(self.proj_list[0](Tx_0))
        h_1 = torch.tanh(self.proj_list[1](Tx_1))
        gamma_0 = self.gamma[:,0].unsqueeze(dim=-1)
        gamma_1 = self.gamma[:,1].unsqueeze(dim=-1)
        eta_0   = torch.matmul(h_0, gamma_0)/self.rank
        eta_1   = torch.matmul(h_1, gamma_1)/self.rank
        eta_0 = torch.mm(CTC, eta_0)
        eta_1 = torch.mm(CTC, eta_1)
        
        hidden = torch.matmul(Tx_0.unsqueeze(dim=-1), eta_0.unsqueeze(dim=-1)).squeeze(dim=-1)
        hidden = hidden + torch.matmul(Tx_1.unsqueeze(dim=-1), eta_1.unsqueeze(dim=-1)).squeeze(dim=-1)
        
        for k in range(1, self.K):
            Tx_2 = 2. * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0
            Tx_0, Tx_1 = Tx_1, Tx_2
            h_k     = torch.tanh(self.proj_list[k+1](Tx_1))
            gamma_k = self.gamma[:,k+1].unsqueeze(dim=-1)
            eta_k   = torch.matmul(h_k, gamma_k)/self.rank
            eta_k   = torch.mm(CTC, eta_k)
            hidden = hidden + torch.matmul(Tx_1.unsqueeze(dim=-1), eta_k.unsqueeze(dim=-1)).squeeze(dim=-1)
            
        return hidden


class CPFGNN(torch.nn.Module):
    def __init__(self, args):
        super(CPFGNN, self).__init__()
        self.lin1 = Linear(args.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, args.num_classes)

        self.prop1 = SGP_prop(args.K, args.alpha, args.Init, args.num_classes, args.rank, args.Gamma)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def forward(self, feature, edges, CTC):
        x, edge_index = feature, edges
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)  
        
        x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.prop1(CTC, x, edge_index)
        
        return F.log_softmax(x, dim=1)
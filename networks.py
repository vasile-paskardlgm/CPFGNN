from typing import Optional
from torch_geometric.typing import OptTensor
import math
import torch
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from torch.nn import Parameter, ModuleList, Linear, init
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian, remove_self_loops
import fast_pytorch_kmeans as fpk


#############################
def poly(i,x):
    if i==0:
        return 1
    elif i==1:
        return x
    else:
        T0=1
        T1=x
        for _ in range(2,i+1):
            T2=2*x*T1-T0
            T0,T1=T1,T2
        return T2


#############################
class CPF_prop_g(MessagePassing):
    '''
    structure-aspect partition filtering
    '''

    def __init__(self, n_dash : int, K: int,
                 normalization: Optional[str] = 'sym', bias: bool = True,
                 **kwargs):
        super(CPF_prop_g, self).__init__(aggr='add', **kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.K = K
        self.normalization = normalization
        self.CLUSTER = Parameter(torch.Tensor(n_dash,K+1))

        self.reset_parameters()

    def reset_parameters(self):
        self.CLUSTER.data.fill_(0.5)
        
    def __norm__(self, edge_index, num_nodes: Optional[int],
                 edge_weight: OptTensor, normalization: Optional[str],
                 lambda_max: OptTensor = None, dtype: Optional[int] = None):

        #edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)

        lambda_max = 2.0 * edge_weight.max()
        assert lambda_max is not None

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
        return 'CPFGNN'

    def forward(self, x: Tensor, edge_index: Tensor, partition: Tensor,
                edge_weight: OptTensor = None, lambda_max: OptTensor = None):
        

        #primary paradigm
        lifting = torch.sparse.mm(partition, self.CLUSTER)

        coe_tmp = F.relu(lifting.mean(dim=0))
        coe=coe_tmp.clone()
        
        for i in range(self.K+1):
            coe[i]=coe_tmp[0]*poly(i,math.cos((self.K+0.5)*math.pi/(self.K+1)))
            for j in range(1,self.K+1):
                x_j=math.cos((self.K-j+0.5)*math.pi/(self.K+1))
                coe[i]=coe[i]+coe_tmp[j]*poly(i,x_j)
            coe[i]=2*coe[i]/(self.K+1)

        #coefficient on partition -> lifting
        lifting = F.normalize(lifting, p=1, dim=0) @ coe.reshape(shape=(coe.shape[0],1))

        # propagate_type: (x: Tensor, norm: Tensor)
        edge_index_tilde, norm_tilde = self.__norm__(edge_index, x.size(self.node_dim),
                                         edge_weight, self.normalization,
                                         lambda_max, dtype=x.dtype)

        Tx_0 = x
        out = coe[0]/2 * Tx_0 * (lifting + 1) + Tx_0 

        Tx_1 = self.propagate(edge_index_tilde, x=Tx_0, norm=norm_tilde, size=None)
        out = out + coe[1] * Tx_1 * (lifting + 1) # + Tx_1  

        for k in range(2,self.K + 1):
            Tx_2 = self.propagate(edge_index_tilde, x=Tx_1, norm=norm_tilde, size=None)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + coe[k] * Tx_2 * (lifting + 1) # + Tx_2 
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out

        r'''
        #another paradigm
        coe_tmp = F.relu(self.CLUSTER)
        coe=coe_tmp.clone()
        
        for i in range(self.K+1):
            coe[:,i]=coe_tmp[:,0]*poly(i,math.cos((self.K+0.5)*math.pi/(self.K+1)))
            for j in range(1,self.K+1):
                x_j=math.cos((self.K-j+0.5)*math.pi/(self.K+1))
                coe[:,i]=coe[:,i]+coe_tmp[:,j]*poly(i,x_j)
            coe[:,i]=2*coe[:,i]/(self.K+1)

        coe = torch.sparse.mm(partition, coe)
        # coe = F.normalize(coe, p=1, dim=0)

        #propagate
        edge_index_tilde, norm_tilde = self.__norm__(edge_index, x.size(self.node_dim),
                                         edge_weight, self.normalization,
                                         lambda_max, dtype=x.dtype)

        Tx_0 = x
        out = Tx_0 * coe[:,0:1]/2 + Tx_0 

        # propagate_type: (x: Tensor, norm: Tensor)
        Tx_1 = self.propagate(edge_index_tilde, x=Tx_0, norm=norm_tilde, size=None)
        out = out + Tx_1 * coe[:,1:2] # + Tx_1  

        for k in range(2,self.K + 1):
            Tx_2 = self.propagate(edge_index_tilde, x=Tx_1, norm=norm_tilde, size=None)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + Tx_2 * coe[:,k:k+1]# + Tx_2 
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out
        '''


#############################
class CPF_prop_f(torch.nn.Module):
    '''
    feature-aspect partition centralization
    '''
    def __init__(
        self,
        num_partition: int,
        channels: int,
        activation=F.tanhshrink,
        mode="euclidean",
        centralization=True,
    ):
        super(CPF_prop_f, self).__init__()

        self.num_partition = num_partition
        self.activation = activation
        self.centralization = centralization
        self.mode = mode

        self.recluster_threshold = 0.5

        self.kmeans = fpk.KMeans(n_clusters=self.num_partition, mode=self.mode)

        self.prev_centroids = None
        self.init_check = None
        self.eps = 1

        TEMP = torch.empty(channels, channels, self.num_partition)
        init.xavier_normal_(TEMP, gain=0.5)
        self.partition_weighting = Parameter(data=TEMP)

    def reset_parameters(self):
        init.xavier_normal_(self.partition_weighting, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().clone()

        if self.init_check is None:
            self.init_check = self.kmeans.fit_predict(x)

        if self._check_recluster_threshold(x):
            self._recluster(x)

        cluster_indices = self.get_cluster_indices(x)
        """temp_m, temp_n = x.size()
        assert temp_n % self.num_partition == 0
        x = x.view(temp_m, self.num_partition, -1)
        x = (x - x.mean(-1, keepdim=True)) / (x.var(-1, keepdim=True) + self.eps).sqrt()
        x = x.view(temp_m, temp_n)
        x = x * self.weight + self.bias"""
        for c in range(self.num_partition):
            if cluster_indices[c].numel() != 0:
                cluster = x[cluster_indices[c]].clone()
                if self.centralization:
                    cluster = (cluster - cluster.mean(-1, keepdim=True)) / (cluster.var(-1, keepdim=True) + self.eps).sqrt()
                # cluster = cluster * self.weight[c] + self.bias[c]
                x[cluster_indices[c]] = cluster @ self.partition_weighting[:,:,c]

        if self.activation == None:
            return x
        else:
            return self.activation(x)

    def _recluster(self, x: torch.Tensor) -> None:
        self.kmeans.fit(x)
        self.prev_centroids = self.kmeans.centroids.clone()

    def _check_recluster_threshold(self, x: torch.Tensor) -> bool:
        if self.prev_centroids is None:
            return True

        # Fit the KMeans model to get the current cluster centroids
        self.kmeans.fit(x)
        current_centroids = torch.tensor(self.kmeans.centroids.clone())

        centroid_distances = torch.cdist(current_centroids, self.prev_centroids)
        mean_distances = centroid_distances.min(dim=1).values
        cluster_means = [
            x[torch.nonzero(self.kmeans.predict(x) == i)].mean(dim=0)
            for i in range(self.num_partition)
        ]

        for i in range(self.num_partition):
            distance_limit = mean_distances[i] * self.recluster_threshold
            if torch.norm(cluster_means[i] - current_centroids[i]) > distance_limit:
                return True

        return False

    def get_cluster_indices(self, x: torch.Tensor) -> torch.Tensor:
        if self.init_check is None:
            self.init_check = self.kmeans.fit_predict(x)
        labels = self.kmeans.predict(x)
        cluster_indices = [
            torch.nonzero(labels == index).squeeze().cpu()
            for index in range(self.num_partition)
        ]
        return cluster_indices


#############################
class CPFGNN(torch.nn.Module):
    def __init__(self, args, data):
        super(CPFGNN, self).__init__()
        self.lin1 = Linear(args.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, args.num_classes)

        self.prop_g = CPF_prop_g(data[2].shape[1], args.K)

        self.prop_f = CPF_prop_f(args.num_classes, args.num_classes)

        # self.prop_f = Linear(args.num_classes, args.num_classes, bias=False)

        self.dprate = args.dprate
        self.dropout = args.dropout

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop_g.reset_parameters()
        self.prop_f.reset_parameters()

    def forward(self, data):
        x, edge_index, partition = data[0], data[1], data[2]

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)  
        
        x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.prop_g(x, edge_index, partition)

        x = self.prop_f(x) + x
        
        return F.log_softmax(x, dim=1)
from tsl.nn.blocks.encoders import RNN
from tsl.nn.blocks.decoders import GCNDecoder
from tsl.nn.models import BaseModel

from typing import Optional
from einops import rearrange
from torch import nn, Tensor
from torch_geometric.typing import Adj, OptTensor
from tsl.nn.blocks.decoders.mlp_decoder import MLPDecoder
from tsl.nn.blocks.encoders import ConditionalBlock
from tsl.nn.blocks.encoders.dcrnn import DCRNN
import torch
from typing import Optional, Tuple, Union, List
from tsl.typing import TensArray, OptTensArray, SparseTensArray, DataArray, ScipySparseMatrix
from types import ModuleType
import numpy as np


class TimeThenSpaceModel(torch.nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 rnn_layers,
                 gcn_layers,
                 horizon):
        super(TimeThenSpaceModel, self).__init__()

        self.input_encoder = torch.nn.Linear(input_size, hidden_size)

        self.encoder = RNN(input_size=hidden_size,
                           hidden_size=hidden_size,
                           n_layers=rnn_layers)

        self.decoder = GCNDecoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=input_size,
            horizon=horizon,
            n_layers=gcn_layers
        )

    # def forward(self, x):
    def forward(self, x, edge_index, edge_weight):

        # print(f'xshape0={x.shape}')

        # x: [batches steps nodes channels]
        x = self.input_encoder(x)
        # print(f'xshape1={x.shape}')
        x = self.encoder(x, return_last_state=True)
        # print(f'xshape2={x.shape}')
        # print(f'xshape3={self.decoder(x, edge_index, edge_weight).shape}')

        return self.decoder(x, edge_index, edge_weight)
        # return self.decoder(x)
    
    


class DCRNNModel_manual(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, horizon: int,
                 exog_size: int = 0,
                 hidden_size: int = 32,
                 kernel_size: int = 2,
                 ff_size: int = 256,
                 n_layers: int = 1,
                 dropout: float = 0.,
                 activation: str = 'relu'):
        super(DCRNNModel_manual, self).__init__()
        if exog_size:
            self.input_encoder = ConditionalBlock(input_size=input_size,
                                                  exog_size=exog_size,
                                                  output_size=hidden_size,
                                                  activation=activation)
        else:
            self.input_encoder = nn.Linear(input_size, hidden_size)

        self.dcrnn = DCRNN(input_size=hidden_size,
                           hidden_size=hidden_size,
                           n_layers=n_layers,
                           k=kernel_size)

        self.readout = MLPDecoder(input_size=hidden_size,
                                  hidden_size=ff_size,
                                  output_size=output_size,
                                  horizon=horizon,
                                  activation=activation,
                                  dropout=dropout)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None,
                u: OptTensor = None) -> Tensor:
        """"""
        if u is not None:
            if u.dim() == 3:
                u = rearrange(u, 'b s c -> b s 1 c')
            x = self.input_encoder(x, u)
        else:
            x = self.input_encoder(x)

        h, _ = self.dcrnn(x, edge_index, edge_weight)
        return self.readout(h)
    
def infer_backend(obj, backend: ModuleType = None):
    if backend is not None:
        return backend
    elif isinstance(obj, Tensor):
        return torch
    elif isinstance(obj, np.ndarray):
        return np
    elif isinstance(obj, SparseTensor):
        return torch_sparse
    raise RuntimeError(f"Cannot infer valid backed from {type(obj)}.")



def adj_to_edge_index(adj: TensArray, backend: ModuleType = None) \
        -> Tuple[TensArray, TensArray]:
            
    backend = infer_backend(adj, backend)
    print(f'{backend=}')
    assert backend in [torch, np]
    assert 2 <= adj.ndim <= 3
    assert adj.shape[-1] == adj.shape[-2]

    if backend is torch:
        print('it is torch')
        adj = torch.transpose(adj, -2, -1)
        index = adj.nonzero(as_tuple=True)
    else:
        print('it is not torch')
        
        adj = np.swapaxes(adj, -2, -1)  # transpose
        index = adj.nonzero()

    edge_attr = adj[index]

    print(len(index))
    if len(index) == 3:
        print('did we even came here')
        batch = index[0] * adj.shape[-1]
        index = (batch + index[1], batch + index[2])

    edge_index = backend.stack(index, 0)

    # return edge_index, np.asarray(edge_attr, dtype='float32')[0,:]
    return edge_index, edge_attr

class GRUGCNModel(BaseModel):
    r"""
    Simple time-then-space model with a GRU encoder and a GCN decoder.
    From Guo et al., "On the Equivalence Between Temporal and Static Equivariant Graph Representations", ICML 2022.
    Args:
        input_size (int): Size of the input.
        hidden_size (int): Number of hidden units in each hidden layer.
        output_size (int): Size of the output.
        horizon (int): Forecasting steps.
        exog_size (int): Size of the optional exogenous variables.
        enc_layers (int): Number of layers in the GRU encoder.
        gcn_layers (int): Number of GCN layers in GCN decoder.
        asymmetric_norm (bool): Whether to use asymmetric or GCN normalization.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 horizon,
                 exog_size,
                 enc_layers,
                 gcn_layers,
                 asymmetric_norm,
                 encode_edges=False,
                 activation='softplus'):
        super(GRUGCNModel, self).__init__()

        input_size += exog_size
        self.input_encoder = RNN(input_size=input_size,
                                 hidden_size=hidden_size,
                                 n_layers=enc_layers,
                                 return_only_last_state=True,
                                 cell='gru')

        if encode_edges:
            self.edge_encoder = nn.Sequential(
                RNN(input_size=input_size,
                    hidden_size=hidden_size,
                    n_layers=enc_layers,
                    return_only_last_state=True,
                    cell='gru'),
                nn.Linear(hidden_size, 1),
                nn.Softplus(),
                Rearrange('e f -> (e f)', f=1)
            )
        else:
            self.register_parameter('edge_encoder', None)

        self.gcn_layers = nn.ModuleList(
            [
                GraphConv(hidden_size,
                          hidden_size,
                          root_weight=False,
                          asymmetric_norm=asymmetric_norm,
                          activation=activation) for _ in range(gcn_layers)
            ]
        )

        self.skip_con = nn.Linear(hidden_size, hidden_size)

        self.readout = MLPDecoder(input_size=hidden_size,
                                  hidden_size=hidden_size,
                                  output_size=output_size,
                                  activation=activation,
                                  horizon=horizon)

    def forward(self, x, edge_index, edge_weight=None, edge_features=None, u=None):
        """"""
        # x: [batches steps nodes features]
        x = utils.maybe_cat_exog(x, u)

        # flat time dimension
        x = self.input_encoder(x)
        if self.edge_encoder is not None:
            assert edge_weight is None
            edge_weight = self.edge_encoder(edge_features)

        out = x
        for layer in self.gcn_layers:
            out = layer(out, edge_index, edge_weight)

        out = out + self.skip_con(x)

        return self.readout(out)

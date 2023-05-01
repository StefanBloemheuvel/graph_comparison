#%%

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='Which dataset to choose', choices=['la','bay','air'])
parser.add_argument('algorithm', help='Which algorithm to use', choices=['gabriel','kmeans','optics','knn_weighted','knn_unweighted','minmax','relative_neighborhood','gaussian','dtw','mic','correlation'])
parser.add_argument('gnn_model', help='Which model to choose', choices=['timethenspace','dcrnn'])

args = parser.parse_args()
# args = parser.parse_args(args=[])
print(args.dataset)
print(args.algorithm)
print(args.gnn_model)

print()

#%%
import torch
import pandas as pd 
from pytorch_lightning.loggers import TensorBoardLogger
import datetime as dt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from tsl.nn.blocks.encoders import RNN
from tsl.nn.blocks.decoders import GCNDecoder
from all_models import adj_to_edge_index
from typing import Optional

from einops import rearrange
from torch import nn, Tensor
from torch_geometric.typing import Adj, OptTensor

from tsl.nn.blocks.decoders.mlp_decoder import MLPDecoder
from tsl.nn.blocks.encoders import ConditionalBlock
from tsl.nn.blocks.encoders.dcrnn import DCRNN
from tsl.ops.connectivity import edge_index_to_adj

import datetime

def print_time():
    parser = datetime.datetime.now() 
    return parser.strftime("%d-%m-%Y %H:%M:%S") 

from all_models import *
from tsl.metrics.torch import MaskedMAE, MaskedMAPE, MaskedMSE, MaskedMSE
from tsl import logger
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import MetrLA, PemsBay, AirQuality
from tsl.datasets.pems_benchmarks import PeMS03, PeMS04, PeMS07, PeMS08
from tsl.experiment import Experiment
from tsl.engines import Predictor
from tsl.metrics import torch as torch_metrics, numpy as numpy_metrics
from tsl.nn import models
from tsl.utils.casting import torch_to_numpy
from tsl.datasets.pems_benchmarks import PeMS03, PeMS04, PeMS07, PeMS08
import pandas as pd
import networkx as nx
from typing import Optional, Tuple, Union, List
from tsl.typing import TensArray, OptTensArray, SparseTensArray, DataArray, ScipySparseMatrix
from types import ModuleType
from torch_sparse import SparseTensor, fill_diag
import torch_sparse
from torch import Tensor
import numpy as np 
import random
from geopy.distance import geodesic
import os 

from geoconnector.graph_maker import graph_maker_function
from geoconnector.newest_graph_maker import graph_generator


from pytorch_lightning import seed_everything



import sys
if args.algorithm == 'gabriel':
    options_list = [0]
if args.algorithm == 'relative_neighborhood':
    options_list = [0]
if args.algorithm == 'kmeans':
    options_list = [i for i in range(16,40)]
if args.algorithm == 'optics':
    options_list = [i for i in range(16,40)]
if args.algorithm == 'knn_weighted':
    options_list = [i for i in range(16,40)]
if args.algorithm == 'knn_unweighted':
    options_list = [i for i in range(16,40)]
# if args.algorithm == 'dbscan':
    # options_list = [i for i in range(2,16)]
if args.algorithm == 'gaussian':
    options_list = [round(i,2) for i in np.arange(0.05,0.95,0.05)]
if args.algorithm == 'minmax':
    options_list = [round(i,2) for i in np.arange(0.05,0.95,0.05)]
# if args.algorithm == 'correlation_top' or args.algorithm == 'correlation_new':
    # options_list = [round(i,2) for i in np.arange(0.1,0.7,0.1)]
if args.algorithm == 'dtw':
    options_list = [round(i,2) for i in np.arange(0.05,0.95,0.05)]
if args.algorithm == 'mic':
    options_list = [round(i,2) for i in np.arange(0.05,0.95,0.05)]
if args.algorithm == 'correlation':
    options_list = [round(i,2) for i in np.arange(0.05,0.95,0.05)]
print(f'went for {args.dataset} and {args.algorithm} \n')
print(f' options are = {options_list}')


graph_generator_obj = graph_generator()

for i in options_list:
    seed = 0
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    np.random.permutation(seed)
    os.environ['PYTHONHASHSEED']=str(seed)# 2. Set `python` built-in pseudo-random generator at a fixed value
    np.random.RandomState(seed)


    #pip install tensorflow-determinism needed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seed_everything(seed)


    print('came here?')
    if args.dataset == 'bay':
        dataset = PemsBay()
    if args.dataset == 'la':
        dataset = MetrLA()
    if args.dataset == 'air':
        dataset = AirQuality(small=True)
        
    print(f'current combination = {i} with {args.dataset} and {args.algorithm}')
    graph_generator_obj = graph_generator()
    if args.algorithm == 'gabriel':
        graph_generator_obj.gabriel(f'sensor_locations/sensor_locations_{args.dataset}.csv')
    # if args.algorithm == 'kmeans_own':
        # graph_generator_obj.kmeans_own(f'sensor_locations/sensor_locations_{args.dataset}.csv',num_clusters=i)
    if args.algorithm == 'optics':
        graph_generator_obj.optics(f'sensor_locations/sensor_locations_{args.dataset}.csv', min_samples=i)
    if args.algorithm == 'knn_weighted':
        graph_generator_obj.knn_weighted(f'sensor_locations/sensor_locations_{args.dataset}.csv', k=i)
    if args.algorithm == 'knn_unweighted':
        graph_generator_obj.knn_unweighted(f'sensor_locations/sensor_locations_{args.dataset}.csv', k=i)
    if args.algorithm == 'minmax':
        graph_generator_obj.minmax(f'sensor_locations/sensor_locations_{args.dataset}.csv', cutoff=i)
    # if args.algorithm == 'dbscan':
        # graph_generator_obj.dbscan(f'sensor_locations/sensor_locations_{args.dataset}.csv', eps=1, min_samples=i)
    if args.algorithm == 'gaussian':
        graph_generator_obj.gaussian(f'sensor_locations/sensor_locations_{args.dataset}.csv', normalized_k=i)
    # if args.algorithm == 'gaussian':
        # graph_generator_obj.gaussian(f'sensor_locations/sensor_locations_{args.dataset}.csv')
    if args.algorithm == 'relative_neighborhood':
        graph_generator_obj.relative_neighborhood(f'sensor_locations/sensor_locations_{args.dataset}.csv')
    if args.algorithm == 'kmeans':
        graph_generator_obj.kmeans(f'sensor_locations/sensor_locations_{args.dataset}.csv', num_clusters=i)
    # if args.algorithm == 'correlation_top':
        # graph_generator_obj.correlation_top(f'sensor_locations/sensor_locations_{args.dataset}.csv', f'sensor_locations/inputs_{args.dataset}.npy',threshold=i)
    # if args.algorithm == 'correlation_new':
        # graph_generator_obj.correlation_new(f'sensor_locations/sensor_locations_{args.dataset}.csv', f'sensor_locations/inputs_{args.dataset}.npy',threshold=i)
    # if args.algorithm == 'dtw':
        # graph_generator_obj.dtw(f'sensor_locations/sensor_locations_{args.dataset}.csv', f'sensor_locations/inputs_{args.dataset}.npy',clips=True,threshold=i)
    
    if args.algorithm == 'dtw' or args.algorithm == 'correlation' or args.algorithm == 'mic' and args.dataset in ['la','bay','air']:
        print(f'went for clips')
        graph_generator_obj.from_signal(f'sensor_locations/sensor_locations_{args.dataset}.csv', f'sensor_locations/inputs_{args.dataset}.npy', variant=args.algorithm, clips=True,threshold=i)
    
    graph_generator_obj.create_adjacency_matrix(fill_diagonal = True)
    graph_generator_obj.summary_statistics()
    
    # graph_generator_obj.create_normalized_laplacian_matrix()
    # graph_generator_obj.summary_statistics()
    
    # if graph_generator_obj.number_of_edges == 0:
        # print('no edges so no solution')
        # print('\n' * 5)
        # continue
    print(f'look here steef')
    print(graph_generator_obj.data.shape[0])
    print(nx.number_of_edges(graph_generator_obj.networkx_graph))

    if graph_generator_obj.number_of_edges == 0:
        print('no edges so no solution')
        print('\n' * 5)
        continue
    
    if graph_generator_obj.data.shape[0] >= graph_generator_obj.number_of_edges:
        print('only self edges')
        print('\n' * 5)
        continue
    else:
        print(f'number of edges is > {graph_generator_obj.data.shape[0]}, namely {graph_generator_obj.number_of_edges}')
        print('\n' * 5)
        
    # adj2 = graph_generator_obj.normalized_laplacian_matrix

    print(f"Sampling period: {dataset.freq}\n"
        f"Has missing values: {dataset.has_mask}\n"
        #   f"Percentage of missing values: {(1 - dataset.mask.mean()) * 100:.2f}%\n"
        f"Has dataset exogenous variables: {dataset.has_covariates}\n"
        f"Relevant attributes: {', '.join(dataset.attributes.keys())}")


    adj2 = graph_generator_obj.adjacency_matrix

    print(graph_generator_obj.networkx_graph)

    print(adj2)
    print(type(adj2))

    adj2 = adj_to_edge_index(adj2)
    print(adj2)
    covariates = {'u': dataset.datetime_encoded('day').values}

    target, idx = dataset.numpy(return_idx=True)
    print()

    connectivity = dataset.get_connectivity(threshold=0.1,
                                        include_self=False,
                                        normalize_axis=1,
                                        layout="edge_index")
    
    perform_classic = True 
    if perform_classic == True:
        print('went for classic gaussian with extra info version')
        connectivity = dataset.get_connectivity(threshold=0.05,
                                        include_self=True,
                                        normalize_axis=1,
                                        layout="edge_index")
        edge_index, edge_weight = connectivity

        print(f'edge_index {edge_index.shape}:\n', edge_index)
        print(f'edge_weight {edge_weight.shape}:\n', edge_weight)
        adj = edge_index_to_adj(edge_index, edge_weight)
        print(f'A {adj.shape}:')
        torch_dataset = SpatioTemporalDataset(target=target,
                                        index=idx,
                                        connectivity=connectivity,
                                        mask=dataset.mask,
                                        horizon=12,
                                        window=12,
                                        stride=1)
    else:
        print('went for experiment version')
        torch_dataset = SpatioTemporalDataset(target=target,
                                            index=idx,
                                            connectivity=adj2,
                                            mask=dataset.mask,
                                            horizon=12,
                                            window=12,
                                            stride=1)

    print(torch_dataset)

    scalers = {'target': StandardScaler(axis=(0, 1))}
    splitter = dataset.get_splitter(val_len=0.1, test_len=0.2)

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=64,
    )
    dm.setup()
    print(dm)


    model_kwargs_timethenspace = dict(n_nodes=torch_dataset.n_nodes,
                            input_size=torch_dataset.n_channels,
                            output_size=torch_dataset.n_channels,
                            horizon=torch_dataset.horizon)


    # print(model_kwargs_timethenspace)

    loss_fn = MaskedMAE()

    metrics = {'mae': MaskedMAE(),
            'mape': MaskedMAPE(),
            'mse': MaskedMSE(),
            'mape': MaskedMAPE(),
            # 'mae_at_15': MaskedMAE(at=2),  # `2` indicated the third time step, which correspond to 15 minutes ahead
            # 'mae_at_30': MaskedMAE(at=5),
            # 'mae_at_60': MaskedMAE(at=11), 
            }

    model_kwargs_timethenspace = {
        'input_size': dm.n_channels,  # 1 channel
        'horizon': dm.horizon,  # 12, the number of steps ahead to forecast
        'hidden_size': 16,
        'rnn_layers': 1,
        'gcn_layers': 2
    }

    model_kwargs_dcrnn = {
        'input_size': dm.n_channels,  # 1 channel
        'horizon': dm.horizon,  # 12, the number of steps ahead to forecast
        'hidden_size': 16,
        'output_size':1
    }

    if args.gnn_model == 'timethenspace':
        max_epochs = 100
        predictor = Predictor(
            model_class=TimeThenSpaceModel,
            model_kwargs=model_kwargs_timethenspace,
            optim_class=torch.optim.Adam,
            optim_kwargs={'lr': 0.003},
            loss_fn=loss_fn,
            metrics=metrics
        )
    if args.gnn_model == 'dcrnn':
        max_epochs = 50
        predictor = Predictor(
            model_class=DCRNNModel_manual,
            model_kwargs=model_kwargs_dcrnn,
            optim_class=torch.optim.Adam,
            optim_kwargs={'lr': 0.003},
            loss_fn=loss_fn,
            metrics=metrics
        )

    # print()
    # print('look here')
    # print()
    # print()
    # sample = torch_dataset[0]
    # print(sample)
    # print(torch_dataset[0].pattern)
    

    # print(dir(TimeThenSpaceModel))
    # print(predictor)
    # summary(predictor, input_size=(64, 12, 207, 1))

    # from torchsummary import summary


    # summary(model, (1, 28, 28))
    # summary_model = TimeThenSpaceModel(input_size=1, hidden_size=16, rnn_layers=1,gcn_layers=2,horizon=12)
    # print('look at this =')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    # print(device)
    # model = summary_model.to(device)
    # summary(model, (64, 12,207,1))
    # summary(summary_model,(64,12, 207, 1), device='cpu')
    # print('now try the following')
    # from torchinfo import summary
    # summary(summary_model,input_size=(12, 207, 1))
    

    # summary(summary_model, input_size=(64, 12, 207,1))

    # logger = TensorBoardLogger(save_dir="logs", name="tsl_intro", version=0)
    

    checkpoint_callback = ModelCheckpoint(
        dirpath='logs',
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )
    
    max_epochs = 100
    trainer = pl.Trainer(max_epochs=max_epochs,
                        # logger=logger,
                        gpus=1 if torch.cuda.is_available() else None,
                        limit_train_batches=100,
                        callbacks=[checkpoint_callback],
                        enable_model_summary=True)
    print('look above')
    
    trainer.fit(predictor, datamodule=dm)
    predictor.load_model(checkpoint_callback.best_model_path)
    predictor.freeze()

    performance = trainer.test(predictor, datamodule=dm)
    output = trainer.predict(predictor, dataloaders=dm.val_dataloader())

    
    print('done ')
    print(output['y'].shape)

    df = pd.DataFrame(performance)
    print(df)

    del trainer
    # df.insert(0,'dataset_name',args.dataset)
    # df.insert(1,'graph_name',args.algorithm)
    # df.insert(0,'time',dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # df.insert(3,'epochs',max_epochs)
    # df.insert(4,'model',args.gnn_model)
    # df.round(7).to_csv('results.csv', mode='a', index=False, header=False)
    with open(f"epoch100.csv", "a") as text_file:
        print(f'{print_time()},{args.dataset},{args.algorithm},{i},{df["test_mse"].item():.5f},{df["test_mae"].item():.5f},{df["test_mape"].item():.5f},{graph_generator_obj.number_of_edges}', file=text_file)

#!/usr/bin/env python
# coding: utf-8

# script to check the amount of time with evaluator 

import pykeen
from pykeen.pipeline import pipeline
import networkx as nx
import pathlib
from random import sample
import pandas as pd
from pykeen.triples import TriplesFactory
import torch
import numpy as np
import time
import wandb

def preprocess_graph_heterogeneous(graph: nx.Graph):
    edge_types = []
    for u, v, data in graph.edges.data():
        edge_types.append(data["edge_type"])

    #edge_codes, edge_types = pd.factorize(edge_types)
    
    node_types = []
    for n, data in graph.nodes.data():
        node_types.append(data["node_type"])

    node_codes, node_types = pd.factorize(node_types)

    preprocessed_graph = nx.DiGraph()
    preprocessed_graph.add_nodes_from(graph.nodes())

    preprocessed_graph.node_codes = node_codes
    preprocessed_graph.node_types = node_types
    # ---- #
    edge_codes, num_counts = np.unique(edge_types, return_counts=True)
    fraction = num_counts / len(edge_types)
    valid_edge_codes = edge_codes[fraction >= 0.001]
    print(len(valid_edge_codes), valid_edge_codes)
    
    for (u,v,w), t in zip(graph.edges.data("edge_weight"), edge_types):
        assert w is not None
        
        # only most relevant relations
        if t in valid_edge_codes and w >=1.0:
            preprocessed_graph.add_edge(u, v, edge_weight=w, edge_code=t)
        
        else : continue
    # ---- #    
    
    preprocessed_graph.edge_types = edge_types

    return preprocessed_graph

# 1. Start a WB run
wandb.init(project="PYKEEN Test", entity="cardiors")

t0 = time.time()

#loading the full graph
print('loading graph ...')
base_path = pathlib.Path(
    "/data/analysis/ag-reils/ag-reils-shared/cardioRS/data/2_datasets_pre/211110_anewbeginning")
G = nx.readwrite.gpickle.read_gpickle('/data/analysis/ag-reils/ag-reils-shared/cardioRS/data/2_datasets_pre/211110_anewbeginning/graph_full_211122.p')

# -------- directly create TRIPLES ------- #
print('preprocessing ...')
SG = preprocess_graph_heterogeneous(G)
print('done ...')

print('generating triplesArray ...')
tripleList=[]
nodes=[]
for u,v,data in SG.edges.data():
    l=[]
    l.append(u)
    nodes.append(u)
    l.append(data['edge_code'])
    l.append(v)
    nodes.append(v)
    tripleList.append(l)

#needs triples as ndarray - shape (n,3), dtype:str 
tripleArray=np.array(tripleList, dtype=str)
print(len(tripleArray))
print('done ...')

#####################
# Easy Pipeline Way #
####################
print('create Triples Factory ...')
tf2 = TriplesFactory.from_labeled_triples(tripleArray, create_inverse_triples=True)

print(tf2.get_most_frequent_relations(3), '\n', 'triple Array: ', len(tripleArray))
# ----------- Splitting into training, testing, validation ---#

training_factory, testing_factory, validation_factory = tf2.split([0.9,0.05,0.05])


# --

# Pick a model
from pykeen.models import TransE

kwargs={'triples_factory': training_factory , 'loss': None, 'predict_with_sigmoid':False, 'preferred_device':None, 'random_seed':None}

tf2_model = TransE(**kwargs, embedding_dim=64) # >64, 256


results = pipeline(
    training = training_factory,
    testing = testing_factory,
    validation = validation_factory,
    loss='marginranking',
    loss_kwargs=dict(margin=1),
    model = tf2_model,
    epochs = 100, 
    training_loop = 'sLCWA',
    negative_sampler = 'basic',
    evaluator = 'rankbased',
    result_tracker='wandb',
    result_tracker_kwargs=dict(
        project='PYKEEN Test',
        experiment='experiment-1'),
    stopper = 'early',
    stopper_kwargs=dict(frequency=10, patience=4, relative_delta=0.002) #delta: The minimum relative improvement necessary to consider it an improved result
    
)

t1 = time.time()
results.plot_losses()
titel= 'Standart_earlyStop_Time' + str(t1-t0)
results.save_to_directory(title)
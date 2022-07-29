#!/usr/bin/env python
# coding: utf-8

# In[12]:


# Start Run of whole Graph on the GPU
# maybe delete the graph for memory effiziency , but keep list of nodes 


import pykeen
from pykeen.pipeline import pipeline
import networkx as nx
import pathlib
from random import sample
import pandas as pd
from pykeen.triples import TriplesFactory
import torch
import numpy as np


# In[6]:


# from https://github.com/nebw/ehrgraphs/blob/master/ehrgraphs/data/data.py#L82-L117
def preprocess_graph_heterogeneous(graph: nx.Graph):
    edge_types = []
    for u, v, data in graph.edges.data():
        edge_types.append(data["edge_type"])

    edge_codes, edge_types = pd.factorize(edge_types)

    node_types = []
    for n, data in graph.nodes.data():
        node_types.append(data["node_type"])

    node_codes, node_types = pd.factorize(node_types)

    preprocessed_graph = nx.DiGraph()
    preprocessed_graph.add_nodes_from(graph.nodes())

    preprocessed_graph.node_codes = node_codes
    preprocessed_graph.node_types = node_types
        
    # drop shortcut edges
    exclude_codes = []
    exclude_codes.append(edge_codes[list(edge_types).index("Subsumes")])
    exclude_codes.append(edge_codes[list(edge_types).index("Is a")])

    for (u, v, w), c in zip(graph.edges.data("edge_weight"), edge_codes):
        assert w is not None

        # drop shortcut edges
        if c in exclude_codes and w < 1.0:
            continue

        preprocessed_graph.add_edge(u, v, edge_weight=w, edge_code=c)

    preprocessed_graph.edge_types = edge_types

    return preprocessed_graph


# In[3]:


#loading the full graph
print('loading graph ...')
base_path = pathlib.Path(
    "/data/analysis/ag-reils/ag-reils-shared/cardioRS/data/2_datasets_pre/211110_anewbeginning")
G = nx.readwrite.gpickle.read_gpickle('/data/analysis/ag-reils/ag-reils-shared/cardioRS/data/2_datasets_pre/211110_anewbeginning/graph_full_211122.p')

# building preprocessed ego graph


# In[7]:


# -------- directly create TRIPLES ------- #
print('preprocessing ... ')
SG = preprocess_graph_heterogeneous(G)

print('building triple Array ...')
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
tripleArray=np.array(sample(tripleList, 100), dtype=str)

# delete NETWROKX Graph
SG.clear()
del SG

# In[8]:





# In[10]:


# ----------- directly loading Triples into PyKEEN ------ #
print('loading triples into pykeen ... ')
tf2 = TriplesFactory.from_labeled_triples(tripleArray, create_inverse_triples=True)
print('done')
# ----------- Training without evaluation ------------ #

# Pick a model
from pykeen.models import TransE

kwargs={'triples_factory': tf2, 'loss': None, 'predict_with_sigmoid':False, 'preferred_device':None, 'random_seed':420}

tf2_model = TransE(**kwargs, embedding_dim=10) # >64, 256

# Pick an optimizer from Torch
from torch.optim import Adam

optimizer = Adam(params=tf2_model.get_grad_params())

# Pick a training approach (sLCWA or LCWA)
from pykeen.training import SLCWATrainingLoop

training_loop = SLCWATrainingLoop(

    model=tf2_model,

    triples_factory=tf2,

    optimizer=optimizer,

)

# Train like Cristiano Ronaldo

_ = training_loop.train(

    triples_factory=tf2,

    num_epochs=10, # ! ! !

    batch_size=256,
    # result_tracker='wandb'
    # result_tracker_kwargs=

)


# In[ ]:


# class RepresentationModule
entity_RepModel = tf2_model.entity_representations[0] # check for more representations

# filter out duplicate nodes
nodes = list(dict.fromkeys(nodes))


# make all entities to ids
embedding_dict={}
all_entities = tf2.entities_to_ids(nodes)

embeddings = entity_RepModel(torch.tensor(all_entities, dtype=torch.int)) # cast list elements into tensors

embedding_dict = dict(zip(nodes, embeddings2.detach().numpy() )) #  Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.


# In[ ]:


# SAVE INTO DATAFRAME 
items = embedding_dict.items()

data_list = list(items)

df = pd.DataFrame(data_list)
df.to_feather('/home/tilingl/GNN/Pykeen')

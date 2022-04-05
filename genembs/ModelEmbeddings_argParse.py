#!/usr/bin/env python
# coding: utf-8

# First version in genembs Module, clean code without testing code-parts
# TODO: structure path to directorys loading and saving
# TODO: import all Losses and Models that will be used

# script to choose a model to test and directly generate embeddings
# use selfmade pipeline from SaveNreload scripts and 
# use Get_embedding script to pull out the embeddings out of the module
# needs 'connected_nodes_list.feather' with non-isolated nodes in it and 'triple_list_211209.feather'

# important: UPDATE of PYKEEN is avaiable

import pykeen
import networkx as nx
import pathlib
import pandas as pd
import numpy as np

from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import torch
import argparse

#graph_path='/sc-projects/sc-proj-ukb-cvd/data/2_datasets_pre/211110_anewbeginning/artifacts/graph_full_211209.p'
#preprocessed_graph = preprocess_graph_heterogeneous(G)
#G = nx.readwrite.gpickle.read_gpickle(graph_path)
# function to drop schortcut edges from:https://github.com/nebw/ehrgraphs/blob/master/ehrgraphs/data/data.py#L82-L117
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

        omop_exclude_codes = []
        ot_exclude_codes = []
        #drop shortcut edges =True
        omop_exclude_codes.append(edge_codes[list(edge_types).index("Subsumes")])
        omop_exclude_codes.append(edge_codes[list(edge_types).index("Is a")])
        # TODO: multiple indices?
        ot_exclude_codes.append(edge_codes[list(edge_types).index("Is descendant of")])
        ot_exclude_codes.append(edge_codes[list(edge_types).index("Is ancestor of")])

        for (u, v, w), c in zip(graph.edges.data("edge_weight"), edge_codes):
            assert w is not None

            # drop shortcut edges
            if c in omop_exclude_codes and w < 1.0:
                continue

            if c in ot_exclude_codes:
                continue

            preprocessed_graph.add_edge(u, v, edge_weight=w, edge_code=c)

        #self.edge_types = edge_types

        return preprocessed_graph



# generates a list of nodes excluding isolated nodes out of an preprocessed networkx graph
# node_list_name should contain .feather ending
def get_connected_nodes(preprocessed_graph: nx.Graph, networkxnode_list_name: str):
    # get isolated nodes and remove them
    print(len(list(nx.isolates(preprocessed_graph))), len(preprocessed_graph.nodes()))
    preprocessed_graph.remove_nodes_from(list(nx.isolates(preprocessed_graph)))
    conected_nodes = list(preprocessed_graph.nodes())
    node_df_list =pd.DataFrame(conected_nodes,  columns=['nodes'])
    node_df_list.to_feather('connected_nodes_list.feather')



# funktion to create the triple_list containing the node-relation-node triple from the latest graph
# title to name triple list
def create_triple_list(title, preprocessed_graph):  
    
    #create Triple List
    tripleList=[]
    for u,v,data in preprocessed_graph.edges.data():
        l=[]
        l.append(u)
        l.append(data['edge_code'])
        l.append(v)
        tripleList.append(l)

    #needs triples as ndarray - shape (n,3), dtype:str 
    tripleArray=np.array(tripleList, dtype=str)
    print(type(tripleArray), tripleArray.shape, tripleArray.dtype)
    
    #save TripleList as feather file
    p_title = '/home/tilingl/Pykeen/Triple_Lists/'+ title +'.feather'
    # triple array to pandas df
    df = pd.DataFrame(tripleArray, columns=['node1', 'relation_code','node2'])
    df.head()
    df.tail()
    df.to_feather(p_title)
    print(p_title)


# function to start the embedding training with all variables and parameters
def start_emb_train(triple_path, inverse, Model, emb_dim, Training_loop, check_name: str, emb_dict_name: str,  Loss=None ):
    # generate Triples Factory
    tripleArray = pd.read_feather(triple_path).to_numpy()
    print('length of the triple array: ', len(tripleArray), type(tripleArray))
    tf = TriplesFactory.from_labeled_triples(tripleArray, create_inverse_triples=inverse)
    
    print('loading TriplesFactory done ... ', type(tf))
    print(tf.num_entities, tf.num_relations)    # 511291 338 old= 511291 326
    
    #pick a Model that was imported
    #choose a loss Class that was imported, default =None
    kwargs={'triples_factory': tf, 'loss': Loss, 'predict_with_sigmoid':False, 'preferred_device':None}
    model = Model(**kwargs, embedding_dim=emb_dim, random_seed=420)
    
    # Pick an optimizer from Torch
    from torch.optim import Adam
    optimizer = Adam(params=model.get_grad_params())
    
    # Pick a training approach that was imported !! contains the losses, choose between SLCWATrainingLoop and LCWATrainingLoop
    training_loop = Training_loop(

        model=model,

        triples_factory=tf,

        optimizer=optimizer,
    )
    
    # just run for one epoch, evaluate losses and restart training where it was left
    for i in range(1,30):
        # Train like Cristiano Ronaldo
        losses = training_loop.train(
            triples_factory=tf,
            num_epochs=i,
            batch_size=256,
            checkpoint_name= check_name, # for example TransE_t2.pt
            checkpoint_frequency=0

        )
        if i>1 and (losses[-2] - losses[-1]) < 1e-7: # changed 1e-6 to 1e-7
            
            #TODO: function to obtain embeddings
            #TODO: speak about normalisation
            #try to switch model to cpu device:
            device = torch.device('cpu')
            model.to(device)
            #tf.to(device)
            #do not need anymore: entity_RepModel = model.entity_representations[0] # check for more representations
            try:
                print(model.entity_representations[1])
            except IndexError:
                print('No more entity_reps in this model ')
            # acces entity_values, mapp them to list and pass the list to the entity_repModel to recieve Embeddings. Next, create embedding_dict and transform to DataFrame
            nodes = pd.read_feather('data/connected_nodes_list.feather')
            # casting node Names from list into equivalent ids(indices) as a torch.LongTensor on CPU --> bc model is also cpu
            entity_ids = torch.as_tensor(tf.entities_to_ids(nodes['nodes']), dtype=torch.long, device=device) # .view()?
            
            #all embeddings as a numpy.ndarray, indices as torch.LongTensor
            entity_embeddings = model.entity_representations[0](indices=entity_ids).detach().numpy() # detach from GPU 
            
            # do not need anymore : embeddings = entity_RepModel(entity_ids) 
            df_dict = pd.DataFrame(dict(nodes= nodes['nodes'], embeddings= list(entity_embeddings)))
            print(df_dict.head())
            # save embedding dict
            emb_path = '/home/tilingl/Pykeen/New_Embedding_Stuff/Embeddings/Embedding_dict_' + emb_dict_name + '.feather'
            df_dict.to_feather(emb_path)
            
            #end the loop
            break


# reminder how to use the function
'''
from pykeen.models import TransE
from pykeen.training import SLCWATrainingLoop 
triple_path = '/home/tilingl/Pykeen/Triple_Lists/triple_list_211209.feather'
             # triple_path, inverse, Model, emb_dim, Training_loop, check_name: str, embedding_dict Name: str,  Loss=None
start_emb_train(triple_path, True, TransE, 256, SLCWATrainingLoop, 'TransE_default.pt', 'TransE', Loss=None)
'''
# argparser for all hyperparameter Options
parser = argparse.ArgumentParser()
parser.add_argument('triple_path', help="declare the path to the triple.feather file", type=str)
parser.add_argument('inverse', help= "add True/False wether inverted triple should created", type=bool)
parser.add_argument('emb_dimension', help="specify embedding dimension" , type=int)
parser.add_argument('check_Name.pt', help=" choose name for the checkpoint.pt", type=str)
parser.add_argument('emb_dict_name', help="choose name for the Embedding_dictionary", type=str)
# add optional arguments
parser.add_argument('-l','--Loss', help="specify Loss class , default None", type=str)

args = parser.parse_args() # returns data from the options specified
print(args.inverse)
if args.Loss is not None:
    l = args.Loss
    print(type(l))
    #from pykeen.losses import l #NSSALoss  error


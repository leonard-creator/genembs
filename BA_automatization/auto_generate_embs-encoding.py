#!/usr/bin/env python
# coding: utf-8


# script to train standard models, extract and save embeddings via Hyperparameter argparser input
# should enable a better automatic workflow for starting multiple runs with only one script
# using Pykeen==1.8.0 (slightly different functions to pykeen==1.7.0)
# needs triple_list_graph_full_v3.feather for faster loading of the "graph"
# --> /sc-projects/sc-proj-ukb-cvd/data/2_datasets_pre/220208_graphembeddings/ triples


# TODO:

import pykeen
import pandas as pd
import numpy as np
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.constants import PYKEEN_CHECKPOINTS
import torch
import argparse
from pykeen.regularizers import Regularizer, regularizer_resolver
from pykeen.utils import resolve_device

#import all models and parameter
from pykeen.models  import ConvE, TransE, ComplEx, MuRE, RotatE, TuckER, DistMult, RESCAL
from pykeen.training import LCWATrainingLoop, SLCWATrainingLoop
from pykeen.losses import BCEWithLogitsLoss, SoftplusLoss, NSSALoss, SoftMarginRankingLoss, PairwiseLogisticLoss

# testing:
from pykeen.datasets import Nations

# helper function to return the classes and not argparsed strings
def help_inverse(t):
    if t =="True":
        return True
    if t =="False":
        return False
    else:
        raise TypeError("expected True or False, got:" , t)

def help_model(m_str):
    if m_str == "ConvE":
        return ConvE
    if m_str == "TransE":
        return TransE
    if m_str == "ComplEx":
        return ComplEx
    if m_str == "MuRE":
        return MuRE
    if m_str == "RotatE":
        return RotatE
    if m_str == "TuckER":
        return TuckER
    if m_str == "DistMult":
        return DistMult
    if m_str == "RESCAL":
        return RESCAL
    else:
        raise TypeError("unknown, add Model to help_model func!")

def help_loss(l_str):
    if l_str is None or l_str == "None":
        return None
    if l_str == "SoftplusLoss":
        return SoftplusLoss
    if l_str == "BCEWithLogitsLoss":
        return BCEWithLogitsLoss
    if l_str == "NSSALoss":
        return NSSALoss
    if l_str == "SoftMarginRankingLoss":
        return SoftMarginRankingLoss
    if l_str == "PairwiseLogisticLoss":
        return PairwiseLogisticLoss
    else:
         raise TypeError("unknown, add loss to help_loss func!")

def help_loop(loop_str):
    if loop_str == "SLCWATrainingLoop":
        return SLCWATrainingLoop
    else:
        return LCWATrainingLoop
    


# function to start the embedding training with all variables and parameters
def start_emb_train(triple_path, inverse, Model,
                    emb_dim, Training_loop,
                    check_name: str, emb_dict_name: str,
                    batch_s=None, sub_batch=None, 
                    slice_s=None, Loss=None,):
    
    # make sure, that gpu is avaiable and chosen 
    device = 'gpu' 
    _device: torch.device = resolve_device(device)
    print(f"Using device: {device}", type(_device))
    
    # generate Triples Factory
    tripleArray = pd.read_feather(triple_path).to_numpy()
    print('length of the triple array: ', len(tripleArray), type(tripleArray))
    tf = TriplesFactory.from_labeled_triples(tripleArray, create_inverse_triples=inverse)
    
    print('loading TriplesFactory done ... ', type(tf))
    print(tf.num_entities, tf.num_relations)    #700380 404 old=511291 338 oldest= 511291 326
    
    #pick a Model that was imported
    #choose a loss Class that was imported, default =None
    kwargs={'triples_factory': tf, 'loss': Loss, 'predict_with_sigmoid':False}
    model = Model(**kwargs, embedding_dim=emb_dim, random_seed=420)
    model= model.to(_device) # important otherwise fall back to cpu
    
    # Pick an optimizer from Torch
    from torch.optim import Adam
    optimizer = Adam(params=model.get_grad_params())
    
    # Pick a training approach that was imported !! contains the losses, choose between SLCWATrainingLoop and LCWATrainingLoop
    training_loop = Training_loop(

        model=model,
        triples_factory=tf,
        optimizer=optimizer,
        automatic_memory_optimization=True, 
    )
    
    
    # just run for one epoch, evaluate losses and restart training where it was left
    for i in range(1,100):
        if i >1:
            #make shure the loaded checkpoint is has the right mapping:
            checkpoint = torch.load(PYKEEN_CHECKPOINTS.joinpath('/sc-scratch/sc-scratch-ukb-cvd/checkpoints_pykeen_leonard/'+check_name))
            tf = TriplesFactory.from_labeled_triples(
                triples =tripleArray,
                create_inverse_triples=inverse,
                entity_to_id=checkpoint['entity_to_id_dict'],
                relation_to_id=checkpoint['relation_to_id_dict'],
                )
            print("check done!")
        # Train like Cristiano Ronaldo
        losses = training_loop.train(
            triples_factory=tf,
            num_epochs=i,
            batch_size=batch_s, #256, # if None -> automatic search for the best and greatest
            checkpoint_name= check_name, # for example TransE_t2.pt
            checkpoint_frequency=0,
            checkpoint_directory='/sc-scratch/sc-scratch-ukb-cvd/checkpoints_pykeen_leonard', # new checkpoint dir bc of massive storage needs
            
            sub_batch_size=sub_batch, # not for SLCWA and not supported bc of batch normalization!!
            slice_size=slice_s, # not for SLCWA

        )
        if i>1 and (losses[-2] - losses[-1]) < 1e-7: # changed 1e-6 to 1e-7
            
            #TODO: function to obtain embeddings
            #switch model back to cpu device:
            _device = torch.device('cpu') # 
            model.to(_device)
            #do not need anymore: entity_RepModel = model.entity_representations[0] # check for more representations
            try:
                print(model.entity_representations[1])
            except IndexError:
                print('\n','Index Error, no more entity_reps ', '\n')
            # acces entity_values, mapp them to list and pass the list to the entity_repModel to recieve Embeddings. Next, create embedding_dict and transform to DataFrame
            ##nodes = pd.read_feather('/sc-projects/sc-proj-ukb-cvd/data/2_datasets_pre/220208_graphembeddings/metadata/connected_nodes_list.feather')
            
            #print(tf.entity_to_id, type(tf.entity_to_id),'e_ids', tf.entity_ids, tf.entity_to_id.keys())
            #
            entity_id_dict = tf.entity_to_id
            entity_ids_as_tensors = torch.as_tensor([int(v) for v in entity_id_dict.values()], dtype=torch.long, device=_device)
            
            # casting node Names from list into equivalent ids(indices) as a torch.LongTensor on CPU --> bc model is also cpu
            ##entity_ids = torch.as_tensor(tf.entities_to_ids(nodes['nodes']), dtype=torch.long, device=_device) # .view()?
            
            #all embeddings as a numpy.ndarray, indices as torch.LongTensor
            entity_embeddings = model.entity_representations[0](indices=entity_ids_as_tensors).detach().numpy() # detach tensor 
            
            # do not need anymore : embeddings = entity_RepModel(entity_ids) 
            df_dict = pd.DataFrame(dict(nodes= entity_id_dict.keys(), embeddings=list(entity_embeddings)))
            print(df_dict.head())
            # save embedding dict
            emb_path = '/sc-projects/sc-proj-ukb-cvd/data/2_datasets_pre/220208_graphembeddings/embeddings_leonard/Embedding_dict_' + emb_dict_name + '.feather'
            print('saved in: ',emb_path)
            df_dict.to_feather(emb_path)
          
            #end the loop
            break
            
            



# argparser for all hyperparameter Options
parser = argparse.ArgumentParser()
parser.add_argument('inverse', help= "add True/False wether inverted triple should created", type=str)
parser.add_argument('Model', help="choose the model to train with. NOT NodePiece!", type=str)
parser.add_argument('emb_dimension', help="specify embedding dimension" , type=int)
parser.add_argument('SLCWA_LCWA', help="SLCWATrainingLoop or LCWA training Loop", type=str)
parser.add_argument('check_name', help=" choose name for the checkpoint.pt", type=str)
parser.add_argument('emb_dict_name', help="name for the Embedding_dictionary, newest= xxx_full_v3", type=str)
# add optional arguments
parser.add_argument('-b', '--batch_size', help="select batch size, if None: automatic batchsize search", type=int,  default=None)
parser.add_argument('-sb', '--sub_batch_size', help="only for LCWA, sub batch_size for effizient memory usage", type=int, default=None)
parser.add_argument('-s', '--slize_size', help="only for LCWA, divisor for slicing batches for single calculations",type=int, default=None)
parser.add_argument('-l','--Loss', help="specify Loss class , default None", type=str, default=None)

args = parser.parse_args() # returns data from the options specified

# transform strings in classes:
truefalse=help_inverse(args.inverse)
Model = help_model(args.Model)
loop = help_loop(args.SLCWA_LCWA)
Loss = help_loss(args.Loss)
triple_path = '/sc-projects/sc-proj-ukb-cvd/data/2_datasets_pre/220208_graphembeddings/triples/triple_list_graph_full_v3.feather'

print('\n','Script for creating ', args.emb_dict_name, ' embeddings', '\n')
print('Parameters: ', truefalse, Model, loop , Loss, args.emb_dimension, args.check_name, args.emb_dict_name, args.batch_size, args.sub_batch_size, args.slize_size, '\n')
#start run with passed arguments
 # sub_batching and slicing only for SLCWA
    # triple_path, inverse, Model, emb_dim, Training_loop, check_name: str, embedding_dict Name: str, batch_size, sub_batch_size, slice_size,  Loss=None
start_emb_train(triple_path, truefalse, Model, args.emb_dimension, loop, args.check_name,args.emb_dict_name, args.batch_size, args.sub_batch_size, args.slize_size, Loss)


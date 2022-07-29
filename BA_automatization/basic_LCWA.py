#!/usr/bin/env python
# coding: utf-8

# script with standard pipeline including evaluation that takes a lot of time
# test out the avaiability of LCWA without loading it back into memory that leads to memory error in all the previous attempts
# also benefit from automatic batch and subbatch search for the Model
# do not save in checkpoints bc it would reload the setting!

# start with RotatE and go on Test performance on TuckER


import pykeen
import pandas as pd
import numpy as np
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import torch
import argparse
from pykeen.regularizers import Regularizer, regularizer_resolver
from pykeen.utils import resolve_device

#import all models and parameter
from pykeen.models  import ConvE, TransE, ComplEx, MuRE, RotatE, TuckER
from pykeen.training import LCWATrainingLoop, SLCWATrainingLoop
from pykeen.losses import BCEWithLogitsLoss, SoftplusLoss, NSSALoss, SoftMarginRankingLoss, PairwiseLogisticLoss



# function to start the embedding training with all variables and parameters
def start_train(triple_path, inverse, Model,
                    emb_dim, Training_loop,
                    batch_s=None, sub_batch=None, 
                    slice_s=None, Loss=None,):    
    # generate Triples Factory
    tripleArray = pd.read_feather(triple_path).to_numpy()
    print('length of the triple array: ', len(tripleArray), type(tripleArray))
    tf = TriplesFactory.from_labeled_triples(tripleArray, create_inverse_triples=inverse)
    
    training, testing, validation = tf.split([.9, .05, .05])
    result = pipeline(
        training=training,
        testing=testing,
        validation=validation,
        model=Model,
        stopper='early',
        epochs=100,
        #model
        dimensions=emb_dim,
        loss=Loss,
        optimizer_kwargs=dict(lr=1.0e-1),
        training_kwargs=dict(num_epochs=5 ),
        #evaluation_kwargs=dict(use_tqdm=False),
        random_seed=420,
        #device='cpu',
        clear_optimizer = True,
        training_loop=Training_loop,
    )
        
    result.save_to_directory()
    results.plot()

triple_path = '/sc-projects/sc-proj-ukb-cvd/data/2_datasets_pre/220208_graphembeddings/triples/triple_list_graph_full_v3.feather'
#start run with passed arguments
 # sub_batching and slicing only for SLCWA
    # triple_path, inverse, Model, emb_dim, Training_loop, check_name: str, embedding_dict Name: str, batch_size, sub_batch_size, slice_size,  Loss=None
start_train(triple_path, True, RotatE, 256, LCWATrainingLoop, None, None,None, SoftplusLoss)
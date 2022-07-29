#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Hyperparamter tuning with TransE,
## DEFAULT ----- CHANGED TO ##
## 0.0001  ----- 1e-6 early stopper difference
## create_inverse_triples=False ---- False
## SLCWA ---- LCWA
## 256 ---- 128 batch size
## added [ slice_size=4,only_size_probing=False, clear_optimizer=True,] to .train

# create one epoche, safe and evaluate manuel . after that, start again

import pykeen
from pykeen.pipeline import pipeline
import networkx as nx
import pathlib
from random import sample
import pandas as pd
from pykeen.triples import TriplesFactory
import torch
import numpy as np
from pykeen.models import TransE

# load tripleList from /home/tilingl/GNN/Pykeen
tripleArray = pd.read_feather('triple_list.feather').to_numpy()
print('length of the triple array: ', len(tripleArray), type(tripleArray))


# In[22]:


# defragmented version - first create triple factory
'''When continuing the training or in general using the model after resuming training, it is critical that the entity label to
identifier (entity_to_id) and relation label to identifier (relation_to_id) mappings are the same as the ones that were used 
when saving the checkpoint. If they are not, then any downstream usage will be nonsense. '''
tf = TriplesFactory.from_labeled_triples(tripleArray, create_inverse_triples=False)
#training = tf.to_core_triples_factory()
print('loading TriplesFactory done ... ', type(tf))

#training_factory, testing_factory = tf.split([1.0, 0.0])
#entity_mapping = tf.entity_to_id
#relation_mapping = tf.relation_to_id
print(tf.num_entities, tf.num_relations)

#training = TriplesFactory.from_labeled_triples(training_factory, create_inverse_triples= True, )
#
#testing = TriplesFactory.from_labeled_triples(testing_factory, create_inverse_triples=True,
#                                              entity_to_id=train.entity_to_id,
 #                                             relation_to_id=train.relation_to_id,)
                                              

# Pick a model
# loss:None -> loss default specific to the model subclass.
kwargs={'triples_factory': tf, 'loss': None, 'predict_with_sigmoid':False, 'preferred_device':None}
model = TransE(**kwargs, embedding_dim=256)

# Pick an optimizer from Torch
from torch.optim import Adam
optimizer = Adam(params=model.get_grad_params())

# Pick a training approach !! contains the losses
from pykeen.training import LCWATrainingLoop 
training_loop = LCWATrainingLoop(

    model=model,

    triples_factory=tf,

    optimizer=optimizer,
)

# just run for one epoch, evaluate losses and restart training where it was left
for i in range(1,10):
    # Train like Cristiano Ronaldo
    losses2 = training_loop.train(
        triples_factory=tf,
        num_epochs=i,
        batch_size=128,
        checkpoint_name= 'TransE_test!.pt',
        checkpoint_frequency=0,
        
        slice_size=4, #>0 divisor
        only_size_probing=True, # only on GPU True
        clear_optimizer=True,
    )
    if i>1 and (losses2[-2] - losses2[-1]) < 1e-6: # maybe add |x-y| bc of 
        break


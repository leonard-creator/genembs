# script to run the NodePiece Model and extract the embeddings for all entitys out of it

# TODO: - test different versions of the aggregation function and the Tokenization techniques. 
#       - Pre-compute tokenization
#       - check number of unique relations
#       - epochs

import pykeen
import pandas as pd
import numpy as np
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.constants import PYKEEN_CHECKPOINTS
import torch
import torch.nn
import argparse
from pykeen.regularizers import Regularizer, regularizer_resolver
from pykeen.utils import resolve_device

#import all models and parameter
from pykeen.models  import ConvE, TransE, ComplEx, MuRE, RotatE, TuckER, DistMult, RESCAL, NodePiece
from pykeen.training import SLCWATrainingLoop
from pykeen.losses import BCEWithLogitsLoss, SoftplusLoss, NSSALoss, SoftMarginRankingLoss, PairwiseLogisticLoss

# testing:
from pykeen.datasets import FB15k237, Nations

# Aggregation Function for NeuralNodePiece
class DeepSet(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, dim=-2):
        x = self.encoder(x).mean(dim)
        x = self.decoder(x)
        return x

# Importing Data over the triple List, transforming to TripleFactory
triple_path = '/sc-projects/sc-proj-ukb-cvd/data/2_datasets_pre/220208_graphembeddings/triples/triple_list_graph_full_v3.feather'
data = pd.read_feather(triple_path).to_numpy()
tf = TriplesFactory.from_labeled_triples(data, create_inverse_triples=True) # must be true for NodePiece
emb_dict_name="NodePiece_t3"
print('loading TF done ..')
del data

# build simple NodePiece Model
# rules of thumb:
# -keeping num_anchors as 1-10% of total nodes in the graph is a good start
# -graph density is a major factor: the denser the graph, the fewer num_anchors you’d need. For dense FB15k237 100 total anchors (over 15k total nodes) seems to be good enough, while for sparser WN18RR we needed at least 500 anchors (over 40k total nodes). For dense OGB WikiKG2 of 2.5M nodes a vocab of 20K anchors (< 1%) already leads to SOTA results
# -the same applies to anchors per node: you’d need more tokens for sparser graphs and fewer for denser
# --> ca.700.000 nodes in Graph, really dense so 700.000 * 0,01 = 7.000 -> 5.000 should be enough
# -he size of the relational context depends on the density and number of unique relations in the graph, eg, in FB15k237 we have 237 * 2 = 474 unique relations and only 11 * 2 = 22 in WN18RR. If we select a too large context, most tokens will be PADDING_TOKEN and we don’t want that.
# --> 202 relations, often use , because in FB15k237 we have 237 * 2 = 474 -> 12 we choose 12

#crashes:
"""
simple_model = NodePiece(
    triples_factory=tf,
    random_seed=420, # working?
    tokenizers=["AnchorTokenizer", "RelationTokenizer"],
    num_tokens=[5000, 12], # default selection=32
    embedding_dim=128, #1024
)"""

# make sure, that gpu is avaiable and chosen 
device = 'gpu' 
_device: torch.device = resolve_device(device)
print(f"Using device: {device}", type(_device))

# [5000,12], 1400 num_anchors -> 46.3% do not have any anchors
# [500,12], 20000 num_anchors -> 9% do not have any anchors
# [20, 12], 20000 num_anchors -> 3% " - "
# [18,12], 23000 num_anchors -> 2,68% " - "
"""
big_Dat_model= NodePiece(
    triples_factory=tf,
    random_seed=420,
    tokenizers=["AnchorTokenizer", "RelationTokenizer"],
    num_tokens=[18,12], #500, 5000 bc. of saturation point at 20 anchors per node even in million node graphs
    tokenizers_kwargs=[
        dict(
            selection="MixtureAnchorSelection",
            selection_kwargs=dict(
                selections=["degree", "pagerank","random"],
                ratios=[0.4, 0.4, 0.2],
                num_anchors=23000,
            ),
            searcher="ScipySparse", #breadth-first search for big data!
        ),
        dict(),  # empty dict for the RelationTokenizer - it doesn't need any kwargs
    ],
    embedding_dim=1024,
    interaction="TransE", #added at 11.07.22
)
"""
"""
relation_model= NodePiece(
    triples_factory=tf,
    random_seed=420,
    tokenizers="RelationTokenizer",
    num_tokens=12,
    embedding_dim=1024,
    interaction="distmult", #default
    relation_initializer="init_phases",
    relation_constrainer="complex_normalize",
    entity_initializer="xavier_uniform_",
    aggregation=DeepSet(hidden_dim=1024),
)
"""
#crashes CUDA out of memory:
NeuralNodePiece=NodePiece(
    triples_factory=tf,
    tokenizers=["AnchorTokenizer", "RelationTokenizer"],
        num_tokens=[20, 12],
        tokenizers_kwargs=[
            dict(
                selection="MixtureAnchorSelection",
                selection_kwargs=dict(
                    selections=["degree", "pagerank", "random"],
                    ratios=[0.4, 0.4, 0.2],
                    num_anchors=20000,
                ),
                searcher="ScipySparse", #breadth-first search for big data!
            ),
            dict(),  # empty dict for the RelationTokenizer - it doesn't need any kwargs
        ],
        embedding_dim=1024,
        interaction="distmult", #default
        relation_initializer="init_phases",
        relation_constrainer="complex_normalize",
        entity_initializer="xavier_uniform_",
        aggregation=DeepSet(hidden_dim=1024),
)
# pass the model
Model = NeuralNodePiece
Model= Model.to(_device) # important otherwise fall back to cpu

# start the pipeline process

# try own pipeline with simple model and Nations dataset:

# Pick an optimizer from Torch
from torch.optim import Adam
optimizer = Adam(params=Model.get_grad_params())

# Pick a training approach that was imported !! contains the losses, choose between SLCWATrainingLoop and LCWATrainingLoop
training_loop = SLCWATrainingLoop(    
    model=Model,
    triples_factory=tf,
    optimizer=optimizer,
    automatic_memory_optimization=True, 
)
for i in range(1,100):
    losses = training_loop.train(
        triples_factory=tf,
        num_epochs=i,
        batch_size=16384, #256, # if None -> automatic search for the best and greatest
        checkpoint_name= "NodePiece_t3.pt", # for example TransE_t2.pt
        checkpoint_frequency=0,
        checkpoint_directory='/sc-scratch/sc-scratch-ukb-cvd/checkpoints_pykeen_leonard', # new checkpoint dir bc of massive storage needs
        )
    if i>1 and (losses[-2] - losses[-1]) < 1e-7:
        #switch model back to cpu device:
        _device = torch.device('cpu') # 
        Model.to(_device)
        entit_rep=Model.entity_representations[0]
        
        entity_id_dict = tf.entity_to_id
        entity_ids_as_tensors = torch.as_tensor([int(v) for v in entity_id_dict.values()], dtype=torch.long)
        entity_embeddings =entit_rep(indices=entity_ids_as_tensors).detach().numpy()

        df_dict = pd.DataFrame(dict(nodes= entity_id_dict.keys(), embeddings=list(entity_embeddings)))
        print(df_dict.head())
        emb_path = '/sc-projects/sc-proj-ukb-cvd/data/2_datasets_pre/220208_graphembeddings/embeddings_leonard/Embedding_dict_' + emb_dict_name + '.feather'
        print('saved in: ',emb_path)
        df_dict.to_feather(emb_path)
        break # to stop the loop!
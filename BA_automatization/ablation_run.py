#  this scripts test how to perform an ablation study with pykeen to find the perfect model, loss, optimizer, and hyperparameters
# To do so, we must use the metrics from pykeen and split out graph data in train/test/validation set.
# if the best parameters where filtered out, test those in the start_all.sh setting to see how it actual performs
# They may be is a difference in the evaluation progress, because pykeen focuses mostly on link-prediction task..

import pykeen
import pandas as pd
import numpy as np
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import torch
import torch.nn
import argparse
from pykeen.regularizers import Regularizer, regularizer_resolver
from pykeen.utils import resolve_device
from pykeen.ablation import ablation_pipeline

#import all models and parameter
from pykeen.models  import ConvE, TransE, ComplEx, MuRE, RotatE, TuckER, DistMult, RESCAL, NodePiece
from pykeen.training import SLCWATrainingLoop
from pykeen.losses import BCEWithLogitsLoss, SoftplusLoss, NSSALoss, SoftMarginRankingLoss, PairwiseLogisticLoss

# STARTING THE ABLATION STUDY #
#dataset=Nations
directory = "/sc-projects/sc-proj-ukb-cvd/data/2_datasets_pre/220208_graphembeddings/metadata/Ablation/TransE_randomGraph3"
# define HPO ranges
model_to_model_kwargs_ranges = {
    "TransE": {
        "embedding_dim": {
            "type": "int",
            "low": 6,
            "high": 9,
            "scale": "power_two"
        }
    }    
}

model_to_training_loop_to_training_kwargs = {
    "TransE": {
        "slcwa": { # all trainingloop.train() parameters
            "num_epochs": 25,
        }
    }
}

model_to_training_loop_to_training_kwargs_ranges= {
   "TransE": {
       "slcwa": {
           "batch_size": {
               "type": "int",
               "low": 9,
               "high": 15,
               "scale": "power_two"
           },
           "num_epochs": {
               "type": "int",
               "low": 4,
               "high": 5,
               "scale": "power_two"
           }
       }
   }
}

model_to_optimizer_to_optimizer_kwargs_ranges= {
   "TransE": {
       "adam": {
           "lr": {
               "type": "float",
               "low": 0.001,
               "high": 0.1,
               "scale": "log"
           }
       }
   }
}


ablation_pipeline(
    directory=directory,
    metadata = dict(title="Ablation Study Over Graph Data for TransE.",),
    models="TransE",
    datasets="Nations", # Nations filled with GraphData binarys
    losses=["BCEWithLogitsLoss", "MarginRankingLoss", "SoftplusLoss", "NSSALoss", "PairwiseLogisticLoss"],
    training_loops="slcwa",
    optimizers="Adam",
    negative_sampler=['BasicNegativeSampler','BernoulliNegativeSampler' ],
    
    model_to_model_kwargs_ranges=model_to_model_kwargs_ranges,
    model_to_training_loop_to_training_kwargs=model_to_training_loop_to_training_kwargs,
    model_to_training_loop_to_training_kwargs_ranges=model_to_training_loop_to_training_kwargs_ranges,
    model_to_optimizer_to_optimizer_kwargs_ranges=model_to_optimizer_to_optimizer_kwargs_ranges,
    create_inverse_triples=[True, False],
    stopper="early",
    stopper_kwargs={
        "frequency": 8,
        "patience": 6,
        "relative_delta": 0.009, #In order to continue training, we expect the model to obtain an improvement > 0.9% in Hits@10.
        "metric": "hits@10",
    },
    evaluator="RankBasedEvaluator",
    evaluator_kwargs={
        "slice_size":256,
        "filtered":True, # due to KGE comparison paper
    },
        
    # Optuna-related arguments
    n_trials=3,
    #timeout=200, # either n_trials or timeout, depending on what comes first
    metric="hits@10",
    direction="maximize",
    sampler="random",
    pruner= "nop",
    best_replicates=0, # how often retrain and re-evaluate the best run

)
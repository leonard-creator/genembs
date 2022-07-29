# script to run UMAP over a couple of Embeddings


# imports 
import numpy as np
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
import seaborn as sns
import torch
import pandas as pd
import umap
from umap import UMAP
from ehrgraphs.training import setup_training
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# function to setup presentable layout (Bachelor Arbeit) for matplotlob
def setup_matplotlib():
    def setSnsStyle(style):
        sns.set(style=style, font_scale=1.5)
        font = {"family": "serif", "weight": "normal", "size": 30}
        matplotlib.rc("font", **font)
        matplotlib.rcParams["xtick.labelsize"] = 16
        matplotlib.rcParams["ytick.labelsize"] = 16
        matplotlib.rcParams["axes.titlesize"] = 24
        matplotlib.rcParams["axes.labelsize"] = 20

    setSnsStyle("ticks")


# function to print out multiple UMAP graphs from different embeddings
# allow hyperparametertuning
# implemented single use instead of data == list
# data has to be either a full path to the embedding.feather or just the name in the /..../embeddings_leonard folder
def build_UMAP(concepts, data, min_dist=0.1, metric='euclidean', n_neighbors=15, emb_dim=1024):
    # save row DataFrame in list for plotting
    row_list=[]
    # convert string emb_name into list
    if type(data) is not list:
        temp_list =[]
        temp_list.append(data)
        data = temp_list
    
    for data_name in data:
        # include path arguments
        if data_name.startswith('/'):
            path = data_name
        else:
            path = '/sc-projects/sc-proj-ukb-cvd/data/2_datasets_pre/220208_graphembeddings/embeddings_leonard/Embedding_dict_' + data_name + '.feather'
        print('Name: ', data_name)
        embedding_df = pd.read_feather(path)
        #num_record_nodes = len(datamodule.record_node_indices) #72036
        without_phecodes=71036 
        embedder = torch.nn.Embedding(without_phecodes, emb_dim)#args.model.num_outputs)

        embedding_df.set_index('nodes', inplace=True)

        # filter node-embedding out of all nodes
        for i, c in enumerate(datamodule.record_cols):
            # exclude phecodes bc there not present in datamodule.record_cols
            if not c.startswith('phecode'):
                try:
                    e = embedding_df.loc[c].embeddings
                    embedder.weight.data[i] = torch.from_numpy(e)
                except KeyError as err:
                    x=0 # placeholder
                    #print(f'Embedding missing for {err}')

        # fit data in UMAP and Transoform to 2D-embedding-space 
        es = embedder.weight.data.numpy()
        print('length embedder.weight.data.numpy: ',len(es))
        # build UMAP with hyperparameters
        fit = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            n_jobs=-1, # all cores
            n_epochs=500, # larger value more accurate -> 200 default
            )
        es_2d = fit.fit_transform(es)
        
        rows = concepts.set_index('concept_id').loc[[r for r in datamodule.record_cols if not r.startswith('phecode')]].copy()
        print(len(rows), len(es_2d))
        # %%
        rows['e0'] = es_2d[:, 0]
        rows['e1'] = es_2d[:, 1]
        
        # decide the return type of the function (DataFrame or List of DataFrames)
        if len(data)==1: 
            return rows
        else:
            row_list.append(rows)
    return row_list

# function to plot multiple UMAP figures
# row_list contains the UMAP-DataFrames
# subplot_name contains the min_dist and num_neighbor string corresponding to every subplot
# emb_title is the Header of the picture and the name of the saved figure
#subPlot is a tupel specifying the #row #collumn of the figure
def plot_figure(row_list, subplot_name, emb_title,  subPlot: tuple):
    setup_matplotlib()
    # build multiple figures in one plot
    fig, axes = plt.subplots(subPlot[0],subPlot[1], figsize=(44, 22), sharey=False, sharex=False) #((ax1, ax2, ax3, ax4),(ax5, ax6, ax7, ax8),(ax9, ax10, ax11, ax12),(ax13, ax14, ax15, ax16))
    fig.suptitle(emb_title) #score hinzufügen emb_title
    ite=0 # counter to accsess title in subplot_name list
    for ax_list in axes:
        for ax in ax_list:
            if ite ==len(row_list)-1:
                has_legend=True
            else:
                has_legend=False
            g = sns.scatterplot(data=row_list[ite], x='e0', y='e1', hue='domain_id', ax=ax, legend=has_legend) #style='concept_class_id' --> would blow up the legend
            if has_legend:
                g.legend_.remove()
            ax.set_title(subplot_name[ite])
            ite+=1
    handles, labels = axes[-1][-1].get_legend_handles_labels() 
    fig.legend(handles, labels, loc='lower right', fontsize = 'small',markerscale=3)
    plt.tight_layout()
    path = emb_title + '.png'
    fig.savefig(path)



# ONE
hydra.core.global_hydra.GlobalHydra().clear()

initialize(config_path="../../../ehrgraphs/config")
args = compose(
    config_name="config",
    overrides=[
        "user_config=/home/tilingl/ehrgraphs/config/experiments/graphembeddings_leonard_thesis_220421",
        "model.num_outputs=1024",
    ],
)
print(OmegaConf.to_yaml(args))


# TWO
datamodule, model, tags = setup_training(args)

# THREE
concepts = pd.read_csv("/sc-projects/sc-proj-ukb-cvd/data/mapping/athena/CONCEPT.csv", sep="\t")
concepts.concept_id = concepts.concept_id.apply(lambda s: f"OMOP_{s}")

# Embeddings to calculate

# score list contains C-Index max Values in same order as emb_list
score_list=['0.7434','0.7434' , '0.7434', '0.7434', '0.7426' ,'0.736', '0.7287', '0.7223', '0.7184', '0.695' ]
emb_list=['ComplEx', 'RotatE_batch', 'random_rotate', 'DistMult', 'random_normal', 'TransE', 'MuRE', 'NodePiece_t1', 'ConvE_t1', 'NodePiece_t2']
score_list_rest=['0.745', '0.7445', '0.7437', '0.7433', '0.7403', '0.7397', '0.7375', '0.7334']
rest=['TransE_Test', 'RotatE', 'TransE_false', 'ComplEx_batch', 'RESCAL', 'TransE_batch', 'DistMult_t2', 'ConvE']

big_all_score_list=['0.745', '0.7445', '0.7439','0.7437','0.7434','0.7434', '0.7434','0.7434','0.7433','0.7433','0.7429','0.7426','0.7403','0.7397','0.7375','0.736','0.7334','0.7287', '0.7223','0.7184', '0.695']
big_all=['TransE_Test','RotatE','TransE_less_relations','TransE_false','ComplEx','DistMult','random_rotate','RotatE_batch','ComplEx_batch','TransE_rare_relations','TransE_less001','random_normal','RESCAL','TransE_batch','DistMult_t2','TransE','ConvE','MuRE',  'NodePiece_t1','ConvE_t1', 'NodePiece_t2']
transformer=['/sc-projects/sc-proj-ukb-cvd/data/2_datasets_pre/220208_graphembeddings/embeddings/Transformer_256.feather']
##########################################################################################################################
##### Produce Detailed UMAPs with different Hyperparameter for BACHELOR THESIS
###############################################################################################################################

# HYPERPARAMETER BIG PICTURE

i=0 # to iterate over score_list        ADAPT THIS ONE #!!
for emb_name in big_all:
    rows_list=[]
    subplot_title_list=[]
    if emb_name.startswith('NodePiece'):
        print('NodePiece run')
        for d in tqdm((0.5, 0.75, 0.99), desc='min_dist'):
            for n in tqdm((15, 20, 75), desc='num_neighbours'):
                rows_list.append(build_UMAP(concepts,emb_name, min_dist=d, metric='euclidean', n_neighbors=n, emb_dim=1024))
                subplot_title_list.append('min_dist={}'.format(d) + '  num_neighb={}'.format(n))
        title = emb_name + 'C-Index max: ' + big_all_score_list[i]
        plot_figure(rows_list, subplot_title_list, title, (3,3))
        i+=1
        print("-----SAVED PICTURE--------")
    else:
        for d in tqdm((0.1,0.5, 0.99), desc='min_dist'):
            for n in tqdm((5, 20, 75), desc='num_neighbours'):
                rows_list.append(build_UMAP(concepts, emb_name, min_dist=d, metric='euclidean', n_neighbors=n, emb_dim=1024))
                subplot_title_list.append('min_dist={}'.format(d) + '  num_neighb={}'.format(n))
        title = emb_name + ' CIndex max: ' + big_all_score_list[i]
        #title = "Transformer_256"
        plot_figure(rows_list, subplot_title_list, title, (3,3))
        i+=1
        print("----------SAVED PICTURE ------------")

"""
################################################################################################################################
##### Big Picture of specific Embeddings in one UMAP for BACHELOR THESIS
###############################################################################################################################
big_picture_score_list=['0.745', '0.7445', '0.7439','0.7434','0.7434', , '0.7434','0.7287', '0.7184', '0.695']
big_picture=['TransE_Test','RotatE','TransE_less_relations','ComplEx','DistMult','random_rotate','MuRE', 'ConvE_t1', 'NodePiece_t2']

j=0
rows_list=[]
subplot_title_list=[]
for emb_name in big_picture:
    # start calculating umap with gloabl perspectiv
    rows_list.append(build_UMAP(concepts,emb_name, min_dist=0.99, metric='euclidean', n_neighbors=75, emb_dim=1024))
    subplot_title_list.append(emb_name + '  CIndex={}'.format(big_picture_score_list[j]))
    j+=1
    print("-----calculated UMAP--------")

# plotting one big plot
title = 'UMAPs'
plot_figure(rows_list, subplot_title_list, title, (3,3))
"""
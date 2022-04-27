#! /bin/sh

#S #BATCH --gres=gpu:nvidia_a100-sxm-80gb:1
#SBATCH --gres=gpu:1
#SBATCH -N 1-1
#SBATCH --cpus-per-gpu=128
#SBATCH -p gpu
#SBATCH --mem=100G
#SBATCH --time 96:00:0
#SBATCH --job-name="TransE_batch"
#SBATCH --output=/home/tilingl/Pykeen/outputs/emb_fullv3%j.out
EXP_NAME="TransE_batch" # ComplEx_dimTest
EXP_FOLDER="TransE"
RUN_PATH="/home/tilingl/Pykeen/New_Embedding_Stuff/Embedding_out"
 # triple_path, inverse, Model, emb_dim, Training_loop, check_name, embedding_dict_Name, batch_size, sub_batch_size, slice_size,  Loss=None
PARAMETERS_ARRAY=(True TransE 1024 SLCWATrainingLoop 'TransEv3_batch.pt' $EXP_NAME '-l None' '--batch_size None' '--sub_batch_size NOne' '--slize_size None') 
# indices 0 to 9

prep_experiment() {
    EXPERIMENT_HOME=$1 #/home/tilingl/Pykeen/New_Embedding_Stuff/Evaluation_out
    EXPERIMENT_FOLDER=$2 #name of the folder
    EXPERIMENT_NAME=$3 # name of the experiment
    VAR=$4  #!"
    EXPERIMENT_PATH=$EXPERIMENT_HOME/$EXPERIMENT_FOLDER
    if [[ ! -d $EXPERIMENT_PATH ]]
    then
        mkdir $EXPERIMENT_PATH
    fi
    # link the output and errors to exp directory
   
    ln -s /home/tilingl/Pykeen/outputs/$VAR $EXPERIMENT_PATH/softlink_$EXPERIMENT_NAME$SLURM_JOB_ID # delete if only succsesfull runs should be in the directory

}

# creating links and folder structure
prep_experiment $RUN_PATH $EXP_FOLDER $EXP_NAME emb_fullv3$SLURM_JOB_ID.out 

source /home/tilingl/.bashrc
#start shared env
conda activate /sc-projects/sc-proj-ukb-cvd/environments/gnn
echo "Starting Embedding Modelling "
echo "\n"
echo "Executing on $(hostname)"
echo "in $(which python)"
echo "with $(python -c "import torch; print(torch.cuda.device_count())") gpus"
echo "specifically GPUs $CUDA_VISIBLE_DEVICES"

# add argparser for detailled configuration
# could print out all changed arguments e.g. tuning parameter

#python /home/tilingl/Pykeen/New_Embedding_Stuff/$NAME

python /home/tilingl/Pykeen/New_Embedding_Stuff/BA\automatization/auto_generate_embs.py ${PARAMETERS_ARRAY[0]} ${PARAMETERS_ARRAY[1} ${PARAMETERS_ARRAY[2]} ${PARAMETERS_ARRAY[3]} ${PARAMETERS_ARRAY[4]} ${PARAMETERS_ARRAY[5]} ${PARAMETERS_ARRAY[6]} ${PARAMETERS_ARRAY[7]} ${PARAMETERS_ARRAY[8]} ${PARAMETERS_ARRAY[9]}


echo "Starting to evaluate Embedding."
echo "\n"
# NEW: name with v for mark as comparable Data version
NAME="emb_run_thesis_$EXP_NAME" #for wandb!
FEATHER=".feather"
EMB_PATH="/home/tilingl/Pykeen/New_Embedding_Stuff/Embeddings/Embedding_dict_$EXP_NAME$FEATHER"

conda deactivate
echo "SLURM_JOBID="$SLURM_JOBID
echo $(hostname) 
echo $(which python) 
# Specify conda environment, dataset and model name and run experiment
echo $EMB_PATH
conda activate ehrgraphs2
echo "environment: ehgraphs2"
# adapt runpath for Eval. output
#RUN_PATH="/home/tilingl/Pykeen/New_Embedding_Stuff/Evaluation_out"

#start embedding evaluation
# NEW: leonard_thesis -> Updatet graph and graphdata with data version 0
# shared Emb Folder: /sc-projects/sc-proj-ukb-cvd/data/2_datasets_pre/220208_graphembeddings/embeddings/TransE_1024.feather

python /home/tilingl/ehrgraphs/ehrgraphs/scripts/train_recordgraphs.py setup.name=$NAME user_config=/home/tilingl/ehrgraphs/config/experiments/graphembeddings_leonard_thesis_220421.yaml datamodule.load_embeddings_path=$EMB_PATH setup.tags='["leonard_thesis:v0", "first_test"]' 



echo "Done with submission script"
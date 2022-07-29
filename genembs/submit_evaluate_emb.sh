#! /bin/sh

#SBATCH --gres=gpu:1
#SBATCH -N 1-1
#SBATCH --cpus-per-gpu=32
#SBATCH -p gpu
#SBATCH --mem=200G
#SBATCH --time 24:00:0
#SBATCH --job-name="Complex_batch"
#SBATCH --output=/home/tilingl/Pykeen/outputs/eval_emb%j.out

echo Starting to create Embedding.
echo "SLURM_JOBID="$SLURM_JOBID
echo $(hostname) 
echo $(which python) 
echo environment: ehgraphs2

EXP_NAME="ComplEx_batch" #!!specify version f.e. baseline / t1 /t2
EXP_FOLDER="ComplEx" # do not addapt _t1 _t2 just base name 
RUN_PATH="/home/tilingl/Pykeen/New_Embedding_Stuff/Evaluation_out"
DATE=$( date '+%F' )
NOW=$( date '+%F' )

#prepare git: 
prep_git() {
    # Sets the snapshot of the code to be used.
    GIT_USER="leonard-creator"
    GIT_TOKEN=$1
    REPO="github.com/nebw/ehrgraphs.git"
	# pull newest git version
	git pull https://$GIT_USER:$GIT_TOKEN@$REPO
}

prep_experiment() {
    EXPERIMENT_HOME=$1 #/home/tilingl/Pykeen/New_Embedding_Stuff/Evaluation_out
    EXPERIMENT_FOLDER=$2 #name of the folder
    EXPERIMENT_NAME=$3 # name of the experiment
    EXPERIMENT_PATH=$EXPERIMENT_HOME/$EXPERIMENT_FOLDER
    if [[ ! -d $EXPERIMENT_PATH ]]
    then
        mkdir $EXPERIMENT_PATH
    fi
    # link the output and errors to exp directory
    VAR=eval_emb$SLURM_JOB_ID.out  #!"
    ln -s /home/tilingl/Pykeen/outputs/$VAR $EXPERIMENT_PATH/softlink_$EXPERIMENT_NAME$SLURM_JOB_ID # delete if only succsesfull runs should be in the directory

}

# check repository status
#prep_git "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"

# creating links and folder structure
prep_experiment $RUN_PATH $EXP_FOLDER $EXP_NAME

# NEW: name with v for mark as comparable Data version
source /home/tilingl/.bashrc
NAME="emb_run_v_$EXP_NAME"
#NAME="graph_embs2_mlpH_$EXP_NAME"
FEATHER=".feather"
EMB_PATH="//home/tilingl/Pykeen/New_Embedding_Stuff/Embeddings/Embedding_dict_$EXP_NAME$FEATHER"

# Specify conda environment, dataset and model name and run experiment
echo $EMB_PATH
conda activate ehrgraphs2
echo "Starting python"
echo "Executing on $(hostname)"
echo "in $(which python)"
echo "with $(python -c "import torch; print(torch.cuda.device_count())") gpus"
echo "specifically GPUs $CUDA_VISIBLE_DEVICES"

#start python command
##python /home/tilingl/python_test.py
# NEW: add specific Data_identifier
python /home/tilingl/ehrgraphs/ehrgraphs/scripts/train_recordgraphs.py model=graph_embeddings head=mlp training.gradient_checkpointing=False setup.name=$NAME setup.tags='["freshStart", "v19", "batch_test"]' setup.data_identifier='WandBGraphDataNoShortcuts256:v19' datamodule.load_embeddings_path=$EMB_PATH
echo Tags: freshstart, $EXP_NAME
echo "Done with submission script"

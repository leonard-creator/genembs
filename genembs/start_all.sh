#! /bin/sh

#S #BATCH --gres=gpu:nvidia_a100-sxm-80gb:1
#S #BATCH --gres=gpu:1
#S #BATCH -N 1-1
#S #BATCH --cpus-per-gpu=128
#SBATCH -p compute
#SBATCH --mem=200G
#SBATCH --time 96:00:0
#SBATCH --job-name="TESTA"
#SBATCH --output=/home/tilingl/Pykeen/outputs/emb%j.out
EXP_NAME="ConvE" # ComplEx_dimTest
EXP_FOLDER="Transe_test"
RUN_PATH="/home/tilingl/Pykeen/New_Embedding_Stuff/Embedding_out"

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
prep_experiment $RUN_PATH $EXP_FOLDER $EXP_NAME emb$SLURM_JOB_ID.out 

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
NAME=Model_$EXP_NAME.py
python /home/tilingl/Pykeen/New_Embedding_Stuff/$NAME


echo "Starting to evaluate Embedding."
echo "\n"
# NEW: name with v for mark as comparable Data version
NAME="emb_run_v_$EXP_NAME" #for wandb!
FEATHER=".feather"
EMB_PATH="//home/tilingl/Pykeen/New_Embedding_Stuff/Embeddings/Embedding_dict_$EXP_NAME$FEATHER"

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

#start python command
# NEW: leonard_thesis -> Updatet graph and graphdata with data version 0

python /home/tilingl/ehrgraphs/ehrgraphs/scripts/train_recordgraphs.py setup.name=$NAME user_config=/home/tilingl/ehrgraphs/config/experiments/graphembeddings_leonard_thesis_220421.yaml datamodule.load_embeddings_path=/sc-projects/sc-proj-ukb-cvd/data/2_datasets_pre/220208_graphembeddings/embeddings/TransE_1024.feather setup.tags='["leonard_thesis:v0", "batch_test"]' 

echo "Done with submission script"
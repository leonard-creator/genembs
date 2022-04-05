#! /bin/sh

#SBATCH --gres=gpu:1
#SBATCH -N 1-1
#SBATCH --cpus-per-gpu=32
#SBATCH -p gpu
#SBATCH --mem=64G
#SBATCH --time 24:00:0
#SBATCH --job-name="RotatE"
#SBATCH --output=/home/tilingl/Pykeen/outputs/create_emb%j.out
EXP_NAME=
EXP_FOLDER=
RUN_PATH="/home/tilingl/Pykeen/New_Embedding_Stuff/Embedding_out"

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
    VAR=create_emb$SLURM_JOB_ID.out  #!"
    ln -s /home/tilingl/Pykeen/outputs/$VAR $EXPERIMENT_PATH/softlink_$EXPERIMENT_NAME$SLURM_JOB_ID # delete if only succsesfull runs should be in the directory

}


# creating links and folder structure
prep_experiment $RUN_PATH $EXP_FOLDER $EXP_NAME


source /home/tilingl/.bashrc
#start shared env
conda activate /sc-projects/sc-proj-ukb-cvd/environments/gnn
echo "Starting python"
echo "Executing on $(hostname)"
echo "in $(which python)"
echo "with $(python -c "import torch; print(torch.cuda.device_count())") gpus"
echo "specifically GPUs $CUDA_VISIBLE_DEVICES"

# add argparser
NAME=MODEL_$EXP_NAME.py
python /home/tilingl/Pykeen/New_Embedding_Stuff/$NAME

echo "Done with submission script"
#! /bin/sh

##SBATCH --gres=gpu:nvidia_a100-sxm-80gb:1
#SBATCH --gres=gpu:1
#SBATCH -N 1-1
#SBATCH --cpus-per-gpu=128
#SBATCH -p gpu
#SBATCH --mem=100G
#SBATCH --time 12:00:0
#SBATCH --job-name="Multiple Embs"
#SBATCH --output=/home/tilingl/Pykeen/outputs/emb_fullv3%j.out
EXP_NAME="TransE_64" #"TransE_test_checkpoint" # ComplEx_dimTest
EXP_FOLDER="TransE"
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

# ------------------------------------------------------------------------------------------------------------------------ #
 # inverse, Model, emb_dim, Training_loop, check_name, embedding_dict_Name, batch_size(-b), sub_batch_size(-sb), slice_size(-s),  Loss(-l)
python /home/tilingl/Pykeen/New_Embedding_Stuff/BA_automatization/auto_generate_embs.py True TransE 64 SLCWATrainingLoop 'TransE_64.pt' $EXP_NAME -b 32768
#python /home/tilingl/Pykeen/New_Embedding_Stuff/BA_automatization/NodePiece_run.py
# ------------------------------------------------------------------------------------------------------------------------ #

## +++++++++++++++++++++++++++++++++  ONLY TO CREATE EMBS +++++++++++++++++++++++
echo "Starting Next Embedding Modelling, emb_dim=256 TransE -b 32768"
EXP_NAME="TransE_256"
python /home/tilingl/Pykeen/New_Embedding_Stuff/BA_automatization/auto_generate_embs.py True TransE 256 SLCWATrainingLoop 'TransE_256.pt' $EXP_NAME -b 32768

echo "Starting Next Embedding Modelling, emb_dim=512 TransE -b 32768"
EXP_NAME="TransE_512"
python /home/tilingl/Pykeen/New_Embedding_Stuff/BA_automatization/auto_generate_embs.py True TransE 512 SLCWATrainingLoop 'TransE_512.pt' $EXP_NAME -b 32768

echo "Starting Next Embedding Modelling, emb_dim=64 RotatE-b 32768 "
EXP_NAME="RotatE_64"
python /home/tilingl/Pykeen/New_Embedding_Stuff/BA_automatization/auto_generate_embs.py True RotatE 32 SLCWATrainingLoop 'RotatE_64.pt' $EXP_NAME -b 32768

echo "Starting Next Embedding Modelling, emb_dim=256 RotatE -b 32768"
EXP_NAME="RotatE_256"
python /home/tilingl/Pykeen/New_Embedding_Stuff/BA_automatization/auto_generate_embs.py True RotatE 128 SLCWATrainingLoop 'RotatE_256.pt' $EXP_NAME -b 32768

echo "Starting Next Embedding Modelling, emb_dim=64 Complex, -b 4096 "
EXP_NAME="ComplEx_64"
python /home/tilingl/Pykeen/New_Embedding_Stuff/BA_automatization/auto_generate_embs.py True ComplEx 32 SLCWATrainingLoop 'ComplEx_64.pt' $EXP_NAME -b 4096

echo "Starting Next Embedding Modelling, emb_dim=256 Complex, -b 4096 "
EXP_NAME="ComplEx_256"
python /home/tilingl/Pykeen/New_Embedding_Stuff/BA_automatization/auto_generate_embs.py True ComplEx 128 SLCWATrainingLoop 'ComplEx_256.pt' $EXP_NAME -b 4096

echo "Starting Next Embedding Modelling, emb_dim=64 DistMult, -b 32768 "
EXP_NAME="DistMult_64"
python /home/tilingl/Pykeen/New_Embedding_Stuff/BA_automatization/auto_generate_embs.py True DistMult 64 SLCWATrainingLoop 'DistMult_64.pt' $EXP_NAME -b 32768

echo "Starting Next Embedding Modelling, emb_dim=256 DistMult, -b 32768 "
EXP_NAME="DistMult_256"
python /home/tilingl/Pykeen/New_Embedding_Stuff/BA_automatization/auto_generate_embs.py True DistMult 256 SLCWATrainingLoop 'DistMult_256.pt' $EXP_NAME -b 32768

##  +++++++++++++++++++++++++++++++++  ONLY TO CREATE EMBS +++++++++++++++++++++++
### only commented out to create multiple embs

###echo "Starting to evaluate Embedding."
###echo "\n"
# NEW: name with v for mark as comparable Data version
###NAME="emb_run_thesis_$EXP_NAME" #for wandb!
###FEATHER=".feather"
###EMB_PATH="/sc-projects/sc-proj-ukb-cvd/data/2_datasets_pre/220208_graphembeddings/embeddings_leonard/Embedding_dict_$EXP_NAME$FEATHER"

###conda deactivate
###echo "SLURM_JOBID="$SLURM_JOBID
###echo $(hostname) 
###echo $(which python) 
# Specify conda environment, dataset and model name and run experiment
###echo $EMB_PATH
###conda activate ehrgraphs2
###echo "environment: ehgraphs2"
# adapt runpath for Eval. output
#RUN_PATH="/home/tilingl/Pykeen/New_Embedding_Stuff/Evaluation_out"

#start embedding evaluation
# NEW: leonard_thesis -> Updatet graph and graphdata with data version 0
# shared Emb Folder: /sc-projects/sc-proj-ukb-cvd/data/2_datasets_pre/220208_graphembeddings/embeddings/TransE_1024.feather
# ------------------------------------------------------------------------------------------------------------------------------ #
###python /home/tilingl/ehrgraphs/ehrgraphs/scripts/train_recordgraphs.py setup.name=$NAME user_config=/home/tilingl/ehrgraphs/config/experiments/graphembeddings_leonard_thesis_220421.yaml datamodule.load_embeddings_path=$EMB_PATH setup.tags='["leonard_thesis:v0", "-dim 64", "32768"]' 
# ------------------------------------------------------------------------------------------------------------------------------ #


echo "Done with submission script"
#! /bin/sh

#SBATCH --gres=gpu:1
#SBATCH -N 1-1
#SBATCH --cpus-per-gpu=128
#SBATCH -p gpu
#SBATCH --mem=200G
#SBATCH --time 120:00:0
#SBATCH --job-name="less001V2"
#SBATCH --output=/home/tilingl/Pykeen/outputs/emb_full_v3%j.out


#echo "Starting to evaluate random normalverteilung sd=0.14262535238676427 from rotatE Embedding on datamodule.partition=21."
echo "starting to evaluate TransE_less001V2 to check runtime with old run"
echo "\n"
echo "SLURM_JOBID="$SLURM_JOBID
EXPERIMENT_NAME="TransE_less001V2"

ln -s /home/tilingl/Pykeen/outputs/emb_full_v3$SLURM_JOB_ID.out  /home/tilingl/Pykeen/New_Embedding_Stuff/Embedding_out/TransE/softlink_$EXPERIMENT_NAME$SLURM_JOB_ID # delete if only succsesfull runs should be in the directory

# NEW: name with v for mark as comparable Data version
NAME="emb_run_thesis_TransE_less001V2" #for wandb!
FEATHER=".feather"
EMB_PATH='/sc-projects/sc-proj-ukb-cvd/data/2_datasets_pre/220208_graphembeddings/embeddings_leonard/Embedding_dict_TransE_less001.feather'
#'/sc-projects/sc-proj-ukb-cvd/data/2_datasets_pre/220208_graphembeddings/embeddings_leonard/Embedding_dict_TransE_Test.feather'
echo $(hostname) 
echo $(which python) 
# Specify conda environment, dataset and model name and run experiment
echo $EMB_PATH

source /home/tilingl/.bashrc
conda activate ehrgraphs2
echo "environment: ehgraphs2"

python /home/tilingl/ehrgraphs/ehrgraphs/scripts/train_recordgraphs.py setup.name=$NAME user_config=/home/tilingl/ehrgraphs/config/experiments/graphembeddings_leonard_thesis_220421.yaml datamodule.load_embeddings_path=$EMB_PATH setup.tags='["leonard_thesis:v0", "TransE_less001 rerun","compare to first earlys stopping" , "-dim 1024"]' 

#setup.data_identifier="WandBGraphDataNoShortcuts_emb256_min100_leonard_thesis:latest" für 256 embeddings!
#! /bin/sh

#SBATCH --gres=gpu:1
#SBATCH -N 1-1
#SBATCH --cpus-per-gpu=128
#SBATCH -p gpu
#SBATCH --mem=200G
#SBATCH --time 96:00:0
#SBATCH --job-name="AblationStudie"
#SBATCH --output=/home/tilingl/Pykeen/outputs/ablation%j.out

echo "starting the ablation script !"
EXPERIMENT_NAME="TransE"
echo $EXPERIMENT_NAME
echo "saving in : /sc-projects/sc-proj-ukb-cvd/data/2_datasets_pre/220208_graphembeddings/metadata/Ablation "
echo "\n"
echo "SLURM_JOBID="$SLURM_JOBID
echo $EXPERIMENT_NAME
echo $(hostname) 

ln -s /home/tilingl/Pykeen/outputs/ablation$SLURM_JOB_ID.out  /home/tilingl/Pykeen/New_Embedding_Stuff/Embedding_out/Ablation/softlink_$EXPERIMENT_NAME$SLURM_JOB_ID # delete if only succsesfull runs should be in the directory

source /home/tilingl/.bashrc
#start shared env
conda activate /sc-projects/sc-proj-ukb-cvd/environments/gnn
echo $(which python)

# start python script
python /home/tilingl/Pykeen/New_Embedding_Stuff/BA_automatization/ablation_run.py

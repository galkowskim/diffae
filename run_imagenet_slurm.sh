#!/usr/bin/env bash
#SBATCH --partition long
#SBATCH --account=mi2lab-normal
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem=40G
#SBATCH --time 3-23:59:00
#SBATCH --job-name=diffae
#SBATCH --output=logs/diffae-%A.log

# echo file content to logs
script_path=$(readlink -f "$0")
cat $script_path

source /etc/environment
export NCCL_IB_DISABLE=1

module load anaconda/4.0

# activate env
# . /mnt/evafs/groups/ganzha_23/mgalkowski/miniconda3/etc/profile.d/conda.sh
# conda activate inp_exp

# run exp
# cd /mnt/evafs/groups/ganzha_23/mgalkowski/masters/inp_exp

conda init
eval "$(conda shell.bash hook)" # Add this line

conda activate /mnt/evafs/groups/ganzha_23/mgalkowski/miniconda3/envs/diffae

python run_imagenet256.py
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=180G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

echo $SLURM_JOBID
echo $SLURM_NODELIST

source ~/featsynth_env/bin/activate

# change this path
ROOT=/user_data/mmhender/

# put the code directory on your python path
PYTHONPATH=:${ROOT}featsynth/code/

# go to folder where script is located
cd ${ROOT}featsynth/code/image_analysis/run/

debug=0
image_set_name=images_ecoset
layer_do='pool4'

python3 -c 'from image_analysis import get_gram_matrices; get_gram_matrices.pca_gram_matrices('${debug}', "'${image_set_name}'", "'${layer_do}'")'
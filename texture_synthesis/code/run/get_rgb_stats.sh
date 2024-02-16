#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --gres=gpu:0
#SBATCH --mem=72G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=mind-1-34
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

echo $SLURM_JOBID
echo $SLURM_NODELIST

source ~/featsynth_env/bin/activate
cd /user_data/mmhender/featsynth/texture_synthesis/code/

debug=0
# debug=1

n_ims_do=10

python3 -c 'import get_matrices_fornoise; get_matrices_fornoise.get_rgb_stats('${debug}','${n_ims_do}')'
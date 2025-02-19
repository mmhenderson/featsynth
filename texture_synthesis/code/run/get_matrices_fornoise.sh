#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
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

n_ims_do=20

batch_number=0

python3 get_matrices_fornoise.py --debug $debug --n_ims_do $n_ims_do --batch_number $batch_number


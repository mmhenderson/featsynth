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

n_ims_do=40
# n_ims_do=1
# n_steps=1000
n_steps=50
save_loss=1
all_layers=1


python3 make_featsynth_images_comb64.py --debug $debug --n_ims_do $n_ims_do --save_loss $save_loss --n_steps $n_steps --all_layers $all_layers


#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=mind-1-34
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

echo $SLURM_JOBID
echo $SLURM_NODELIST

source ~/myenv/bin/activate
cd /user_data/mmhender/featsynth/texture_synthesis/code/

debug=0
# debug=1

n_ims_do=10
# n_ims_do=1

ngrid_values=(1)

for ngrid in ${ngrid_values[@]}
do

    python3 make_featsynth_images_v1.py --debug $debug --n_ims_do $n_ims_do --n_grid_eachside $ngrid 

done
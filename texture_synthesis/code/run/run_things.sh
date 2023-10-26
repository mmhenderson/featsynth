#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

echo $SLURM_JOBID
echo $SLURM_NODELIST

source ~/myenv/bin/activate
cd /user_data/mmhender/texture_synthesis/code/

debug=0
# debug=1

n_ims_do=4

save_loss=1

which_grid=5

# ngrid_values=(2)
ngrid_values=(1)

for ngrid in ${ngrid_values[@]}
do

    python3 synthesize_things.py --debug $debug --n_ims_do $n_ims_do --save_loss $save_loss --n_grid_eachside $ngrid --which_grid $which_grid

done
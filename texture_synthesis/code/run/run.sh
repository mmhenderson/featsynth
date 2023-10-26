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

n_ims_do=100
# n_ims_do=1

save_loss=1

# do_sqrt_values=(0 1)
do_sqrt_values=(1)

which_grid_values=(2)
# which_grid_values=(1 2 3)

# ngrid_values=(2)
ngrid_values=(1 2 4 8)

for ngrid in ${ngrid_values[@]}
do

    for do_sqrt in ${do_sqrt_values[@]}
    do

        for which_grid in ${which_grid_values[@]}
        do


            python3 synthesize_cocoims_spat.py --debug $debug --n_ims_do $n_ims_do --save_loss $save_loss --n_grid_eachside $ngrid --which_grid $which_grid --do_sqrt $do_sqrt

        done

    done
    
done
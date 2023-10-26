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


which_grid_values=(5)

ngrid_values=(1 4)

for which_grid in ${which_grid_values[@]}
do

    for ngrid in ${ngrid_values[@]}
    do    
        echo $which_grid $ngrid
        python3 compute_grid_overlap.py --debug $debug --n_grid_eachside $ngrid --which_grid $which_grid

    done
    
done
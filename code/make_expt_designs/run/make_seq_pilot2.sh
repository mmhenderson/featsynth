#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

echo $SLURM_JOBID
echo $SLURM_NODELIST

source ~/myenv/bin/activate
cd /user_data/mmhender/featsynth/make_expt_designs/
PYTHONPATH=:/user_data/mmhender/featsynth/${PYTHONPATH}

python3 -c 'import make_pilot2; make_pilot2.make_trial_info()'

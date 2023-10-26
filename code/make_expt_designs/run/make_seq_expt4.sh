#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --gres=gpu:0
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

echo $SLURM_JOBID
echo $SLURM_NODELIST

source ~/featsynth_env/bin/activate
cd /user_data/mmhender/featsynth/code/make_expt_designs/
PYTHONPATH=:/user_data/mmhender/featsynth/code/${PYTHONPATH}


python3 -c 'import make_expt4; make_expt4.make_trial_info()'

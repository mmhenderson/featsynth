#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=mind-1-34
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

echo $SLURM_JOBID
echo $SLURM_NODELIST

source ~/imstat_env/bin/activate

# change this path
ROOT=/user_data/mmhender/

# put the code directory on your python path
PYTHONPATH=:${ROOT}featsynth/

# go to folder where script is located
cd ${ROOT}featsynth/texture_synthesis/code/


debug=0
# debug=1

n_ims_do=10

python3 -c 'import filter_ims_spatfreq; filter_ims_spatfreq.filter_ims('${debug}', '${n_ims_do}')'

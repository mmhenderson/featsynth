#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
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
cd ${ROOT}featsynth/code_imageanalysis/run/

use_noise=0

# python3 -c 'from code_imageanalysis import make_model_preds; make_model_preds.get_preds('${use_noise}')'
python3 -c 'from code_imageanalysis import make_model_preds_version2; make_model_preds_version2.get_preds('${use_noise}')'
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

echo $SLURM_JOBID
echo $SLURM_NODELIST

source ~/featsynth_env/bin/activate

# change this path
ROOT=/user_data/mmhender/

# put the code directory on your python path
PYTHONPATH=:${ROOT}featsynth/code/

# go to folder where script is located
cd ${ROOT}featsynth/code/prep_images/run/

debug=0

python3 -c 'from prep_images import proc_ecoset_categs; proc_ecoset_categs.get_ecoset_file_info_fornoise('${debug}')'
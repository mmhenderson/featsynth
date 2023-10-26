#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
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

grayscale=1
python3 -c 'from code_imageanalysis import prep_images; prep_images.prep('${grayscale}')'

debug=0
training_type=simclr
image_set_name=images_expt2_withscrambled

python3 -c 'from code_imageanalysis import get_features; get_features.get_resnet_features('${debug}', "'${training_type}'", "'${image_set_name}'")'

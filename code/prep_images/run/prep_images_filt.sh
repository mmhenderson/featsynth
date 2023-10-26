#!/bin/bash
#SBATCH --partition=gpu
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
cd ${ROOT}featsynth/code/prep_images/run/

grayscale=0
highpass_sigma=0
sigmas=(4 13 26)
for lowpass_sigma in ${sigmas[@]}
do
    python3 -c 'from prep_images import prep_images_v1; prep_images_v1.prep('${grayscale}', '${lowpass_sigma}', '${highpass_sigma}')'
done

lowpass_sigma=0
sigmas=(4 13 26)
for highpass_sigma in ${sigmas[@]}
do
    python3 -c 'from prep_images import prep_images_v1; prep_images_v1.prep('${grayscale}', '${lowpass_sigma}', '${highpass_sigma}')'
done

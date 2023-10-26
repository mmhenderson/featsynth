#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
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

sigmas=(4 13 26)

for sigma in ${sigmas[@]}
do
    image_set_name=images_expt1_filt_highpass_sigma${sigma}
    echo ${image_set_name}

    python3 -c 'from code_imageanalysis import get_categ_discrim; get_categ_discrim.get_discrim("'${image_set_name}'")'
    
    image_set_name=images_expt1_filt_lowpass_sigma${sigma}
    echo ${image_set_name}

    python3 -c 'from code_imageanalysis import get_categ_discrim; get_categ_discrim.get_discrim("'${image_set_name}'")'
    
done

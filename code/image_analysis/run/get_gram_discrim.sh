#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
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
cd ${ROOT}featsynth/code/image_analysis/run/

# image_set_name=images_ecoset64
image_set_name=images_ecoset64_grayscale
# image_set_name=images_ecoset_includeperson
layers_process=(pool1 pool2 pool3 pool4)
# layers_process=(pool1)

debug=0
# n_per_categ=256
n_per_categ_vals=(128 256 496)
# n_per_categ_vals=(128)
n_cv=10

for n_per_categ in ${n_per_categ_vals[@]}
do

    for layer_process in ${layers_process[@]}
    do


        python3 -c 'from image_analysis import get_gram_matrix_discrim; get_gram_matrix_discrim.compute_discrim_ecoset("'${image_set_name}'","'${layer_process}'", '${debug}', '${n_per_categ}', '${n_cv}')'

    done
    
done
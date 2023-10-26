#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --gres=gpu:1
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
image_set_name=images_things64
# image_set_name=images_things64_bilinear
# image_set_name=images_things64_music2
grayscale=0

python3 -c 'from prep_images import get_clip_embeddings; get_clip_embeddings.get_embed('${debug}', "'${image_set_name}'",'${grayscale}')'

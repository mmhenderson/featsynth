import argparse
import os
import sys
import numpy as np
import time
import torch

things_stim_path = '/user_data/mmhender/stimuli/things/'
save_stim_path = '/user_data/mmhender/stimuli/featsynth/images_v1_superord_prototypes'

import synthesize_textures
from utils import things_utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
    
def make_ims(args):
    
    st_overall = time.time()
    
    if args.debug:
        print('\nDEBUG MODE\n')

        
    # this is the set of 200 concepts used in behav experiments
    fn = '/user_data/mmhender/featsynth/code/make_expt_designs/expt1_categ_info.npy'
    info = np.load(fn, allow_pickle=True).item()
    categ_names = info['super_names']
    n_categ = len(categ_names)
    concept_names_subsample = [info['basic_names'][info['super_inds_long']==si] \
                              for si in range(n_categ)]
    
    n_conc_each = len(concept_names_subsample[0])
    
    categ_process = np.arange(n_categ)
    conc_process = np.arange(n_conc_each)
   
    im_seed = 934846

    # looping over superordinate categs
    for ca, categ_ind in enumerate(categ_process):
        categ = categ_names[categ_ind]
        
        if args.debug and (ca>2):
            continue
                
        print('\nCATEG %d of %d\n'%(ca, len(categ_process)))
        
        # figure out what images to combine to create the prototype
        # for this superordinate category
        n_exemplars_each = int(np.ceil(args.n_ims_combine/n_conc_each))

        target_image_filenames = np.concatenate(\
                                [[things_utils.get_filename(categ, conc, ii) \
                                  for conc in concept_names_subsample[categ_ind]] \
                                 for ii in range(n_exemplars_each)], \
                                    axis=0)

        # these are the images we want to combine here
        target_image_filenames = target_image_filenames[0:args.n_ims_combine]

        print('combining images in files:')
        print(target_image_filenames)

        sys.stdout.flush()

        name = '%s_prototype_%dims'%(categ, args.n_ims_combine)
        if args.debug:
            out_dir = os.path.join(save_stim_path, 'DEBUG',name)
        else:
            out_dir = os.path.join(save_stim_path, name)

        print('will save images to %s'%out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        else:
            # if the image files are all made already, skip this folder.
            files = os.listdir(out_dir)
            files = [f for f in files if '.png' in f]
            if len(files)>=5:
                print('done with %s'%out_dir)
                continue

        im_seed+=1
        synthesize_textures.make_textures_combineimages(target_image_filenames, \
                                                out_dir, 
                                                layers_do = ['pool1','pool2','pool3','pool4'], \
                                                n_steps = args.n_steps, 
                                                rndseed = im_seed, \
                                                save_loss = args.save_loss)


    elapsed = time.time() - st_overall
    print('\nTook %.5f s (%.2f min) to run entire script'%(elapsed, elapsed/60))
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--n_ims_combine", type=int,default=10,
                    help="how many images to do?")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")   
    parser.add_argument("--save_loss", type=int,default=1,
                    help="want to save loss over time for each image? 1 for yes, 0 for no")
    parser.add_argument("--n_steps", type=int,default=100,
                    help="how many steps to do per image?")

    args = parser.parse_args()

    args.debug=args.debug==1
   
    make_ims(args)

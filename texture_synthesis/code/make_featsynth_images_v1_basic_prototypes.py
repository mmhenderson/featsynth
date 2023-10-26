import argparse
import os
import sys
import numpy as np
import time
import torch

things_stim_path = '/user_data/mmhender/stimuli/things/'
save_stim_path = '/user_data/mmhender/stimuli/featsynth/images_v1_basic_prototypes'

import synthesize_textures
from utils import things_utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
    
def make_ims(args):
    
    st_overall = time.time()
    
    if args.debug:
        print('\nDEBUG MODE\n')

        
    concepts_filename = os.path.join(things_stim_path, 'concepts_use.npy')
    concepts_use = np.load(concepts_filename,allow_pickle=True).item()
    categ_names = concepts_use['categ_names']
    concept_names_subsample = concepts_use['concept_names_subsample']
    image_names = concepts_use['image_names']
    concept_ids_subsample = concepts_use['concept_ids_subsample']
    n_categ = len(categ_names)
    n_conc_each = len(concept_names_subsample[0])
    
    categ_process = np.arange(n_categ)
    conc_process = np.arange(n_conc_each)
    
    # which images will we combine to make each prototype?
    # lets use all 10
    ims_combine = np.arange(args.n_ims_combine)
    
    im_seed = 345646

    # looping over superordinate categs
    for ca, categ_ind in enumerate(categ_process):
        categ = categ_names[categ_ind]
        
        print('\nCATEG %d of %d\n'%(ca, len(categ_process)))
        
        # looping over basic-level categs (making one prototype per basic)
        for co, conc_ind in enumerate(conc_process):
            conc = concept_names_subsample[categ_ind][conc_ind]
            
            if args.debug and (ca>2 or co>1):
                continue
           
            # these are the images we want to combine here
            target_image_filenames = [things_utils.get_filename(categ, conc, ii) \
                                      for ii in ims_combine]
    
            print('\nCONCEPT %d of %d\n'%(co, len(conc_process)))
            print('combining images in files:')
            print(target_image_filenames)
            
            sys.stdout.flush()

            name = '%s_prototype'%(conc)
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

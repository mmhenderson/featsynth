import argparse
import os
import sys
import numpy as np

import torch
import matplotlib.pyplot as plt     # type: ignore

# things_stim_path = '/user_data/mmhender/things/THINGS_smaller/'
save_stim_path = '/user_data/mmhender/stimuli/'
root = os.path.dirname(os.getcwd())

# sys.path.insert(0,root)

import PIL
import time
import shutil

import things_utils

def make_ims(args):
    
    st_overall = time.time()
    
    concepts_filename = '/user_data/mmhender/things/concepts_use.npy'
    concepts_use = np.load(concepts_filename,allow_pickle=True).item()
    categ_names = concepts_use['categ_names']
    concept_names_subsample = concepts_use['concept_names_subsample']
    image_names = concepts_use['image_names']
    concept_ids_subsample = concepts_use['concept_ids_subsample']
    n_categ = len(categ_names)
    n_conc_each = len(concept_names_subsample[0])
    
    categ_process = np.arange(n_categ)
    conc_process = np.arange(n_conc_each)
    ims_process = np.arange(args.n_ims_do)
    
#     categ_process = [0]
#     conc_process = [0]
#     ims_process = [0]
    
    for ca, categ_ind in enumerate(categ_process):
        categ = categ_names[categ_ind]
        
        for co, conc_ind in enumerate(conc_process):
            conc = concept_names_subsample[categ_ind][conc_ind]
            
            if args.debug and (ca>2 or co>1):
                continue
                
            for ii in ims_process:
        
                target_image_filename = things_utils.get_filename(categ, conc, ii)
             
                name = target_image_filename.split('/')[-1].split('.jpg')[0]
                out_dir_orig = os.path.join(save_stim_path, 'things_synth',name)
                out_dir_new = os.path.join(save_stim_path, 'things_synth_stimuli_1x1')
                if not os.path.exists(out_dir_new):
                    os.makedirs(out_dir_new)

                filename_save_orig = os.path.join(out_dir_orig, 'orig.png')
                filename_save_new = os.path.join(out_dir_new, '%s_orig.png'%name)
                print(filename_save_new)
                sys.stdout.flush()
                
                shutil.copyfile(filename_save_orig, filename_save_new)
                    
                important_layers = ['relu1_1', 'pool1','pool2','pool3','pool4']
                
                layers_do = [1,2,3,4]
                for ll in layers_do:

                    filename_save_orig = os.path.join(out_dir_orig, 'grid%d_%dx%d_upto_%s.png'%(args.which_grid, \
                                                                                      args.n_grid_eachside, \
                                                                                      args.n_grid_eachside, \
                                                                                      important_layers[ll]))
                    
                    filename_save_new = os.path.join(out_dir_new, \
                                                     '%s_grid%d_%dx%d_upto_%s.png'%(name, \
                                                                                    args.which_grid, \
                                                                                    args.n_grid_eachside, \
                                                                                    args.n_grid_eachside, \
                                                                                    important_layers[ll]))
                    shutil.copyfile(filename_save_orig, filename_save_new)
                    
                    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--n_ims_do", type=int,default=10,
                    help="how many images to do?")
    parser.add_argument("--n_grid_eachside", type=int,default=2,
                    help="how many grid spaces per square side?")
    parser.add_argument("--n_steps", type=int,default=100,
                    help="how many steps to do per image?")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--save_loss", type=int,default=0,
                    help="want to save loss over time for each image? 1 for yes, 0 for no")
   
    parser.add_argument("--do_sqrt", type=int,default=1,
                    help="take sqrt of overlap? 1 for yes, 0 for no")
    parser.add_argument("--which_grid", type=int,default=1,
                    help="which of the spatial grid methods to use?")

    args = parser.parse_args()

    args.debug=args.debug==1
    args.do_sqrt=args.do_sqrt==1
    
    make_ims(args)

import os
import sys
import numpy as np

import PIL
import PIL.Image

import time

import things_utils

things_stim_path = '/user_data/mmhender/stimuli/things/'
orig_stim_path = '/user_data/mmhender/stimuli/featsynth/images_v1'
save_stim_path = '/user_data/mmhender/stimuli/featsynth/images_v1_grayscale'

def convert_ims(debug=0, n_ims_do=10):
    
    debug = debug==1
    
    st_overall = time.time()
    
    if debug:
        print('\nDEBUG MODE\n')
        sys.stdout.flush()
        
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
    ims_process = np.arange(n_ims_do)
    
    for ca, categ_ind in enumerate(categ_process):
        categ = categ_names[categ_ind]
        
        for co, conc_ind in enumerate(conc_process):
            conc = concept_names_subsample[categ_ind][conc_ind]
            
            if debug and (ca>2 or co>1):
                continue
                
            for ii in ims_process:
        
                target_image_filename = things_utils.get_filename(categ, conc, ii)
    
                print('\nCATEG %d of %d, IMAGE %d\n'%(ca, len(categ_process), ii))
                print('processing target image %s'%target_image_filename)
                sys.stdout.flush()

                name = target_image_filename.split('/')[-1].split('.jpg')[0]
                orig_dir = os.path.join(orig_stim_path, name)
                if debug:
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
                        
                which_grid = 5;
                n_grid_eachside=1; 
                important_layers = ['relu1_1', 'pool1','pool2','pool3','pool4']
                layers_do = [1,2,3,4]
                
                filenames_convert = ['orig.png']
                filenames_convert += ['grid%d_%dx%d_upto_%s.png'%(which_grid, \
                                                                             n_grid_eachside, \
                                                                             n_grid_eachside, \
                                                                            important_layers[ll])\
                                     for ll in layers_do]
                
                for file in filenames_convert:
                    filename_orig = os.path.join(orig_dir, file)
                    filename_save = os.path.join(out_dir, file)
                    
                    print('loading from %s'%(filename_orig))
                    
                    # load it and convert to grayscale format.
                    # this function uses:
                    # L = R * 299/1000 + G * 587/1000 + B * 114/1000
                    image = PIL.Image.open(filename_orig).convert('L')
                    
                    print('saving image to %s'%filename_save)
                    image.save(filename_save)
                
import os
import sys
import numpy as np

import PIL
import PIL.Image

import time


things_stim_path = '/user_data/mmhender/stimuli/things/'
orig_stim_path = '/user_data/mmhender/stimuli/featsynth/images_v1'
save_stim_path = '/user_data/mmhender/stimuli/featsynth/images_v1_filtered'

import things_utils
from code_utils import filter_utils

def filter_ims(debug=0, n_ims_do=10):
    
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
    
    lowpass_cutoffs_cpd = np.array([1, 2, 6])
    highpass_cutoffs_cpd = np.array([1, 2, 6])

    degrees_per_image = 10;
    pix_per_image = 256;

    lowpass_sigmas = np.round(1/lowpass_cutoffs_cpd/degrees_per_image*pix_per_image)
    highpass_sigmas = lowpass_sigmas
    # highpass_sigmas = np.round(1/highpass_cutoffs_cpd/degrees_per_image*pix_per_image)
    print('sigmas are:')
    print(lowpass_sigmas)
    
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
                
                out_dirs_lp = {}
                out_dirs_hp = {}
                
                for si, lps in enumerate(lowpass_sigmas):
                    hps = lps
                    
                    if debug:
                        out_dir = os.path.join(save_stim_path, 'filt_lowpass_sigma%d'%lps, 'DEBUG',name)
                    else:
                        out_dir = os.path.join(save_stim_path, 'filt_lowpass_sigma%d'%lps, name)

                    out_dirs_lp[lps] = out_dir
                    
                    print('will save images to %s'%out_dir)

                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                        
                    if debug:
                        out_dir = os.path.join(save_stim_path, 'filt_highpass_sigma%d'%hps, 'DEBUG',name)
                    else:
                        out_dir = os.path.join(save_stim_path, 'filt_highpass_sigma%d'%hps, name)

                    out_dirs_hp[hps] = out_dir
                    
                    print('will save images to %s'%out_dir)

                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    
                        
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
                    
                    print('loading from %s'%(filename_orig))
                    
                    # load the image
                    image = PIL.Image.open(filename_orig)
                    
                    npix = image.size[0]
                    image_mat = np.reshape(np.array(image.getdata()), [npix, npix, 3]).astype(np.float64)
                    image_mat /= 255

                    for si, lps in enumerate(lowpass_sigmas):
                        
                        hps = lps
                        
                        image_lp = filter_utils.filter_lowpass(image_mat, lps)
                        
                        image_save = PIL.Image.fromarray(image_lp)
                        
                        filename_save = os.path.join(out_dirs_lp[lps], file)
                    
                        print('saving lowpass image to %s'%filename_save)
                        image_save.save(filename_save)
                        
                        
                        image_hp = filter_utils.filter_highpass(image_mat, hps)
                        
                        image_save = PIL.Image.fromarray(image_hp)
                        
                        filename_save = os.path.join(out_dirs_hp[hps], file)
                    
                        print('saving highpass image to %s'%filename_save)
                        image_save.save(filename_save)

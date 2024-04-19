import os
import sys
import numpy as np
import pandas as pd


import PIL
import PIL.Image

orig_stim_path = '/user_data/mmhender/stimuli/featsynth/images_comb64'
save_stim_path = '/user_data/mmhender/stimuli/featsynth/images_comb64_grayscale'
project_root = '/user_data/mmhender/featsynth/'

def convert_ims(debug=0):
    
    debug = debug==1
    
    if debug:
        print('\nDEBUG MODE\n')
        sys.stdout.flush()

    list_folder = os.path.join(project_root, 'features','raw')
    image_set_name = 'images_comb64'
    list_orig = os.path.join(list_folder, '%s_list.csv'%(image_set_name))
    df = pd.read_csv(list_orig)
    
    n_images = len(df)

    if not os.path.exists(save_stim_path):
        os.makedirs(save_stim_path)

    for ii in range(n_images):
    
        if debug and (ii>1):
            continue

        bname = np.array(df['basic_name'])[ii]
        ename = 'ex%02d'%(np.array(df['exemplar_number'])[ii])

        orig_dir = os.path.join(orig_stim_path, bname, ename)
        if debug:
            out_dir = os.path.join(save_stim_path, 'DEBUG', bname, ename)
        else:
            out_dir = os.path.join(save_stim_path, bname, ename)

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
                

        important_layers = ['relu1_1', 'pool1','pool2','pool3','pool4']
        layers_do = [1,2,3,4]
        
        filenames_convert = ['orig.png']
        filenames_convert += ['scramble_upto_%s.png'%(important_layers[ll])\
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
            sys.stdout.flush()
            
                
import argparse
import os
import sys
import numpy as np
import pandas as pd
import time

save_stim_path = '/user_data/mmhender/stimuli/featsynth/images_v2'
# some paths that we need to get image info
ecoset_path = '/lab_data/tarrlab/common/datasets/Ecoset/'
ecoset_info_path = '/user_data/mmhender/stimuli/ecoset_info/'

import synthesize_textures

# this is ECOSET images, in 64 categories
# these are not well-checked images - they are randomly sampled from ecoset.

def make_ims(args):
    
    st_overall = time.time()
    
    if args.debug:
        print('\nDEBUG MODE\n')
    
    # info about which categories to use
    fn = os.path.join(ecoset_info_path, 'categ_use_ecoset.npy')
    info = np.load(fn, allow_pickle=True).item()
    bnames = np.array(list(info['binfo'].keys()))
    
    # list of all files in each category
    fn = os.path.join(ecoset_info_path, 'ecoset_file_info.npy')
    efiles = np.load(fn, allow_pickle=True).item()
    
    pix_thresh = 256

    im_seed = 546456
    
    for bi, bname in enumerate(bnames):
    
        if args.debug and (bi>1):
            continue

        # choose images to analyze here
        folder = os.path.join(ecoset_path, 'train', info['binfo'][bname]['ecoset_folder'])
        imfiles_all = efiles[bname]['train']['images']
        sizes = efiles[bname]['train']['size']
        abv_thresh = np.array([(s1>=pix_thresh) & (s2>=pix_thresh) for s1, s2 in sizes])
        is_rgb = np.array(efiles[bname]['train']['mode'])=='RGB'
        ims_use_all = np.where(abv_thresh & is_rgb)[0]

        np.random.seed(353455+bi)
        ims_use = np.random.choice(ims_use_all, args.n_ims_do, replace=False)

        for ii, im in enumerate(ims_use):

            target_image_filename = os.path.join(folder, imfiles_all[im])

            print('\nCATEG %d of %d, IMAGE %d\n'%(bi, len(bnames), ii))
            print('processing target image %s'%target_image_filename)
            sys.stdout.flush()

            name = '%s_%s'%(bname, imfiles_all[im].split('.')[0])
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
            synthesize_textures.make_textures_oneimage(target_image_filename, \
                                                    out_dir, 
                                                    layers_do = ['pool1','pool2','pool3','pool4'], \
                                                    n_steps = args.n_steps, 
                                                    rndseed = im_seed, \
                                                    save_loss = args.save_loss)
        
        
    elapsed = time.time() - st_overall
    print('\nTook %.5f s (%.2f min) to run entire script'%(elapsed, elapsed/60))
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--n_ims_do", type=int,default=10,
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

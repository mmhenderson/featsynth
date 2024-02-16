import argparse
import os
import sys
import numpy as np
import time
import torch
import pandas as pd

save_stim_path = '/user_data/mmhender/stimuli/featsynth/noise_images/'

# this is where target mean and std of gram matrices are stored
gm_path = '/user_data/mmhender/stimuli/featsynth/matrices_fornoise/'

# ecoset_info_path = '/user_data/mmhender/stimuli/ecoset_info/'

import synthesize_textures

def make_ims(args):
    
    st_overall = time.time()
    
    if args.debug:
        print('\nDEBUG MODE\n')
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
 
    ims_process = np.arange(args.n_ims_do)

    # loading all the stats computed over a random set of ecoset images
    # these will be used to constrain the noise to be natural-ish
    layers_all = ['relu1_1', 'pool1','pool2','pool3','pool4']

    gmeans = []; gstds = []

    for layer in layers_all:
        fn = os.path.join(gm_path, 'gm_mean_%s.npy'%layer)
        print(fn)
        m = np.load(fn)
        gmeans += [torch.Tensor(m)]
        fn = os.path.join(gm_path, 'gm_std_%s.npy'%layer)
        print(fn)
        s = np.load(fn)
        gstds += [torch.Tensor(s)]

    fn = os.path.join(gm_path, 'rgb_hist_all.npy')
    print(fn)
    rgb_hist_vals = np.load(fn, allow_pickle=True).item()
    
    rndseed = 345464
    
    for ii in ims_process:
   
        if args.debug and (ii>1):
            continue

        seed_list = np.arange(rndseed, rndseed+4)
        rndseed +=4
        
        # keys = ['seed%d'%s for s in [0,1,2,3]]
        # seed_list = [np.array(labs[k])[ii] for k in keys]

        print(seed_list)
        # im_seed+=4

        # if ii>10:
        #     continue

        # target_image_filename = np.array(labs['image_filename_new'])[ii]
    
        print('making noise image %d of %d'%(ii, args.n_ims_do))
        sys.stdout.flush()

        name = 'ex%02d'%(ii)
        print(save_stim_path)
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
            if len(files)>=4:
                print('done with %s'%out_dir)
                continue

        
        layers_do = ['pool1','pool2','pool3','pool4']
        synthesize_textures.make_textures_noise(target_means=gmeans, \
                                                target_stds=gstds, \
                                                out_dir = out_dir,
                                                layers_do = layers_do, \
                                                n_steps = args.n_steps, 
                                                rndseed = seed_list, 
                                                save_loss = args.save_loss, \
                                                rgb_hist_vals = rgb_hist_vals)
    

    elapsed = time.time() - st_overall
    print('\nTook %.5f s (%.2f min) to run entire script'%(elapsed, elapsed/60))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--n_ims_do", type=int,default=40,
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

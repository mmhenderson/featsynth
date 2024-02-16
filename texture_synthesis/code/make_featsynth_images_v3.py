import argparse
import os
import sys
import numpy as np
import time
import torch

things_stim_path = '/user_data/mmhender/stimuli/things/'
save_stim_path = '/user_data/mmhender/stimuli/featsynth/images_v3'
texture_synth_root = os.path.dirname(os.getcwd())
ecoset_info_path = '/user_data/mmhender/stimuli/ecoset_info/'

# this is THINGS images, in 64 categories like ecoset.
# but this is the OLD version of the 64 categories (includes ukulele and mandolin).
# this is for behav expt 3

import synthesize_textures

def make_ims(args):
    
    st_overall = time.time()
    
    if args.debug:
        print('\nDEBUG MODE\n')
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # info about which categories to use
    fn = os.path.join(ecoset_info_path, 'categ_use_ecoset_OLDVERSION.npy')
    info = np.load(fn, allow_pickle=True).item()
    bnames = np.array(list(info['binfo'].keys()))
    
    fn2load = os.path.join(things_stim_path, 'things_file_info.npy')
    tfiles = np.load(fn2load, allow_pickle=True).item()

    ims_process = np.arange(args.n_ims_do)
    im_seed = 1424245
    
    for bi, bname in enumerate(bnames):
      
        conc = bname

        if args.debug and (bi>1):
            continue

        for ii in ims_process:
            
            target_image_filename = os.path.join(things_stim_path, 'Images', conc, tfiles[conc][ii])
        
            print('\nCATEG %d of %d, IMAGE %d\n'%(bi, len(bnames), ii))
            print('processing target image %s'%target_image_filename)
            sys.stdout.flush()

            name = tfiles[conc][ii].split('.jpg')[0]
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

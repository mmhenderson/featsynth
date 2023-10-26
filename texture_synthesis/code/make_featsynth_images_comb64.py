import argparse
import os
import sys
import numpy as np
import time
import torch
import pandas as pd

orig_stim_path = '/user_data/mmhender/stimuli/featsynth/images_comb64_orig'
save_stim_path = '/user_data/mmhender/stimuli/featsynth/images_comb64'

ecoset_info_path = '/user_data/mmhender/stimuli/ecoset_info/'

import synthesize_textures

def make_ims(args):
    
    st_overall = time.time()
    
    if args.debug:
        print('\nDEBUG MODE\n')
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # The full list of images to use is already made.
    # Made this in choose_extra_ecoset_ims.py and prep_images_comb64.py
    image_list_filename = os.path.join(orig_stim_path, 'images_comb64_list.csv')
    print(image_list_filename)
    image_list = pd.read_csv(image_list_filename, index_col=0)

    # save_stim_path = '/user_data/mmhender/stimuli/featsynth/images_comb64'
    # print(save_stim_path)
    # if args.all_layers:
    #     save_stim_path = save_stim_path + '_alllayers'
    # else:
    #     save_stim_path = save_stim_path + '_selectlayers'

    # save_stim_path = save_stim_path + '_%dsteps'%args.n_steps
    
    n_images = image_list.shape[0]

    # info about ecoset categories
    fn = os.path.join(ecoset_info_path, 'categ_use_ecoset.npy')
    info = np.load(fn, allow_pickle=True).item()
    bnames = list(info['binfo'].keys())

    
    ims_process = np.arange(args.n_ims_do)
    # im_seed = 867878

    batches = [np.arange(0, 16), np.arange(16, 32), np.arange(32, 48), np.arange(48, 64)]

    for bi in batches[args.batch_number]:

        bname = bnames[bi]

        labs = image_list.iloc[np.array(image_list['basic_name']==bname)]
      
        if args.debug and (bi>1):
            continue

        for ii in ims_process:

            keys = ['seed%d'%s for s in [0,1,2,3]]
            seed_list = [np.array(labs[k])[ii] for k in keys]

            print(seed_list)
            # im_seed+=4

            # if ii>10:
            #     continue

            target_image_filename = np.array(labs['image_filename_new'])[ii]
        
            print('\nCATEG %d of %d, IMAGE %d\n'%(bi, len(bnames), ii))
            print('processing target image %s'%target_image_filename)
            sys.stdout.flush()

            name = 'ex%02d'%(np.array(labs['exemplar_number'])[ii])
            print(save_stim_path)
            if args.debug:
                out_dir = os.path.join(save_stim_path, 'DEBUG',bname, name)
            else:
                out_dir = os.path.join(save_stim_path, bname, name)

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

            
            layers_do = ['pool1','pool2','pool3','pool4']
            synthesize_textures.make_textures_oneimage(target_image_filename, \
                                                    out_dir, 
                                                    layers_do = layers_do, \
                                                    n_steps = args.n_steps, 
                                                    rndseed = seed_list, 
                                                    # rndseed = im_seed, \
                                                    all_layers = args.all_layers, \
                                                    save_loss = args.save_loss)
        

    elapsed = time.time() - st_overall
    print('\nTook %.5f s (%.2f min) to run entire script'%(elapsed, elapsed/60))



def proc_synth_losses():

    # once the synths are all made, this function goes into folders and collects the 
    # final loss for each synth. this will be used to pick the "best" synths.

    fn = os.path.join(ecoset_info_path, 'categ_use_ecoset.npy')
    info = np.load(fn, allow_pickle=True).item()
    bnames = list(info['binfo'].keys())

    folder = save_stim_path
    
    n_categ = len(bnames)
    n_ex = 40
    
    n_layers = 4;
    layers = ['pool1','pool2','pool3','pool4']  
    
    all_final_l = np.zeros((n_categ, n_ex, n_layers))
    
    for bi in range(n_categ):
    
        print(bi)
    
        bname = bnames[bi]
    
        for ee in range(n_ex):
                
            image_path = os.path.join(folder, bname, 'ex%02d'%ee)
            
            for li in range(n_layers):
                
                lfn = os.path.join(image_path, 'loss_scramble_upto_%s.csv'%layers[li])
                
                l = pd.read_csv(lfn)
                
                final_l = np.array(l['loss'])[-1]
        
                all_final_l[bi, ee, li] = final_l
        
    
    # want to combine the losses across synths from different layers.
    # they have different magnitudes, so z-scoring here
    
    vals = np.reshape(all_final_l, [n_categ*n_ex, n_layers], order='F')
    
    zvals = (vals - np.mean(vals, axis=0, keepdims=True)) / np.std(vals, axis=0, keepdims=True)
    
    # then average the z-scores for each layer
    avgz = np.mean(zvals, axis=1)
    
    # put back to original shape
    avgz = np.reshape(avgz, [n_categ, n_ex], order='F')
    
    
    losses = dict()
    
    for bi in range(n_categ):
    
        bname = bnames[bi]
        
        # finding top images based on z-scored loss values
        
        l = avgz[bi,:]
    
        # l = np.mean(all_final_l[bi,:,:], axis=1)
        
        ex_sorted = np.argsort(l)
    
        losses[bname] = dict()
    
        # this is the order to grab exemplars for the experiment (best first)
        losses[bname]['order'] = np.argsort(l)
    
        # also saving original loss vals
        losses[bname]['loss_combined_z'] = l
        losses[bname]['loss_pool1'] = all_final_l[bi,:,0]
        losses[bname]['loss_pool2'] = all_final_l[bi,:,1]
        losses[bname]['loss_pool3'] = all_final_l[bi,:,2]
        losses[bname]['loss_pool4'] = all_final_l[bi,:,3]
        

    fn2save = os.path.join(folder, 'synth_losses_all.npy')
    print(fn2save)
    np.save(fn2save, losses)
    

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
    parser.add_argument("--all_layers", type=int,default=1,
                    help="include all layers up to current (not subset)?")
    parser.add_argument("--batch_number", type=int,default=0,
                    help="batch n out of 4?")

    args = parser.parse_args()

    args.debug=args.debug==1
    args.all_layers=args.all_layers==1
    
    make_ims(args)

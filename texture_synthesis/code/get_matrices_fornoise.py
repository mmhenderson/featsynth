import argparse
import os
import sys
import numpy as np
import time
import torch
import pandas as pd
import PIL.Image
import time

import utilities
import model_spatial
import optimize

# this is where the saved model files is located
texture_synth_root = os.path.dirname(os.getcwd())

model_path = os.path.join(texture_synth_root, 'models','VGG19_normalized_avg_pool_pytorch')
        
# orig_stim_path = '/user_data/mmhender/stimuli/featsynth/images_comb64_orig'
# save_stim_path = '/user_data/mmhender/stimuli/featsynth/images_comb64'

save_path = '/user_data/mmhender/stimuli/featsynth/matrices_fornoise/'

ecoset_path = '/lab_data/tarrlab/common/datasets/Ecoset/'
ecoset_info_path = '/user_data/mmhender/stimuli/ecoset_info/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# import synthesize_textures

def make_ims(args):
    
    st_overall = time.time()
    
    if args.debug:
        print('\nDEBUG MODE\n')
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    fn = os.path.join(ecoset_info_path, 
                  'ecoset_files_use_fornoise.npy')
    info = np.load(fn, allow_pickle=True).item()
    
    bnames = list(info.keys())

    print('%d images per categ'%len(info[bnames[0]]['images']))

    enames_fn = os.path.join(ecoset_info_path, 'ecoset_names.npy')
    enames = np.load(enames_fn, allow_pickle=True).item()

    n_batches = 4;
    batch_size = np.ceil(len(bnames)/n_batches)
    batches = [np.arange(bb*batch_size, np.min([(bb+1)*batch_size, len(bnames)])).astype(int) for bb in range(n_batches)]
    
    ims_process = np.arange(args.n_ims_do)

    n_images = len(batches[args.batch_number]) * args.n_ims_do

    # im_seed = 867878

    
    # batches = [np.arange(0, 16), np.arange(16, 32), np.arange(32, 48), np.arange(48, 64)]

    layers_all = ['relu1_1', 'pool1','pool2','pool3','pool4']

    sizes = [64,64,128,256,512];

    gms_all = [np.zeros([s,s,n_images],dtype=np.float32) for s in sizes]

    print([g.shape for g in gms_all])
    
    xx = -1
    
    for bi in batches[args.batch_number]:

        bname = bnames[bi]

        # labs = image_list.iloc[np.array(image_list['basic_name']==bname)]
      
        if args.debug and (bi>1):
            continue

        for ii in ims_process:

            if args.debug and (ii>1):
                continue

            target_image_filename = os.path.join(ecoset_path, 'train', \
                                                 enames[bname], info[bname]['images'][ii])

            print('processing target image %s'%target_image_filename)
            sys.stdout.flush()

            st = time.time()
            # load target image to get gram matrix
            target_image = utilities.preprocess_image(
                utilities.load_image(target_image_filename)
            )
            elapsed = time.time() - st
            print('took %.5f s to preproc image for synthesis'%elapsed)

            net = model_spatial.Model(model_path, device, target_image, \
                                  important_layers = layers_all, \
                                  spatial_weights_list = None, 
                                  layer_weights = [1e09 for l in layers_all], \
                                  do_sqrt = True)

            gm = net.gram_loss_hook.target_gram_matrices

            xx+=1
            for ll, g in enumerate(gm):
                gms_all[ll][:,:,xx] = g.to('cpu').numpy()

    for ll, g in enumerate(gms_all):

        save_fn = os.path.join(save_path, \
                               'gram_matrices_batch%d_%s.npy'%(args.batch_number, layers_all[ll]))
        print('saving to %s'%save_fn)
        print(g.shape)
        np.save(save_fn, g)

    elapsed = time.time() - st_overall
    print('\nTook %.5f s (%.2f min) to run entire script'%(elapsed, elapsed/60))

def get_gm_stats():

    gm_path = save_path

    batches_load = [0,1,2,3]
    layers_all = ['relu1_1', 'pool1','pool2','pool3','pool4']
    
    # gmeans = []
    # gstds = []
    
    # for layer in layers_all[0:1]:
    for layer in layers_all:
    
        gcat = []
        
        for bi in batches_load:
    
            fn = os.path.join(gm_path, 'gram_matrices_batch%d_%s.npy'%(bi, layer))
            print(fn)
            g = np.load(fn)
    
            gcat += [g]
    
        gcat = np.concatenate(gcat, axis=2)
        print(gcat.shape)
        
        gmean = np.mean(gcat, axis=2)
        gstd = np.std(gcat, axis=2)
        
        fn2save = os.path.join(gm_path, 'gm_mean_%s.npy'%layer)
        print(fn2save)
        np.save(fn2save, gmean)
    
        fn2save = os.path.join(gm_path, 'gm_std_%s.npy'%layer)
        print(fn2save)
        np.save(fn2save, gstd)
        
def get_rgb_stats(debug=False, n_ims_do = 20):

    fn = os.path.join(ecoset_info_path, 
                  'ecoset_files_use_fornoise.npy')
    info = np.load(fn, allow_pickle=True).item()
    
    bnames = list(info.keys())

    ims_process = np.arange(n_ims_do)


    print('%d images per categ'%len(info[bnames[0]]['images']))

    enames_fn = os.path.join(ecoset_info_path, 'ecoset_names.npy')
    enames = np.load(enames_fn, allow_pickle=True).item()

    targ_vals_rgb = dict([])

    r = []; g = []; b = [];

    for bi, bname in enumerate(bnames):

        if debug and (bi>1):
            continue

        print('%s, %d of %d'%(bname, bi, len(bnames)))
        for ii in ims_process:
    
            if debug and (ii>1):
                continue

            
            target_image_filename = os.path.join(ecoset_path, 'train', \
                                                 enames[bname], info[bname]['images'][ii])
    
            print('processing target image %s'%target_image_filename)
            sys.stdout.flush()
            target_img = utilities.load_image(target_image_filename)
        
            proc_size = target_img.size[0]
            target_img_numpy = utilities.preprocess_image_simple(target_img, \
                                                         new_size=proc_size)*255
    
            r += [target_img_numpy[:,:,0].ravel()]
            g += [target_img_numpy[:,:,1].ravel()]
            b += [target_img_numpy[:,:,2].ravel()]
             
    r = np.concatenate(r, axis=0)
    g = np.concatenate(g, axis=0)
    b = np.concatenate(b, axis=0)

    print(r.shape, g.shape, b.shape)
    # find the distribution of values in target images
    n_bins = 100
    print('computing hist values...')
    sys.stdout.flush()
    
    r_hist, r_bin_edges = np.histogram(r, bins=n_bins, density=True)

    targ_vals_rgb['r_hist'] = r_hist
    targ_vals_rgb['r_bin_edges'] = r_bin_edges

    g_hist, g_bin_edges = np.histogram(g, bins=n_bins, density=True)

    targ_vals_rgb['g_hist'] = g_hist
    targ_vals_rgb['g_bin_edges'] = g_bin_edges

    b_hist, b_bin_edges = np.histogram(b, bins=n_bins, density=True)

    targ_vals_rgb['b_hist'] = b_hist
    targ_vals_rgb['b_bin_edges'] = b_bin_edges

    fn2save = os.path.join(save_path, 'rgb_hist_all.npy')
    print(fn2save)
    np.save(fn2save, targ_vals_rgb)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--n_ims_do", type=int,default=40,
                    help="how many images to do?")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    
    parser.add_argument("--batch_number", type=int,default=0,
                    help="batch n out of 4?")

    args = parser.parse_args()

    args.debug=args.debug==1
    
    make_ims(args)

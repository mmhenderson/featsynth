import os
import sys
import numpy as np
import torch
import pandas as pd
import PIL
import time

# this is where the saved model files is located
texture_synth_root = os.path.dirname(os.getcwd())

import utilities
import model_spatial
import optimize_TEST as optimize

import model_spatial_combineimages
import optimize_combineimages

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# this code is originally from:
# https://github.com/honzukka/texture-synthesis-pytorch
# modified by MMH in 2022 

def make_textures_oneimage(target_image_filename, \
                        out_dir, \
                        layers_do = ['pool1','pool2','pool3','pool4'], 
                        all_layers = True, \
                        n_steps = 100, \
                        rndseed = None, 
                        save_loss = False, \
                        jitter_pix = None):

    """
    Run texture synthesis algorithm for one target image
    Make textures for layers up to each of the specified layers
    """
    
    model_path = os.path.join(texture_synth_root, 'models','VGG19_normalized_avg_pool_pytorch')

    class a:
        def __init__():
            return a
        
    args = a
    args.lr = 1.0
    args.max_iter = 20
    args.checkpoint_every = 1
    args.n_steps = n_steps
    args.do_sqrt = True
    if jitter_pix is None:
        args.jitter_pix = 0
    else:
        args.jitter_pix = jitter_pix

    print('jitter_pix = %d'%args.jitter_pix)
    # args.rndseed = rndseed


    st = time.time()
    target_image_tosave = utilities.preprocess_image_tosave(
        utilities.load_image(target_image_filename)
    )
    filename_save = os.path.join(out_dir, 'orig.png')
    print('saving image to %s'%filename_save)
    target_image_tosave.save(filename_save)
    elapsed = time.time() - st
    print('took %.5f s to preproc and save orig image'%elapsed)

    st = time.time()
    # load target image to use in synthesis (different preproc than above)
    target_image = utilities.preprocess_image(
        utilities.load_image(target_image_filename)
    )
    elapsed = time.time() - st
    print('took %.5f s to preproc image for synthesis'%elapsed)

    # this is a list of all the layers that we're considering here
    
    if all_layers:
        # for this we will consider every layer in the model
        layers_all = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
           'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
           'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4',
           'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
           'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
           'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4',
           'pool5']
    else:
        # not considering every feature map but just a key subset.
        # this is like the "~177k parameters" version in Gatys 2015.
        # doesn't seem  to make huge difference.
        layers_all = ['relu1_1', 'pool1','pool2','pool3','pool4']
        
    
    # loop over layers - making a different texture for each layer
    # each will match the statistics of that layer and the ones before.

    for ll in range(len(layers_do)):

        curr_layer = layers_do[ll]

        # current layer has to match one of the names in layers_all
        index = np.where(curr_layer==np.array(layers_all))[0][0]
        print(ll, index)
        
        # which layers to match now? grab all layers up to and including this one.
        layers_match = layers_all[0:index+1]
        
        print('making texture for layers:')
        print(layers_match)
        sys.stdout.flush()

        st = time.time()
        net = model_spatial.Model(model_path, device, target_image, \
                                  important_layers = layers_match, \
                                  spatial_weights_list = None, 
                                  layer_weights = [1e09 for l in layers_match], \
                                  do_sqrt = args.do_sqrt)

        # synthesize
        # args.rndseed += 1
        if rndseed is None:
            args.rndseed = None
        elif hasattr(rndseed, '__len__'):
            args.rndseed = rndseed[ll]
        else:
            args.rndseed = rndseed + ll + 1
            
        optimizer = optimize.Optimizer(net, args)
        result = optimizer.optimize()
        elapsed = time.time() - st
        print('took %.5f s to run synthesis'%elapsed)

        # save result
        final_image = utilities.postprocess_image(
            result, utilities.load_image(target_image_filename)
        )
        filename_save = os.path.join(out_dir, 'scramble_upto_%s.png'%(curr_layer))
        print('saving image to %s'%filename_save)
        final_image.save(filename_save)
        sys.stdout.flush()

        if save_loss:
            filename_save = os.path.join(out_dir, 'loss_scramble_upto_%s.csv'%(curr_layer))
            print('saving loss to %s'%filename_save)
            loss_df = pd.DataFrame({'loss': optimizer.losses})
            loss_df.to_csv(filename_save)

            
def make_textures_combineimages(target_image_filenames, \
                                out_dir, \
                                layers_do = ['pool1','pool2','pool3','pool4'], 
                                n_steps = 100, \
                                rndseed = None, 
                                save_loss = False):

    """
    Run texture synthesis algorithm for a list of target images
    Make textures for layers up to each of the specified layers
    This version takes a list of images, and will create a texture that reflects 
    the average of those images (in gram matrix space)
    """
    
    model_path = os.path.join(texture_synth_root, 'models','VGG19_normalized_avg_pool_pytorch')

    class a:
        def __init__():
            return a
        
    args = a
    args.lr = 1.0
    args.max_iter = 20
    args.checkpoint_every = 1
    args.n_steps = n_steps
    args.do_sqrt = True
    args.rndseed = rndseed

    # there are multiple original images here, so we are not going to 
    # create "orig.png" like for the single images
    
    st = time.time()
    # load target image to use in synthesis 
    # this is a list of however many images we want to merge here
    target_images = [utilities.preprocess_image(utilities.load_image(tname)) \
                        for tname in target_image_filenames]

    elapsed = time.time() - st
    print('took %.5f s to preproc image for synthesis'%elapsed)

    # loop over layers - making a different texture for each layer
    # each will match the statistics of that layer and the ones before.
    for ll in range(len(layers_do)):

        layers_match = layers_do[0:ll+1]

        print('making texture for layers:')
        print(layers_match)
        sys.stdout.flush()

        st = time.time()
        net = model_spatial_combineimages.Model(model_path, device, \
                                                target_images, \
                                  layers_all=layers_match, \
                                  spatial_weights_list = None, 
                                  do_sqrt = args.do_sqrt)

        # synthesize
        optimizer = optimize_combineimages.Optimizer(net, args)
        result = optimizer.optimize()
        elapsed = time.time() - st
        print('took %.5f s to run synthesis'%elapsed)

        # finalize the image
        # histogram matching gets done in the postprocessing function, 
        # this will ensure the colors/luminance are correct. 
        # here we are combining histograms across all the target images
        target_img_list = [utilities.load_image(tname) for tname in target_image_filenames]

        final_image = utilities.postprocess_image_multiple_targets(result, \
                                                                   target_img_list)

        # save result
        filename_save = os.path.join(out_dir, 'scramble_upto_%s.png'%(layers_match[ll]))
        print('saving image to %s'%filename_save)
        final_image.save(filename_save)
        sys.stdout.flush()

        if save_loss:
            filename_save = os.path.join(out_dir, 'loss_scramble_upto_%s.csv'%(layers_match[ll]))
            print('saving loss to %s'%filename_save)
            loss_df = pd.DataFrame({'loss': optimizer.losses})
            loss_df.to_csv(filename_save)

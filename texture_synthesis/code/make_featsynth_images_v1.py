import argparse
import os
import sys
import numpy as np

import torch
import matplotlib.pyplot as plt     # type: ignore

things_stim_path = '/user_data/mmhender/stimuli/things/'
save_stim_path = '/user_data/mmhender/stimuli/featsynth/images_v1'
texture_synth_root = os.path.dirname(os.getcwd())

import utilities
import model_spatial
import optimize
import pandas as pd
import PIL
import time

import things_utils

def make_ims(args):
    
    st_overall = time.time()
    
    if args.debug:
        print('\nDEBUG MODE\n')
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
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
    ims_process = np.arange(args.n_ims_do)
    
    for ca, categ_ind in enumerate(categ_process):
        categ = categ_names[categ_ind]
        
        for co, conc_ind in enumerate(conc_process):
            conc = concept_names_subsample[categ_ind][conc_ind]
            
            if args.debug and (ca>2 or co>1):
                continue
                
            for ii in ims_process:
        
                target_image_filename = things_utils.get_filename(categ, conc, ii)
    
                print('\nCATEG %d of %d, IMAGE %d\n'%(ca, len(categ_process), ii))
                print('processing target image %s'%target_image_filename)
                sys.stdout.flush()

                name = target_image_filename.split('/')[-1].split('.jpg')[0]
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
                        
                model_path = os.path.join(texture_synth_root, 'models','VGG19_normalized_avg_pool_pytorch')

                args.lr = 1.0
                args.max_iter = 20
                args.checkpoint_every = 1
                if args.debug:
                    args.n_steps = 1
                args.rndseed = None

                layer_names_uppercase = ['Conv1','MaxPool1','MaxPool2','MaxPool3','MaxPool4']
                n_layers = len(layer_names_uppercase)
                overlap_each_layer = []
                for ll in range(n_layers):

                    fn = os.path.join(texture_synth_root,'grid_overlap','vgg19_gridoverlap_grid%d_%dx%d_%s.npy'%(args.which_grid,
                                                                                           args.n_grid_eachside, 
                                                                                    args.n_grid_eachside, 
                                                                                    layer_names_uppercase[ll]))
                    print('loading overlap from %s'%fn)
                    overlap = np.load(fn)
                    overlap_each_layer.append(overlap)

                st = time.time()
                target_image_tosave = utilities.preprocess_image_tosave(utilities.load_image(target_image_filename))
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

                
                layers_do = [1,2,3,4]
                for ll in layers_do:

                    layers_match = important_layers[0:ll+1]
                    spatial_weights_use = overlap_each_layer[0:ll+1]
                    print('making texture for layers:')
                    print(layers_match)
                    sys.stdout.flush()

                    st = time.time()
                    net = model_spatial.Model(model_path, device, target_image, \
                                              important_layers=layers_match, \
                                              spatial_weights_list = spatial_weights_use, 
                                              do_sqrt = args.do_sqrt)

                    # synthesize
                    optimizer = optimize.Optimizer(net, args)
                    result = optimizer.optimize()
                    elapsed = time.time() - st
                    print('took %.5f s to run synthesis'%elapsed)

                    # save result
                    final_image = utilities.postprocess_image(
                        result, utilities.load_image(target_image_filename)
                    )
                    filename_save = os.path.join(out_dir, 'grid%d_%dx%d_upto_%s.png'%(args.which_grid, \
                                                                                      args.n_grid_eachside, \
                                                                                    args.n_grid_eachside, \
                                                                                   important_layers[ll]))
                    print('saving image to %s'%filename_save)
                    final_image.save(filename_save)
                    sys.stdout.flush()

                    if args.save_loss:
                        filename_save = os.path.join(out_dir, 'loss_grid%dx%d_upto_%s.csv'%(args.n_grid_eachside, \
                                                                                    args.n_grid_eachside, \
                                                                                   important_layers[ll]))
                        print('saving loss to %s'%filename_save)
                        loss_df = pd.DataFrame({'loss': optimizer.losses})
                        loss_df.to_csv(filename_save)

    elapsed = time.time() - st_overall
    print('\nTook %.5f s (%.2f min) to run entire script'%(elapsed, elapsed/60))
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--n_ims_do", type=int,default=10,
                    help="how many images to do?")
    parser.add_argument("--n_grid_eachside", type=int,default=2,
                    help="how many grid spaces per square side?")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    
    parser.add_argument("--save_loss", type=int,default=1,
                    help="want to save loss over time for each image? 1 for yes, 0 for no")
    parser.add_argument("--n_steps", type=int,default=100,
                    help="how many steps to do per image?")
    parser.add_argument("--do_sqrt", type=int,default=1,
                    help="take sqrt of overlap? 1 for yes, 0 for no")
    parser.add_argument("--which_grid", type=int,default=5,
                    help="which of the spatial grid methods to use?")

    args = parser.parse_args()

    args.debug=args.debug==1
    args.do_sqrt=args.do_sqrt==1
    
    make_ims(args)

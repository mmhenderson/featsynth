import sys, os
import numpy as np
import time
from sklearn import decomposition
import pandas as pd
import torch
import PIL

project_root = '/user_data/mmhender/featsynth/'
texture_synth_root = os.path.join(project_root, 'texture_synthesis')

# these are in the 'texture_synthesis' folder
sys.path.append(os.path.join(texture_synth_root, 'code'))
import utilities
import model_spatial

# from image_analysis import extract_resnet_features
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
    
def get_gram_matrices(debug=False, image_set_name = 'images_ecoset64', \
                      layer_do = 'all', grayscale=False):

    debug = debug==1
    grayscale = grayscale==1

    print('debug=%s, grayscale=%s'%(debug, grayscale))
    
    feat_path = os.path.join(project_root, 'features', 'gram_matrices')
        
    if debug:
        feat_path = os.path.join(feat_path,'DEBUG')

    if not os.path.exists(feat_path):
        os.makedirs(feat_path)

    # load images to process
    folder_images = os.path.join(project_root, 'features','raw')
    image_data_filename = os.path.join(folder_images, '%s_preproc.npy'%(image_set_name))
    print('loading images from %s'%image_data_filename)
    image_data = np.load(image_data_filename)
    n_images = image_data.shape[0]
    
    if grayscale:
        list_orig = os.path.join(folder_images, '%s_list.csv'%(image_set_name))
        list_new = os.path.join(folder_images, '%s_grayscale_list.csv'%(image_set_name))
        print('duplicating list from %s to %s'%(list_orig, list_new))
        d = pd.read_csv(list_orig)
        d.to_csv(list_new)
        
    
    # stuff about the model to get gram matrices from
    model_path = os.path.join(texture_synth_root, 'models','VGG19_normalized_avg_pool_pytorch')

    if layer_do=='all':     
        layers_process = ['pool1','pool2','pool3','pool4']
    else:
        layers_process = [layer_do]
    # how many features will we get from each layer? the number of maps squared
    n_feats = {'pool1': 4096, 'pool2': 16384, 'pool3': 65536, 'pool4': 262144}
    gram_features = [np.zeros((n_images, n_feats[l]), dtype=np.float32) for l in layers_process]
    
    for ii in range(n_images):

        if debug and ii>1:
            continue
            
        print('proc image %d of %d'%(ii, n_images))
        sys.stdout.flush()
        
        st = time.time()
        # get image into format the model is expecting
        im = PIL.Image.fromarray(np.moveaxis(image_data[ii].astype(np.uint8),[0],[2]))

        # print(len(np.unique(im.getdata())))
        
        if grayscale:
            print('convert image to grayscale')
            # converting to to grayscale
            # L = R * 299/1000 + G * 587/1000 + B * 114/1000
            # then back to RGB to get shape right
            im = im.convert('L').convert('RGB')
            # print(len(np.unique(im.getdata())))
        
        target_image = utilities.preprocess_image(im)
        elapsed = time.time() - st
        print('took %.5f s to preproc image for synthesis'%elapsed)
        sys.stdout.flush()
        
        # this is the same model used during texture synthesis procedure
        # but now we are not synthesizing, just getting target gram matrices
        net = model_spatial.Model(model_path, device, target_image, \
                                                          important_layers=layers_process, \
                                                          spatial_weights_list = None,\
                                                          do_sqrt = True)
        
        gram_mats = net.gram_loss_hook.target_gram_matrices
    
        for li, gm in enumerate(gram_mats):

            # the features are all elements of the gram matrix, unraveled
            gram_features[li][ii,:] = gm.detach().cpu().numpy().ravel()

    for li, feat_raw in enumerate(gram_features):

        if grayscale:
            feat_file_name = os.path.join(feat_path, \
                                          '%s_grayscale_gram_matrices_%s_raw.npy'%(image_set_name,\
                                                                   layers_process[li]))
        else:
            feat_file_name = os.path.join(feat_path, \
                                          '%s_gram_matrices_%s_raw.npy'%(image_set_name,\
                                                                   layers_process[li]))
        print('size of feat_raw is:')
        print(feat_raw.shape)
        print('saving to %s'%feat_file_name)
        np.save(feat_file_name, feat_raw)
        
def pca_gram_matrices(debug=False, image_set_name = 'images_ecoset', layer_do = 'all'):

    debug = debug==1
    
    if layer_do=='all':     
        layers_process = ['pool1','pool2','pool3','pool4']
    else:
        layers_process = [layer_do]
        
    n_comp_keep = 500;
    
    feat_path = os.path.join(project_root, 'features', 'gram_matrices')

    # now going to perfom PCA so that the features are smaller
    # for li, feat_raw in enumerate(gram_features):
    for li in range(len(layers_process)):
        
        feat_file_name_raw = os.path.join(feat_path, \
                                      '%s_gram_matrices_%s_raw.npy'%(image_set_name,\
                                                               layers_process[li]))
        print('loading from %s'%feat_file_name_raw)
        feat_raw = np.load(feat_file_name_raw, allow_pickle=True)
        
        # reduce the dimensionality of the activs here
        scores, ev = compute_pca(feat_raw, max_pc_to_retain = n_comp_keep)

        cum_ev = np.cumsum(ev)
        inds = np.array([10, 50, 100, 200, 499])
        print(inds, cum_ev[inds])
        
        n_keep = np.min([scores.shape[1], n_comp_keep])
        
        scores = scores[:,0:n_keep]

        feat_file_name = os.path.join(feat_path, \
                                      '%s_gram_matrices_%s_pca.npy'%(image_set_name,\
                                                               layers_process[li]))
        # feat_file_name = os.path.join(feat_path, \
        #                               '%s_gram_matrices_%s_pca_TEST2.npy'%(image_set_name,\
        #                                                        layers_process[li]))
        print('size of scores is:')
        print(scores.shape)
        print('saving to %s'%feat_file_name)
        np.save(feat_file_name, scores)
        
        print('deleting %s'%(feat_file_name_raw))
        os.remove(feat_file_name_raw)


def compute_pca(values, max_pc_to_retain=None):
    """
    Apply PCA to the data, return reduced dim data as well as weights, var explained.
    """
    n_features_actual = values.shape[1]
    n_trials = values.shape[0]
    
    if max_pc_to_retain is not None:        
        n_comp = np.min([np.min([max_pc_to_retain, n_features_actual]), n_trials])
    else:
        n_comp = np.min([n_features_actual, n_trials])
         
    print('Running PCA: original size of array is [%d x %d], dtype=%s'%\
          (n_trials, n_features_actual, values.dtype))
    sys.stdout.flush()
    t = time.time()
    if values.shape[0]<=5000:
        
        print('using full pca')
        pca = decomposition.PCA(n_components = n_comp, copy=False)
        unshuff_order = np.arange(values.shape[0]) 
        
    else:
        print('using incremental PCA w shuffling first')
        pca = decomposition.IncrementalPCA(n_components = n_comp, \
                                           copy=False, batch_size=5000)
        # shuffling data before this, because there is structure in the rows 
        # (superordinate groups) and this makes the resulting data weird
        shuff_order = np.random.permutation(values.shape[0])
        unshuff_order = np.argsort(shuff_order)
        values = values[shuff_order,:]
        
    scores = pca.fit_transform(values)    

    # if regular pca this does nothing
    # if incremental pca it undoes shuffling
    scores = scores[unshuff_order,:]
    
    elapsed = time.time() - t
    print('Time elapsed: %.5f'%elapsed)
    values = None            
   
    ev = pca.explained_variance_
    ev = ev/np.sum(ev)*100
    
    return scores,ev

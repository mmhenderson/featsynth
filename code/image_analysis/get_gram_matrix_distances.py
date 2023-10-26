import os, sys
import numpy as np
import pandas as pd
import scipy.stats
import scipy.spatial

import sklearn.linear_model

import time

project_root = '/user_data/mmhender/featsynth/'
ecoset_info_path = '/user_data/mmhender/stimuli/ecoset_info/'

# sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'code'))
from utils import stats_utils


def compute_distances_allims(image_set_name = 'images_things200', 
                      layers_process = ['pool1','pool2','pool3','pool4'], 
                      distance_metric='cosine',
                      n_comp_keep = 100):

    # load gram matrix features for each image
    feat_path = os.path.join(project_root, 'features', 'gram_matrices')
    
    feat_all = []

    for li in range(len(layers_process)):

        feat_file_name = os.path.join(feat_path, \
                                      '%s_gram_matrices_%s_pca.npy'%(image_set_name,\
                                                               layers_process[li]))
        
        print(feat_file_name)
        feat = np.load(feat_file_name)
        
        feat = feat[:,0:n_comp_keep]

        feat_all += [feat]

    feat_all = np.concatenate(feat_all, axis=1)
    feat = feat_all
    feat = scipy.stats.zscore(feat, axis=0)
      
    # load corresponding labels
    image_list_filename = os.path.join(project_root, 'features','raw', '%s_list.csv'%(image_set_name))
    labels = pd.read_csv(image_list_filename)
    
    basic_labels = np.array(labels['basic_index'])
    super_labels = np.array(labels['super_index'])
    n_super = len(np.unique(super_labels))
    n_basic = len(np.unique(basic_labels))
    n_basic_each_super = int(n_basic/n_super)
    super_inds_long = np.repeat(np.arange(n_super), n_basic_each_super)
    
    # within/across basic, all categories included
    within_b, across_b_all = get_within_across_distances(feat, basic_labels, \
                                                     distance_metric=distance_metric)
    
    # within/across basic, just within each superordinate categ
    across_b_within_s = np.zeros_like(across_b_all)
    for si in range(n_super):
        
        inds = super_labels==si
        
        wb, ab = get_within_across_distances(feat[inds,:], \
                                             basic_labels[inds], \
                                             distance_metric=distance_metric)
        
        assert(np.all(wb==within_b[super_inds_long==si])) # these are identical, check
        across_b_within_s[super_inds_long==si] = ab
        
    # within/across superordinate
    within_s, across_s = get_within_across_distances(feat, super_labels, \
                                                    distance_metric=distance_metric)
      
    return within_b, across_b_within_s, across_b_all, \
            within_s, across_s




def compute_distances_ecoset(image_set_name = 'images_ecoset64', 
                      layers_process = ['pool1','pool2','pool3','pool4'], 
                      distance_metric='cosine', \
                      n_per_categ = 100, \
                      n_comp_keep = 100):
    
    # load image features
    feat_path = os.path.join(project_root, 'features', 'gram_matrices')

    feat_all = []

    for li in range(len(layers_process)):

        feat_file_name = os.path.join(feat_path, \
                                      '%s_gram_matrices_%s_pca.npy'%(image_set_name,\
                                                               layers_process[li]))
        print(feat_file_name)
        feat = np.load(feat_file_name)

        feat = feat[:,0:n_comp_keep]

        feat_all += [feat]

    feat_all = np.concatenate(feat_all, axis=1)
    feat = feat_all
    feat = scipy.stats.zscore(feat, axis=0)
      
    # load corresponding labels
    image_list_filename = os.path.join(project_root, 'features','raw', '%s_list.csv'%(image_set_name))
    labels = pd.read_csv(image_list_filename)
    
    basic_labels = np.array(labels['basic_index'])
    super_labels = np.array(labels['super_index'])
    n_super = len(np.unique(super_labels))
    n_basic = len(np.unique(basic_labels))
    n_basic_each_super = int(n_basic/n_super)
    super_inds_long = np.repeat(np.arange(n_super), n_basic_each_super)
    
    n_images = labels.shape[0]
    
    # info about ecoset categories
    fn = os.path.join(ecoset_info_path, 'categ_use_ecoset.npy')
    info = np.load(fn, allow_pickle=True).item()
    basic_names = np.array(list(info['binfo'].keys()))
    super_names = np.array(list(info['sinfo'].keys()))
    
    # get basic-level separability:
    # first create a set of images that have the desired number per categ
    ims_use_subsample = np.zeros((n_images,),dtype=bool)
    
    for bi, bname in enumerate(basic_names):
        inds = np.where(np.array(labels['basic_name'])==bname)[0]
        if len(inds)>=n_per_categ:
            inds_use = np.random.choice(inds, n_per_categ, replace=False)
        else:
            if bi==0:
                # usually this shouldn't happen
                print('warning: there are only %d trials to sample %d from'%(len(inds), n_per_categ))
            inds_use = inds
            
        ims_use_subsample[inds_use] = True     
            
    print('total ims subsampled for basic: %d'%(np.sum(ims_use_subsample)))
    
    # within/across basic, all categories included
    within_b, across_b_all = get_within_across_distances(feat[ims_use_subsample,:], \
                                                         basic_labels[ims_use_subsample], \
                                                     distance_metric=distance_metric)
    
    # within/across basic, just within each superordinate categ
    across_b_within_s = np.zeros_like(across_b_all)
    for si in range(n_super):
        
        inds = ims_use_subsample & (super_labels==si)
        
        wb, ab = get_within_across_distances(feat[inds,:], \
                                             basic_labels[inds], \
                                             distance_metric=distance_metric)
        # within-basic should be identical with these two methods
        assert(np.all(wb==within_b[super_inds_long==si])) 
        across_b_within_s[super_inds_long==si] = ab
     
    # get super-level separability:
    # first create a set of images that have the desired number per categ
    ims_use_subsample = np.zeros((n_images,),dtype=bool)
    
    # n_per_categ is how many we want total per superordinate categ.
    # want to divide these evenly across the basics
    n_per_basic = int(np.ceil(n_per_categ/n_basic_each_super))
   
    for sname in super_names:
        for bname in info['sinfo'][sname]['basic_names']:

            inds = np.where((np.array(labels['super_name'])==sname) & \
                            (np.array(labels['basic_name'])==bname))[0]
            
            if len(inds)>=n_per_basic:
                inds_use = np.random.choice(inds, n_per_basic, replace=False)
            else:
                inds_use = inds
            ims_use_subsample[inds_use] = True  
        
    print('total ims subsampled for super: %d'%(np.sum(ims_use_subsample)))
    
    # within/across superordinate
    within_s, across_s = get_within_across_distances(feat[ims_use_subsample,:], \
                                                     super_labels[ims_use_subsample], \
                                                    distance_metric=distance_metric)
     
    return within_b, across_b_within_s, across_b_all, \
            within_s, across_s

def get_ecoset_dist_vary_nums(image_set_name = 'images_ecoset64', \
                              debug=0):
    
    debug=debug==1
    
    feat_path = os.path.join(project_root, 'features', 'gram_matrices')

    save_dir = os.path.join(feat_path, 'distances')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fn2save = os.path.join(save_dir, 'cosine_distances_%s.npy'%(image_set_name))
    print('will save to %s'%fn2save)
    
    fn = os.path.join(ecoset_info_path, 'categ_use_ecoset.npy')
    info = np.load(fn, allow_pickle=True).item()
    n_basic = len(info['binfo'].keys())
    n_super = len(info['sinfo'].keys())
    
    layer_names = ['pool1','pool2', 'pool3','pool4']
    n_layers = len(layer_names)

    n_ims_vals =np.arange(8, 500, 8)
    n_v = len(n_ims_vals)

    within_b = np.zeros((n_basic, n_layers, n_v))
    across_b_within_s = np.zeros((n_basic, n_layers, n_v))
    across_b_all = np.zeros((n_basic, n_layers, n_v))

    within_s = np.zeros((n_super, n_layers, n_v))
    across_s = np.zeros((n_super, n_layers, n_v))

    for ni, nn in enumerate(n_ims_vals):

        for li, ll in enumerate(layer_names):

            distance_metric='cosine'
            
            print([nn, ll])
            sys.stdout.flush()
    
            wb, acb, acba, ws, acs = \
                    compute_distances_ecoset(image_set_name = image_set_name, \
                                                                distance_metric=distance_metric, \
                                                                layers_process = [ll], \
                                                                n_per_categ=nn, \
                                                               )
            within_b[:, li, ni] = wb
            across_b_within_s[:, li, ni] = acb
            across_b_all[:, li, ni] = acba
            within_s[:, li, ni] = ws
            across_s[:, li, ni] = acs
    
    print('saving to %s'%fn2save)
    np.save(fn2save, {'within_b': within_b, \
                     'across_b_within_s': across_b_within_s, \
                     'across_b_all': across_b_all, \
                     'within_s': within_s, \
                     'across_s': across_s})
    
    
def get_within_across_distances(feat, labels, distance_metric='cosine'):
    
    assert(feat.shape[0]==len(labels))
    
    un, counts = np.unique(labels, return_counts=True)
    assert(np.all(counts==counts[0])) # checking the groups are even
    
    n_labels = len(un)
    within_dist = np.zeros((n_labels,))
    across_dist = np.zeros((n_labels,))
    
    for li, lab in enumerate(un):
        
        f1 = feat[labels==lab]
        f2 = feat[labels!=lab]
        
        # within-group distances
        d1 = scipy.spatial.distance.pdist(f1, metric=distance_metric)
    
        # across-group distances
        d2 = scipy.spatial.distance.cdist(f1, f2, metric=distance_metric)
    
        within_dist[li] = np.mean(d1.ravel())
        across_dist[li] = np.mean(d2.ravel())
        
    return within_dist, across_dist




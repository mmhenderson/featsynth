import os, sys
import numpy as np
import pandas as pd
import scipy.stats
import scipy.spatial

import sklearn.linear_model

import time

project_root = '/user_data/mmhender/featsynth/'
# sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'code'))
from utils import stats_utils


def compute_distances(image_set_name = 'images_expt1', distance_metric='cosine'):
    
    # load image features
    # simclr model
    
    feat_path = os.path.join(project_root, 'features', 'simclr')
    training_type='simclr'
    feat_all = []

    for ll in [2,6,12,15]:

        feat_file_name = os.path.join(feat_path, \
                                              '%s_%s_block%d_pca.npy'%(image_set_name,\
                                                                       training_type, \
                                                                       ll))
        print(feat_file_name)
        feat = np.load(feat_file_name)

        feat_all += [feat]

    feat_all = np.concatenate(feat_all, axis=1)
    feat = feat_all
    feat = scipy.stats.zscore(feat, axis=0)
      
    # load corresponding labels
    image_list_filename = os.path.join(project_root, 'features','raw', '%s_list.csv'%(image_set_name))
    labels = pd.read_csv(image_list_filename)
    
    # figure out some image/category properties here
    n_ims_each = np.sum(np.array(labels['basic_name'])==np.array(labels['basic_name'])[0])
    basic_names = np.array(labels['basic_name'][0::n_ims_each])
    super_names_long = np.array(labels['super_name'][0::n_ims_each])
    basic_inds = np.array(labels['basic_index'][0::n_ims_each])
    super_inds_long = np.array(labels['super_index'][0::n_ims_each])
    n_basic = len(basic_names)
    n_super = len(np.unique(super_names_long))
    n_basic_each_super  = int(n_basic/n_super)
    super_names = super_names_long[0::n_basic_each_super]
    super_cbinds = np.repeat(np.array([0,1]), n_basic_each_super)
    super_inds = np.arange(n_super)

    # more image properties to organize images
    image_type_names = ['pool1','pool2','pool3','pool4','orig']
    n_image_types = len(image_type_names)
    cue_level_names = ['basic','super']
    

    # within and across basic-categs
    within_b = np.zeros((n_basic, n_image_types))
    across_b_within_s = np.zeros((n_basic, n_image_types))
    across_b_all = np.zeros((n_basic, n_image_types))

    for ii, imtype in enumerate(image_type_names):

        for bi in range(n_basic):

            inds = (np.array(labels['basic_index'])==bi) & (np.array(labels['image_type'])==imtype)
            f1 = feat[inds,:]
    
            # comparing images from the same basic-level category
            d1 = scipy.spatial.distance.pdist(f1, metric=distance_metric)
    
            # average across all pairwise comparisons
            within_b[bi,ii] = np.mean(d1.ravel())
    
            # now find all images that are in same super-ordinate category but different basic-level
            si = super_inds_long[bi]
            inds2 = (labels['basic_index']!=bi) & (labels['super_index']==si)  & \
                    (np.array(labels['image_type'])==imtype)
    
            f2 = feat[inds2,:]
    
            # comparing images from same superordinate, different basic-level
            d2 = scipy.spatial.distance.cdist(f1, f2, metric=distance_metric)
    
            # average over all pairwise comparisons
            across_b_within_s[bi,ii] = np.mean(d2.ravel())
    
            # find all images that are in any different basic-level (can be different super too)
            inds3 = (labels['basic_index']!=bi) & (np.array(labels['image_type'])==imtype)
    
            f3 = feat[inds3,:]
    
            # comparing images from same superordinate, different basic-level
            d3 = scipy.spatial.distance.cdist(f1, f3, metric=distance_metric)
    
            # average over all pairwise comparisons
            across_b_all[bi,ii] = np.mean(d3.ravel())
    

    # now across/within super-categs
    
    n_samples = 10
    
    within_s = np.zeros((n_super, n_image_types, n_samples))
    across_s = np.zeros((n_super, n_image_types, n_samples))
    
    # use the exemplar labels to sub-sample trials (match number available for basic-level)
    sample_labels = np.mod(np.array(labels['exemplar_number']), 10)

    
    for ii, imtype in enumerate(image_type_names):
    
        for si in range(n_super):
    
            for sa in range(n_samples):
                    
                inds =  (np.array(labels['super_index'])==si)  & \
                        (np.array(labels['image_type'])==imtype) & \
                        (sample_labels==sa)
                
                f = feat[inds,:]
                # print(f.shape)
        
                # comparing images from the same super-level category
                d = scipy.spatial.distance.pdist(f, metric=distance_metric)
        
                # average across all pairwise comparisons
                within_s[si,ii,sa] = np.mean(d.ravel())
        
                # different superordinate categ, but same image set (groups of 10 super categ)
                inds2 = (np.array(labels['super_index'])!=si) & \
                        (super_cbinds[np.array(labels['super_index'])]==super_cbinds[si]) & \
                        (np.array(labels['image_type'])==imtype) & \
                        (sample_labels==sa)
        
                f2 = feat[inds2,:]
                # print(f2.shape)
        
                # comparing images from diff super-level
                d = scipy.spatial.distance.cdist(f, f2, metric=distance_metric)
        
                # average over all pairwise comparisons
                across_s[si,ii,sa] = np.mean(d.ravel())
    
    within_s = np.mean(within_s, axis=2)
    across_s = np.mean(across_s, axis=2)
    
    return within_b, across_b_within_s, across_b_all, \
            within_s, across_s




def get_discrim(image_set_name = 'images_expt1', shuffle=0, subsample_super=0):
    
    shuffle = shuffle==1
    subsample_super = subsample_super==1
    # load image features
    # simclr model
    
    print(image_set_name, shuffle)
    
    feat_path = os.path.join(project_root, 'features', 'simclr')
    training_type='simclr'
    feat_all = []

    for ll in [2,6,12,15]:

        feat_file_name = os.path.join(feat_path, \
                                              '%s_%s_block%d_pca.npy'%(image_set_name,\
                                                                       training_type, \
                                                                       ll))
        print(feat_file_name)
        feat = np.load(feat_file_name)

        feat_all += [feat]

    feat_all = np.concatenate(feat_all, axis=1)
    feat = feat_all
    feat = scipy.stats.zscore(feat, axis=0)
    
    save_dir = os.path.join(feat_path, 'categ_discrim')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if shuffle:
        fn2save = os.path.join(save_dir, 'categ_discrim_%s_shuffle.npy'%image_set_name)
        if subsample_super:
            fn2save = os.path.join(save_dir, 'categ_discrim_%s_shuffle_subsample_super_method2.npy'%image_set_name)
    else:
        fn2save = os.path.join(save_dir, 'categ_discrim_%s.npy'%image_set_name)
        if subsample_super:
            fn2save = os.path.join(save_dir, 'categ_discrim_%s_subsample_super_method2.npy'%image_set_name)
            
    print('will save to %s'%fn2save)
    
    # load corresponding labels
    image_list_filename = os.path.join(project_root, 'features','raw', '%s_list.csv'%(image_set_name))
    labels = pd.read_csv(image_list_filename)
    # imfns_raw = np.array(labels['image_filename'])
    # if 'images_v1_grayscale' in imfns_raw[0]:
    #     imfns_raw = [imfn.split('/images_v1_grayscale/')[1] for imfn in imfns_raw]
    # else:
    #     imfns_raw = [imfn.split('/images_v1/')[1] for imfn in imfns_raw]

    # figure out some image/category properties here
    n_ims_each = np.sum(np.array(labels['basic_name'])==np.array(labels['basic_name'])[0])
    basic_names = np.array(labels['basic_name'][0::n_ims_each])
    super_names_long = np.array(labels['super_name'][0::n_ims_each])
    basic_inds = np.array(labels['basic_index'][0::n_ims_each])
    super_inds_long = np.array(labels['super_index'][0::n_ims_each])
    n_basic = len(basic_names)
    n_super = len(np.unique(super_names_long))
    n_basic_each_super  = int(n_basic/n_super)
    super_names = super_names_long[0::n_basic_each_super]
    super_cbinds = np.repeat(np.array([0,1]), n_basic_each_super)
    super_inds = np.arange(n_super)
    n_exemplars = 10;
    
    # more image properties to organize images
    image_type_names = ['pool1','pool2','pool3','pool4','orig']
    n_image_types = len(image_type_names)
    cue_level_names = ['basic','super']
    
    n_image_sets = 2;

    # acc_super_overall = np.zeros((n_image_types, n_image_sets))
    # dprime_super_overall = np.zeros((n_image_types, n_image_sets))
    acc_super_overall = np.zeros((n_image_types, n_image_sets, n_exemplars))
    dprime_super_overall = np.zeros((n_image_types, n_image_sets, n_exemplars))
    
    acc_basic_overall = np.zeros((n_image_types, n_super))
    dprime_basic_overall = np.zeros((n_image_types, n_super))

    # acc_each_supcat = np.zeros((n_super, n_image_types))
    acc_each_supcat = np.zeros((n_super, n_image_types, n_exemplars))
    acc_each_bascat = np.zeros((n_basic, n_image_types))
    # dprime_each_supcat = np.zeros((n_super, n_image_types))
    dprime_each_supcat = np.zeros((n_super, n_image_types, n_exemplars))
    dprime_each_bascat = np.zeros((n_basic, n_image_types))

    # loop over types of images [scrambled, intact...]
    for ii, imtype in enumerate(image_type_names):

        # doing superordinate classification, for each of the two image sets 
        # (cb indexes these)
        for cbi, cb in enumerate([1,2]):

            for ei,ee in enumerate(np.arange(1,n_exemplars+1)):
    
                if subsample_super:
                    # get images for this group
                    # inds = (labels['image_type']==imtype) & \
                    #         np.isin(np.array(labels['super_index']), super_inds[super_cbinds==cbi]) & \
                    #         (np.mod(labels['basic_index'], n_basic_each_super)==0)
                    inds = (labels['image_type']==imtype) & \
                            np.isin(np.array(labels['super_index']), super_inds[super_cbinds==cbi]) & \
                            (labels['exemplar_number']==ee)
                    cv_labs = np.mod(np.array(labels['basic_index']), n_basic_each_super)
                    cv_labs = cv_labs[inds]
    
                else:
                    
                    # get images for this group
                    inds = (labels['image_type']==imtype) & \
                            np.isin(np.array(labels['super_index']), super_inds[super_cbinds==cbi])
                    cv_labs = np.array(labels['exemplar_number'])[inds]
    
                    
                labs_use = np.array(labels['super_index'])[inds]
                feat_use = feat[inds,:]
                
                if (ii==0) & (cbi==0):
                    print(np.sum(inds))
                    print(cv_labs)
                    
                if shuffle:
                    tmp = np.zeros_like(labs_use)
                    for cv in np.unique(cv_labs):
                        cvinds = cv_labs==cv
                        tmp[cvinds] = labs_use[cvinds][np.random.permutation(np.sum(cvinds))]
                    labs_use = tmp
                
                # run multi-class classifier
                pred_labs = logreg_clf(feat_use, labs_use, cv_labs)
                assert(not np.any(np.isnan(pred_labs)))
                a = np.mean(pred_labs==labs_use)
                acc_super_overall[ii, cbi, ei] = a
                d = stats_utils.get_dprime(pred_labs, labs_use)
                dprime_super_overall[ii, cbi, ei] = d
                print('image set %d, %s, super acc=%.2f, super dprime=%.2f'%(cb, imtype, a, d))
                sys.stdout.flush()
                
                # get accuracy for each individual super-category
                sinds = super_inds[super_cbinds==cbi]
                for si in sinds: 
                    inds = labs_use==si
                    if not subsample_super:
                        assert(np.sum(inds)==100)
                    acc_each_supcat[si,ii,ei] = np.mean(pred_labs[inds]==labs_use[inds])
    
                    # getting d-prime for each category
                    # this is actually using all trials - but only measuring performance 
                    # based on whether the presence/absence of this categ was correct.
                    # convert the labels to binary yes/no
                    pred_tmp = np.zeros(np.shape(pred_labs))
                    labs_tmp = np.zeros(np.shape(labs_use))
                    pred_tmp[pred_labs==si] = 1
                    pred_tmp[pred_labs!=si] = 0
                    labs_tmp[labs_use==si] = 1
                    labs_tmp[labs_use!=si] = 0
    
                    dprime_each_supcat[si,ii,ei] = stats_utils.get_dprime(pred_tmp, labs_tmp)

        # doing basic classification, within each super category
        for si, supname in enumerate(super_names):

            # get images for this group
            inds = (labels['image_type']==imtype) & (labels['super_name']==supname)

            labs_use = np.array(labels['basic_index'])[inds]
            # print(labs_use)
            # print(np.unique(labs_use, return_counts=True))
           
            feat_use = feat[inds,:]
            cv_labs = np.array(labels['exemplar_number'])[inds]

            if shuffle:
                tmp = np.zeros_like(labs_use)
                for cv in np.unique(cv_labs):
                    cvinds = cv_labs==cv
                    tmp[cvinds] = labs_use[cvinds][np.random.permutation(np.sum(cvinds))]
                labs_use = tmp
            
            # run multi-class classifier
            pred_labs = logreg_clf(feat_use, labs_use, cv_labs).astype(int)
            
            assert(not np.any(np.isnan(pred_labs)))
            a = np.mean(pred_labs==labs_use)
            acc_basic_overall[ii, si] = a
            d = stats_utils.get_dprime(pred_labs, labs_use)
            dprime_basic_overall[ii, si] = d
            print('%s, %s, basic acc=%.2f, basic dprime=%.2f'%(supname, imtype, a, d))
            sys.stdout.flush()
            
            # get accuracy for each individual basic-category
            binds = basic_inds[super_inds_long==si]
            for bi in binds: 

                # get accuracy for just these trials
                inds = labs_use==bi
                assert(np.sum(inds)==10)
                acc_each_bascat[bi,ii] = np.mean(pred_labs[inds]==labs_use[inds])

                # getting d-prime for each category
                # this is actually using all trials - but only measuring performance 
                # based on whether the presence/absence of this categ was correct.
                # convert the labels to binary yes/no
                pred_tmp = np.zeros(np.shape(pred_labs))
                labs_tmp = np.zeros(np.shape(labs_use))
                pred_tmp[pred_labs==bi] = 1
                pred_tmp[pred_labs!=bi] = 0
                labs_tmp[labs_use==bi] = 1
                labs_tmp[labs_use!=bi] = 0

                dprime_each_bascat[bi,ii] = stats_utils.get_dprime(pred_tmp, labs_tmp)
    
    print('saving to %s'%fn2save)
    np.save(fn2save, {'acc_super_overall': acc_super_overall, \
                    'dprime_super_overall': dprime_super_overall, \
                    'acc_basic_overall': acc_basic_overall, \
                    'dprime_basic_overall': dprime_basic_overall, \
                    'acc_each_supcat': acc_each_supcat, \
                    'acc_each_bascat': acc_each_bascat, \
                    'dprime_each_supcat': dprime_each_supcat, \
                    'dprime_each_bascat': dprime_each_bascat})
    
            
def logreg_clf(feat, labs, cv_labs):

    pred_labs = np.full(fill_value=np.nan, shape=cv_labs.shape)

    for cvi, cv in enumerate(np.unique(cv_labs)):

        trninds = cv_labs!=cv
        tstinds = cv_labs==cv

        cs = np.logspace(-10, 0, 16)
        try:
            clf = sklearn.linear_model.LogisticRegressionCV(multi_class='multinomial', \
                                                          Cs = cs, fit_intercept=True, \
                                                          penalty = 'l2', \
                                                          refit=True, \
                                                          max_iter=10000, \
                                                         )
            clf.fit(feat[trninds,:], labs[trninds])
            print('cv %d: best C is %.8f'%(cvi, clf.C_[0]))

            p = clf.predict(feat[tstinds,:])

            pred_labs[tstinds] = p
            
        except:
            print('WARNING: problem with classifer, returning nans')
            pred_labs[tstinds] = np.nan
            
    return pred_labs




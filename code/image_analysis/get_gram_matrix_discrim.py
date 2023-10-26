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


def compute_discrim_ecoset(image_set_name = 'images_ecoset64', \
                           layer_process = 'pool1', \
                           debug = 0, \
                           n_per_categ = 100, \
                           n_cv = 10):
    
    debug=debug==1
    print('debug=%s'%debug)
    
    # load image features
    feat_path = os.path.join(project_root, 'features', 'gram_matrices')

    feat_all = []

    layers_process = [layer_process]
    print(layers_process)
    for li in range(len(layers_process)):

        feat_file_name = os.path.join(feat_path, \
                                      '%s_gram_matrices_%s_pca.npy'%(image_set_name,\
                                                               layers_process[li]))
        # feat_file_name = os.path.join(feat_path, \
                                      # '%s_gram_matrices_%s_pca_TEST.npy'%(image_set_name,\
                                                               # layers_process[li]))
        print(feat_file_name)
        feat = np.load(feat_file_name)

        feat_all += [feat]

    feat_all = np.concatenate(feat_all, axis=1)
    feat = feat_all
    feat = scipy.stats.zscore(feat, axis=0)
      
    save_dir = os.path.join(feat_path, 'categ_discrim')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # fn2save = os.path.join(save_dir, 'categ_discrim_%s_%s_%dpercateg_TEST.npy'%\
                           # (image_set_name, layer_process, n_per_categ))
    fn2save = os.path.join(save_dir, 'categ_discrim_%s_%s_%dpercateg.npy'%\
                           (image_set_name, layer_process, n_per_categ))
    print('will save to %s'%fn2save)
    
    
    # load corresponding labels for the images
    image_list_filename = os.path.join(project_root, 'features','raw', '%s_list.csv'%(image_set_name))
    labels = pd.read_csv(image_list_filename)
    n_images = labels.shape[0]
   
    # figure out some image/category properties here
    n_ims_each = np.sum(np.array(labels['basic_name'])==np.array(labels['basic_name'])[0])
    basic_names = np.array(labels['basic_name'][0::n_ims_each])
    super_names_long = np.array(labels['super_name'][0::n_ims_each])
   
    n_basic = len(basic_names)
    n_super = len(np.unique(super_names_long))
    n_basic_each_super  = int(n_basic/n_super)
    super_names = super_names_long[0::n_basic_each_super]
    
    # info about ecoset categories
    fn = os.path.join(ecoset_info_path, 'categ_use_ecoset.npy')
    info = np.load(fn, allow_pickle=True).item()
    
    super_acc = np.zeros((1,))
    super_dprime = np.zeros((1,))
    super_acc_each_supcat = np.zeros((n_super,))
    super_dprime_each_supcat = np.zeros((n_super,))
    
    basic_acc = np.zeros((n_super,))
    basic_dprime = np.zeros((n_super,))
    basic_acc_each_bascat = np.zeros((n_basic,))
    basic_dprime_each_bascat = np.zeros((n_basic,))

    # create a set of images that have the desired number per superord categ
    ims_use_subsample = np.zeros((n_images,),dtype=bool)

    # n_per_categ is how many we want total per superordinate categ.
    # want to divide these evenly across the basics
    n_per_basic = int(np.ceil(n_per_categ/n_basic_each_super))
   
    for sname in super_names:
        for bname in info['sinfo'][sname]['basic_names']:

            inds = np.where((np.array(labels['super_name'])==sname) & \
                            (np.array(labels['basic_name'])==bname))[0]
            print('there are %d images for %s: %s, choosing %d'%(len(inds), sname, bname, n_per_basic))
            inds_use = np.random.choice(inds, n_per_basic, replace=False)

            ims_use_subsample[inds_use] = True  
        
    print('total ims subsampled for super: %d'%(np.sum(ims_use_subsample)))

    # do superordinate classification
    inds = ims_use_subsample
   
    f = feat[inds,:]

    print('size of f:')
    print(f.shape)
    
    labs_use = np.array(labels['super_index'])[inds]

    sys.stdout.flush()
    pred_labs = logreg_clf(f, labs_use, cv_labs=None, n_cv=n_cv, debug=debug).astype(int)

    assert(not np.any(np.isnan(pred_labs)))
    a = np.mean(pred_labs==labs_use)
    super_acc[0] = a
    d = stats_utils.get_dprime(pred_labs, labs_use)
    super_dprime[0] = d
    print('super acc=%.2f, super dprime=%.2f'%(a, d))
    sys.stdout.flush()

    # get accuracy for each individual super-category
    for si in np.unique(labs_use):

        # accuracy just for images in this category
        inds = labs_use==si
        assert(np.sum(inds)==f.shape[0]/n_super)
        super_acc_each_supcat[si] = np.mean(pred_labs[inds]==labs_use[inds])

        # getting d-prime for each category
        # this is actually using all trials - but only measuring performance 
        # based on whether the presence/absence of this categ was correct.
        # convert the labels to binary yes/no
        pred_tmp = (pred_labs==si).astype('int')
        labs_tmp = (labs_use==si).astype('int')

        super_dprime_each_supcat[si] = stats_utils.get_dprime(pred_tmp, labs_tmp)


    # now do basic-level classification,  within each superord category separately
    for si, sname in enumerate(super_names):

        if debug and si>1:
            continue
            
        # create a set of images that have the desired number per categ
        ims_use_subsample = np.zeros((n_images,),dtype=bool)

        for bi, bname in enumerate(info['sinfo'][sname]['basic_names']):
            inds = np.where((np.array(labels['super_name'])==sname) & \
                            (np.array(labels['basic_name'])==bname))[0]
            print('there are %d images for %s: %s, choosing %d'%(len(inds), sname, bname, n_per_categ))
            inds_use = np.random.choice(inds, n_per_categ, replace=False)

            ims_use_subsample[inds_use] = True     

        print('total ims subsampled for basic within %s: %d'%(sname, np.sum(ims_use_subsample)))


        inds =  (np.array(labels['super_name'])==sname) & ims_use_subsample

        print(np.sum(inds))
        
        f = feat[inds,:]
        
        print('size of f:')
        print(f.shape)

        labs_use = np.array(labels['basic_index'])[inds]
        
        # run multi-class classifier
        pred_labs = logreg_clf(f, labs_use, cv_labs=None, n_cv=n_cv,  debug=debug).astype(int)

        assert(not np.any(np.isnan(pred_labs)))
        a = np.mean(pred_labs==labs_use)
        basic_acc[si] = a
        d = stats_utils.get_dprime(pred_labs, labs_use)
        basic_dprime[si] = d
        print('%s, basic acc=%.2f, basic dprime=%.2f'%(sname, a, d))
        sys.stdout.flush()

        # get accuracy for each individual basic-category
        for bi, basic_ind in enumerate(np.unique(labs_use)):

            print([si, bi, basic_ind])
            # accuracy just for images in this category
            inds = labs_use==basic_ind
            # print([np.sum(inds),f.shape[0],n_basic])
            assert(np.sum(inds)==f.shape[0]/n_basic_each_super)
            basic_acc_each_bascat[basic_ind] = np.mean(pred_labs[inds]==labs_use[inds])

            # getting d-prime for each category
            # this is actually using all trials - but only measuring performance 
            # based on whether the presence/absence of this categ was correct.
            # convert the labels to binary yes/no
            pred_tmp = (pred_labs==basic_ind).astype('int')
            labs_tmp = (labs_use==basic_ind).astype('int')

            basic_dprime_each_bascat[basic_ind] = stats_utils.get_dprime(pred_tmp, labs_tmp)


    print('saving to %s'%fn2save)
    np.save(fn2save, {'super_acc': super_acc, \
                    'basic_acc': basic_acc, \
                    'super_dprime': super_dprime, \
                    'basic_dprime': basic_dprime, \
                    'super_acc_each_supcat': super_acc_each_supcat, \
                    'basic_acc_each_bascat': basic_acc_each_bascat, \
                    'super_dprime_each_supcat': super_dprime_each_supcat, \
                    'basic_dprime_each_bascat': basic_dprime_each_bascat, 
                     })
    



def compute_discrim_ecoset_allbasic(image_set_name = 'images_ecoset64', \
                           layer_process = 'pool1', \
                           debug = 0, \
                           n_per_categ = 100, \
                           n_cv = 10):
    
    debug=debug==1
    print('debug=%s'%debug)
    
    # load image features
    feat_path = os.path.join(project_root, 'features', 'gram_matrices')

    feat_all = []

    layers_process = [layer_process]
    print(layers_process)
    for li in range(len(layers_process)):

        feat_file_name = os.path.join(feat_path, \
                                      '%s_gram_matrices_%s_pca.npy'%(image_set_name,\
                                                               layers_process[li]))
        print(feat_file_name)
        feat = np.load(feat_file_name)

        feat_all += [feat]

    feat_all = np.concatenate(feat_all, axis=1)
    feat = feat_all
    feat = scipy.stats.zscore(feat, axis=0)
      
    save_dir = os.path.join(feat_path, 'categ_discrim')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    fn2save = os.path.join(save_dir, 'categ_discrim_allbasic_%s_%s_%dpercateg.npy'%\
                           (image_set_name, layer_process, n_per_categ))
    print('will save to %s'%fn2save)
    
    
    # load corresponding labels for the images
    image_list_filename = os.path.join(project_root, 'features','raw', '%s_list.csv'%(image_set_name))
    labels = pd.read_csv(image_list_filename)
    n_images = labels.shape[0]
   
    # figure out some image/category properties here
    n_ims_each = np.sum(np.array(labels['basic_name'])==np.array(labels['basic_name'])[0])
    basic_names = np.array(labels['basic_name'][0::n_ims_each])
    super_names_long = np.array(labels['super_name'][0::n_ims_each])
   
    n_basic = len(basic_names)
    n_super = len(np.unique(super_names_long))
    n_basic_each_super  = int(n_basic/n_super)
    super_names = super_names_long[0::n_basic_each_super]
    
    # info about ecoset categories
    fn = os.path.join(ecoset_info_path, 'categ_use_ecoset.npy')
    info = np.load(fn, allow_pickle=True).item()
    
    basic_acc = np.zeros((n_super,))
    basic_dprime = np.zeros((n_super,))
    basic_acc_each_bascat = np.zeros((n_basic,))
    basic_dprime_each_bascat = np.zeros((n_basic,))

    # going to compute superordinate-level acc based on output of 
    # the 64-way basic-level classifier
    super_acc = np.zeros((1,))
    super_dprime = np.zeros((1,))
    super_acc_each_supcat = np.zeros((n_super,))
    super_dprime_each_supcat = np.zeros((n_super,))

    # compute superordinate-level accuracy for "random" set of superord categs
    # this is to test if the actual superord categs are adding structure
    n_rand = 1000
    super_acc_rand = np.zeros((1,n_rand))
    super_dprime_rand = np.zeros((1,n_rand))
    super_acc_each_supcat_rand = np.zeros((n_super,n_rand))
    super_dprime_each_supcat_rand = np.zeros((n_super,n_rand))

    # create a set of images that have the desired number per superord categ
    ims_use_subsample = np.zeros((n_images,),dtype=bool)

    # n_per_categ is how many we want total per superordinate categ.
    # want to divide these evenly across the basics
    n_per_basic = n_per_categ
   
    for bname in basic_names:

        inds = np.where((np.array(labels['basic_name'])==bname))[0]
        print('there are %d images for %s, choosing %d'%(len(inds), bname, n_per_basic))
        inds_use = np.random.choice(inds, n_per_basic, replace=False)

        ims_use_subsample[inds_use] = True  
        
    print('total ims subsampled for basic-all: %d'%(np.sum(ims_use_subsample)))

    # do basic-all classification
    inds = ims_use_subsample
   
    f = feat[inds,:]

    print('size of f:')
    print(f.shape)
    
    labs_use = np.array(labels['basic_index'])[inds]

    sys.stdout.flush()
    pred_labs = logreg_clf(f, labs_use, cv_labs=None, n_cv=n_cv, debug=debug).astype(int)

    assert(not np.any(np.isnan(pred_labs)))
    a = np.mean(pred_labs==labs_use)
    basic_acc[0] = a
    d = stats_utils.get_dprime(pred_labs, labs_use)
    basic_dprime[0] = d
    print('basic acc=%.2f, basic dprime=%.2f'%(a, d))
    sys.stdout.flush()

    # get accuracy for each individual basic-category
    for bi in np.unique(labs_use):

        # accuracy just for images in this category
        inds = labs_use==bi
        assert(np.sum(inds)==f.shape[0]/n_basic)
        basic_acc_each_bascat[bi] = np.mean(pred_labs[inds]==labs_use[inds])

        # getting d-prime for each category
        # this is actually using all trials - but only measuring performance 
        # based on whether the presence/absence of this categ was correct.
        # convert the labels to binary yes/no
        pred_tmp = (pred_labs==bi).astype('int')
        labs_tmp = (labs_use==bi).astype('int')

        basic_dprime_each_bascat[bi] = stats_utils.get_dprime(pred_tmp, labs_tmp)

    # mapping the basic-level labels into superordinate categs
    super_inds_long = np.repeat(np.arange(n_super), n_basic_each_super)

    pred_labs_super = super_inds_long[pred_labs]
    actual_labs_super = super_inds_long[labs_use]

    a = np.mean(pred_labs_super==actual_labs_super)
    super_acc[0] = a
    d = stats_utils.get_dprime(pred_labs_super, actual_labs_super)
    super_dprime[0] = d
    print('super acc=%.2f, super dprime=%.2f'%(a, d))
    sys.stdout.flush()
    
    # get accuracy for each individual super-category
    for si in np.unique(actual_labs_super):

        # accuracy just for images in this category
        inds = actual_labs_super==si
        assert(np.sum(inds)==f.shape[0]/n_super)
        super_acc_each_supcat[si] = np.mean(pred_labs_super[inds]==actual_labs_super[inds])

        # getting d-prime for each category
        # this is actually using all trials - but only measuring performance 
        # based on whether the presence/absence of this categ was correct.
        # convert the labels to binary yes/no
        pred_tmp = (pred_labs_super==si).astype('int')
        labs_tmp = (actual_labs_super==si).astype('int')

        super_dprime_each_supcat[si] = stats_utils.get_dprime(pred_tmp, labs_tmp)

    for ri in range(n_rand):
        
        # randomize the mapping
        super_inds_long_rand = super_inds_long[np.random.permutation(len(super_inds_long))]

        pred_labs_super = super_inds_long_rand[pred_labs]
        actual_labs_super = super_inds_long_rand[labs_use]

        a = np.mean(pred_labs_super==actual_labs_super)
        super_acc_rand[0,ri] = a
        d = stats_utils.get_dprime(pred_labs_super, actual_labs_super)
        super_dprime_rand[0,ri] = d
        print('super acc=%.2f, super dprime=%.2f'%(a, d))
        sys.stdout.flush()

        # get accuracy for each individual super-category
        for si in np.unique(actual_labs_super):

            # accuracy just for images in this category
            inds = actual_labs_super==si
            assert(np.sum(inds)==f.shape[0]/n_super)
            super_acc_each_supcat_rand[si,ri] = np.mean(pred_labs_super[inds]==actual_labs_super[inds])

            # getting d-prime for each category
            # this is actually using all trials - but only measuring performance 
            # based on whether the presence/absence of this categ was correct.
            # convert the labels to binary yes/no
            pred_tmp = (pred_labs_super==si).astype('int')
            labs_tmp = (actual_labs_super==si).astype('int')

            super_dprime_each_supcat_rand[si,ri] = stats_utils.get_dprime(pred_tmp, labs_tmp)

    
    print('saving to %s'%fn2save)
    np.save(fn2save, {'super_acc': super_acc, \
                    'basic_acc': basic_acc, \
                    'super_dprime': super_dprime, \
                    'basic_dprime': basic_dprime, \
                    'super_acc_each_supcat': super_acc_each_supcat, \
                    'basic_acc_each_bascat': basic_acc_each_bascat, \
                    'super_dprime_each_supcat': super_dprime_each_supcat, \
                    'basic_dprime_each_bascat': basic_dprime_each_bascat, 
                    'super_acc_rand': super_acc_rand, \
                    'super_dprime_rand': super_dprime_rand, \
                    'super_acc_each_supcat_rand': super_acc_each_supcat_rand, \
                    'super_dprime_each_supcat_rand': super_dprime_each_supcat_rand, \
                     })
    
                      
    
    
    
def logreg_clf(feat, labs, cv_labs=None, n_cv = 10, debug=False):

    
    if cv_labs is None:
        # making random cross-validation labels
        # balance classes as closely as possible
        cv_labs = np.zeros_like(labs)
        unvals = np.unique(labs)
        for uu in unvals:
            inds = np.where(labs==uu)[0]
            cv_tmp = np.tile(np.arange(n_cv), [int(np.ceil(len(inds)/n_cv)),])
            cv_tmp = cv_tmp[np.random.permutation(len(cv_tmp))][0:len(inds)]
            cv_labs[inds] = cv_tmp
    
    pred_labs = np.full(fill_value=np.nan, shape=cv_labs.shape)
    
    print(np.unique(cv_labs, return_counts=True))
    
    for cvi, cv in enumerate(np.unique(cv_labs)):

        if debug and cvi>1:
            continue
            
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
            sys.stdout.flush()

            p = clf.predict(feat[tstinds,:])

            pred_labs[tstinds] = p
            
        except:
            print('WARNING: problem with classifer, returning nans')
            pred_labs[tstinds] = np.nan
            
    return pred_labs





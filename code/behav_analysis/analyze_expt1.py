import os, sys
import numpy as np
import pandas as pd

project_root = '/user_data/mmhender/featsynth/'
sys.path.append(os.path.join(project_root, 'code'))

from utils import stats_utils

def get_categ_info(image_set_name = 'images_expt1'):
   
    # this is a list of all 10000 images, not all of which are shown in experiment.
    # using this to find names of all categories.
    # make in prep_images
    image_list_filename = os.path.join(project_root, 'features','raw', '%s_list.csv'%(image_set_name))
    labels = pd.read_csv(image_list_filename)
    n_ims_each = np.sum(np.array(labels['basic_name'])==np.array(labels['basic_name'])[0])
    basic_names = np.array(labels['basic_name'][0::n_ims_each])
    # assert(np.all(list(np.array(basic_names_eachset).ravel())==basic_names))
    super_names_long = np.array(labels['super_name'][0::n_ims_each])
    basic_inds = np.array(labels['basic_index'][0::n_ims_each])
    super_inds_long = np.array(labels['super_index'][0::n_ims_each])
    n_basic = len(basic_names)
    n_super = len(np.unique(super_names_long))
    n_basic_each_super  = int(n_basic/n_super)
    super_names = super_names_long[0::n_basic_each_super]
    super_cbinds = np.repeat(np.array([0,1]), n_basic_each_super)
    
    return super_names, super_cbinds, basic_names, basic_inds, \
        super_names_long, super_inds_long, n_basic, n_super, n_basic_each_super

super_names, super_cbinds, basic_names, basic_inds, \
        super_names_long, super_inds_long, n_basic, n_super, n_basic_each_super = \
            get_categ_info()

def save_categ_info():
    
    super_names, super_cbinds, basic_names, basic_inds, \
        super_names_long, super_inds_long, n_basic, n_super, n_basic_each_super = \
            get_categ_info()

    save_filename = os.path.join(project_root, 'code', 'make_expt_designs', 'expt1_categ_info.npy')
    np.save(save_filename, {'super_names': super_names, \
                           'super_cbinds': super_cbinds, \
                           'basic_names': basic_names, \
                           'basic_inds': basic_inds, \
                           'super_names_long': super_names_long, \
                           'super_inds_long': super_inds_long})
    
def load_data(expt_name = 'expt1'):

    expt_name = 'expt1'
    preproc_folder = os.path.join(project_root, 'online_data', expt_name, 'preproc')
    data_filename = os.path.join(preproc_folder, 'preproc_data_all.csv')
    trial_data_all = pd.read_csv(data_filename)
    subjects = np.unique(trial_data_all['subject'])
    n_subjects = len(subjects)
    subject_cb = np.array([np.array(trial_data_all['which_cb'])[trial_data_all['subject']==si][0] \
                  for si in subjects]) - 1
    # check that the super-category names align across behavioral data and image analyses
    for which_cb in [0,1]:
        subject_check = subjects[np.where(subject_cb==which_cb)[0][0]]
        trial_data = trial_data_all[trial_data_all['subject']==subject_check]
        scheck = np.unique(trial_data['super_name'])
        assert(np.all(scheck==super_names[super_cbinds==which_cb]))

    cue_level_names = np.unique(np.array(trial_data_all['cue_level']))
    
    image_type_names = np.unique(np.array(trial_data_all['image_type']))
    type_order_plot = [1,2,3,4,0]
    image_type_names = np.array(image_type_names)[type_order_plot]

    return trial_data_all, subjects, subject_cb, cue_level_names, image_type_names

trial_data_all, subjects, subject_cb, cue_level_names, image_type_names = load_data()
 
n_subjects = len(subjects)
n_cue_levels = len(cue_level_names)
n_image_types = len(image_type_names)

def get_perf_by_supercateg():

    # break trials down by conditions (cue level, image type) and
    # each super-category group
    acc_by_supcat = [np.full(fill_value = np.nan, \
                             shape = (np.sum(subject_cb==cb), n_cue_levels, \
                                      n_image_types, int(n_super/2))) \
                     for cb in [0,1]]
    dprime_by_supcat = [np.full(fill_value = np.nan, \
                             shape = (np.sum(subject_cb==cb), n_cue_levels, \
                                      n_image_types, int(n_super/2))) \
                     for cb in [0,1]]
    
    rt_by_supcat = [np.full(fill_value = np.nan, \
                             shape = (np.sum(subject_cb==cb), n_cue_levels, \
                                      n_image_types, int(n_super/2))) \
                     for cb in [0,1]]
    
    cb_sub_count = [-1,-1]

    for si, ss in enumerate(subjects):

        cbi = subject_cb[si]

        cb_sub_count[cbi]+=1

        trial_data = trial_data_all[trial_data_all['subject']==ss]

        supnames = super_names[super_cbinds==cbi]

        for sc, supcat in enumerate(supnames):

            for cc, cue in enumerate(cue_level_names):

                for ii, imtype in enumerate(image_type_names):

                    inds = (trial_data['cue_level']==cue) & \
                            (trial_data['image_type']==imtype) & \
                            (trial_data['super_name']==supcat)
                    assert(np.sum(inds)==10)

                    predlabs = np.array(trial_data['resp'])[inds]
                    reallabs = np.array(trial_data['correct_resp'])[inds]

                    acc = np.mean(predlabs==reallabs)
                    
                    did_respond = predlabs>-1

                    predlabs = predlabs[did_respond]
                    reallabs = reallabs[did_respond]
                    
                    dprime = stats_utils.get_dprime(predlabs, reallabs)
                    
                    rts = trial_data['rt'][inds]
                    rts_use = rts[did_respond]
            
                    acc_by_supcat[cbi][cb_sub_count[cbi],cc,ii,sc] = acc
                    dprime_by_supcat[cbi][cb_sub_count[cbi],cc,ii,sc] = dprime
                    rt_by_supcat[cbi][cb_sub_count[cbi],cc,ii,sc] = np.mean(rts_use)
                
    n_each_cb = np.array([np.sum(subject_cb==cb) for cb in [0,1]])
    assert(np.all(n_each_cb == np.array(cb_sub_count)+1))
    assert(not np.any([np.any(np.isnan(a)) for a in acc_by_supcat]))
    assert(not np.any([np.any(np.isnan(a)) for a in rt_by_supcat]))
    assert(not np.any([np.any(np.isnan(a)) for a in dprime_by_supcat]))
    
    return acc_by_supcat, dprime_by_supcat, rt_by_supcat

def get_perf_by_cued_supercateg():

    # break trials down by which super-category was cued
    # (regardless of what was visually shown)
    n_cue_levels = 1;
    cue = 'super'
    cc = 0
    
    acc_by_supcat = [np.full(fill_value = np.nan, \
                             shape = (np.sum(subject_cb==cb), n_cue_levels, \
                                      n_image_types, int(n_super/2))) \
                     for cb in [0,1]]
    dprime_by_supcat = [np.full(fill_value = np.nan, \
                             shape = (np.sum(subject_cb==cb), n_cue_levels, \
                                      n_image_types, int(n_super/2))) \
                     for cb in [0,1]]
    
    cb_sub_count = [-1,-1]

    for si, ss in enumerate(subjects):

        cbi = subject_cb[si]

        cb_sub_count[cbi]+=1

        trial_data = trial_data_all[trial_data_all['subject']==ss]

        supnames = super_names[super_cbinds==cbi]

        for sc, supcat in enumerate(supnames):


            for ii, imtype in enumerate(image_type_names):

                inds = (trial_data['cue_level']==cue) & \
                        (trial_data['image_type']==imtype) & \
                        (trial_data['cue_name']==supcat)
                assert(np.sum(inds)==10)

                predlabs = np.array(trial_data['resp'])[inds]
                reallabs = np.array(trial_data['correct_resp'])[inds]

                acc = np.mean(predlabs==reallabs)

                did_respond = predlabs>-1

                predlabs = predlabs[did_respond]
                reallabs = reallabs[did_respond]

                dprime = stats_utils.get_dprime(predlabs, reallabs)

                acc_by_supcat[cbi][cb_sub_count[cbi],cc,ii,sc] = acc
                dprime_by_supcat[cbi][cb_sub_count[cbi],cc,ii,sc] = dprime

    n_each_cb = np.array([np.sum(subject_cb==cb) for cb in [0,1]])
    assert(np.all(n_each_cb == np.array(cb_sub_count)+1))
    assert(not np.any([np.any(np.isnan(a)) for a in acc_by_supcat]))
    
    return acc_by_supcat, dprime_by_supcat


def get_perf_by_basiccateg():

    # not enough trial to separate these out further, have to average over conditions
    acc_by_bascat = [np.full(fill_value = np.nan, \
                             shape = (np.sum(subject_cb==cb), int(n_super/2), n_basic_each_super)) \
                     for cb in [0,1]]
    dprime_by_bascat = [np.full(fill_value = np.nan, \
                             shape = (np.sum(subject_cb==cb), int(n_super/2), n_basic_each_super)) \
                     for cb in [0,1]]

    cb_sub_count = [-1,-1]

    for si, ss in enumerate(subjects):

        cbi = subject_cb[si]

        cb_sub_count[cbi]+=1

        trial_data = trial_data_all[trial_data_all['subject']==ss]

        supnames = super_names[super_cbinds==cbi]

        for sc, supcat in enumerate(supnames):
            
            bnames = basic_names[super_names_long==supcat]
            
            for bb, bascat in enumerate(bnames):

                inds = (trial_data['super_name']==supcat) & (trial_data['basic_name']==bascat)
                assert(np.sum(inds)==10)

                predlabs = np.array(trial_data['resp'])[inds]
                reallabs = np.array(trial_data['correct_resp'])[inds]

                acc = np.mean(predlabs==reallabs)

                did_respond = predlabs>-1

                predlabs = predlabs[did_respond]
                reallabs = reallabs[did_respond]

                dprime = stats_utils.get_dprime(predlabs, reallabs)

                acc_by_bascat[cbi][cb_sub_count[cbi],sc,bb] = acc
                dprime_by_bascat[cbi][cb_sub_count[cbi],sc,bb] = dprime

    n_each_cb = np.array([np.sum(subject_cb==cb) for cb in [0,1]])
    assert(np.all(n_each_cb == np.array(cb_sub_count)+1))
    assert(not np.any([np.any(np.isnan(a)) for a in acc_by_bascat]))

    return acc_by_bascat, dprime_by_bascat


def get_perf_by_basiccateg_combinesubjects():

    # combine trials across subjects to get more stable estimates
    n_image_sets = 2;
    acc_by_bascat = np.full(fill_value = np.nan, \
                            shape = (n_image_sets, n_cue_levels, int(n_super/2), n_basic_each_super))
    dprime_by_bascat = np.full(fill_value = np.nan, \
                            shape = (n_image_sets, n_cue_levels, int(n_super/2), n_basic_each_super))

    
    for cbi, cb in enumerate([1,2]):
      
        trial_data = trial_data_all[trial_data_all['which_cb']==cb]

        supnames = super_names[super_cbinds==cbi]

        for sc, supcat in enumerate(supnames):
            
            bnames = basic_names[super_names_long==supcat]
            
            for bb, bascat in enumerate(bnames):
                
                for cc, cue_level in enumerate(cue_level_names):

                    inds = (trial_data['super_name']==supcat) & \
                            (trial_data['basic_name']==bascat) & \
                            (trial_data['cue_level']==cue_level)
                   
                    predlabs = np.array(trial_data['resp'])[inds]
                    reallabs = np.array(trial_data['correct_resp'])[inds]

                    acc = np.mean(predlabs==reallabs)

                    did_respond = predlabs>-1

                    predlabs = predlabs[did_respond]
                    reallabs = reallabs[did_respond]

                    dprime = stats_utils.get_dprime(predlabs, reallabs)

                    acc_by_bascat[cbi,cc,sc,bb] = acc
                    dprime_by_bascat[cbi,cc,sc,bb] = dprime

    assert(not np.any(np.isnan(acc_by_bascat)))

    return acc_by_bascat, dprime_by_bascat


def get_perf_by_cond():

    acc_by_condition = np.zeros((n_subjects, n_cue_levels, n_image_types))
    dprime_by_condition = np.zeros((n_subjects, n_cue_levels, n_image_types))
    propyes_by_condition = np.zeros((n_subjects, n_cue_levels, n_image_types))
    rt_by_condition = np.zeros((n_subjects, n_cue_levels, n_image_types))

    for si, ss in enumerate(subjects):

        trial_data = trial_data_all[trial_data_all['subject']==ss]

        for cc, cue in enumerate(cue_level_names):

            for ii, imtype in enumerate(image_type_names):

                inds = (trial_data['cue_level']==cue) & \
                        (trial_data['image_type']==imtype)
                assert(np.sum(inds)==100)

                predlabs = np.array(trial_data['resp'])[inds]
                reallabs = np.array(trial_data['correct_resp'])[inds]
                rts = np.array(trial_data['rt'])[inds]
                
                acc_by_condition[si,cc,ii] = np.mean(predlabs==reallabs)

                did_respond = predlabs>-1

                predlabs = predlabs[did_respond]
                reallabs = reallabs[did_respond]

                dprime_by_condition[si,cc,ii] = stats_utils.get_dprime(predlabs, reallabs)

                propyes_by_condition[si,cc,ii] = np.mean(predlabs==1)
                
                rt_by_condition[si,cc,ii] = np.mean(rts[did_respond])

    return acc_by_condition, dprime_by_condition, propyes_by_condition, rt_by_condition


def get_perf_by_cond_excludebirds():

    acc_by_condition = np.zeros((n_subjects, n_cue_levels, n_image_types))
    dprime_by_condition = np.zeros((n_subjects, n_cue_levels, n_image_types))
    propyes_by_condition = np.zeros((n_subjects, n_cue_levels, n_image_types))
    rt_by_condition = np.zeros((n_subjects, n_cue_levels, n_image_types))

    for si, ss in enumerate(subjects):

        trial_data = trial_data_all[trial_data_all['subject']==ss]

        for cc, cue in enumerate(cue_level_names):

            for ii, imtype in enumerate(image_type_names):

                inds = (trial_data['cue_level']==cue) & \
                        (trial_data['image_type']==imtype) & \
                        (trial_data['super_name']!='bird')
                
                # print(np.sum(inds))

                predlabs = np.array(trial_data['resp'])[inds]
                reallabs = np.array(trial_data['correct_resp'])[inds]
                rts = np.array(trial_data['rt'])[inds]
                
                acc_by_condition[si,cc,ii] = np.mean(predlabs==reallabs)

                did_respond = predlabs>-1

                predlabs = predlabs[did_respond]
                reallabs = reallabs[did_respond]

                dprime_by_condition[si,cc,ii] = stats_utils.get_dprime(predlabs, reallabs)

                propyes_by_condition[si,cc,ii] = np.mean(predlabs==1)
                
                rt_by_condition[si,cc,ii] = np.mean(rts[did_respond])

    return acc_by_condition, dprime_by_condition, propyes_by_condition, rt_by_condition


def get_perf_by_nat():

    # natural or artificial categories?
    is_natural = np.array([1,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1]).astype(bool)
    kind_names = ['Artificial','Natural']
    n_kinds = len(kind_names)
    
    acc_by_condition = np.zeros((n_subjects, n_cue_levels, n_image_types, n_kinds))
    dprime_by_condition = np.zeros((n_subjects, n_cue_levels, n_image_types, n_kinds))
    propyes_by_condition = np.zeros((n_subjects, n_cue_levels, n_image_types, n_kinds))
    rt_by_condition = np.zeros((n_subjects, n_cue_levels, n_image_types, n_kinds))

    for si, ss in enumerate(subjects):

        trial_data = trial_data_all[trial_data_all['subject']==ss]
        cbi = np.array(trial_data['which_cb'])[0] - 1

        for cc, cue in enumerate(cue_level_names):

            for ii, imtype in enumerate(image_type_names):

                for kk in range(n_kinds):

                    supcats_use = super_names[(is_natural==kk) & (super_cbinds==cbi)]
                    # print(kind_names[kk])
                    # print(supcats_use)

                    inds = (trial_data['cue_level']==cue) & \
                            (trial_data['image_type']==imtype) & \
                            np.isin(trial_data['super_name'], supcats_use)

                    # print(np.sum(inds))

                    predlabs = np.array(trial_data['resp'])[inds]
                    reallabs = np.array(trial_data['correct_resp'])[inds]
                    rts = np.array(trial_data['rt'])[inds]
                
                    acc_by_condition[si,cc,ii, kk] = np.mean(predlabs==reallabs)

                    did_respond = predlabs>-1

                    predlabs = predlabs[did_respond]
                    reallabs = reallabs[did_respond]

                    dprime_by_condition[si,cc,ii,kk] = stats_utils.get_dprime(predlabs, reallabs)

                    propyes_by_condition[si,cc,ii,kk] = np.mean(predlabs==1)

                    rt_by_condition[si,cc,ii,kk] = np.mean(rts[did_respond])

    return acc_by_condition, dprime_by_condition, propyes_by_condition, rt_by_condition

def get_perf_by_cued_nat():

    # natural or artificial categories?
    is_natural = np.array([1,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1]).astype(bool)
    kind_names = ['Artificial','Natural']
    n_kinds = len(kind_names)
    
    acc_by_condition = np.zeros((n_subjects, n_cue_levels, n_image_types, n_kinds))
    dprime_by_condition = np.zeros((n_subjects, n_cue_levels, n_image_types, n_kinds))
    propyes_by_condition = np.zeros((n_subjects, n_cue_levels, n_image_types, n_kinds))

    acc_absent_by_condition = np.zeros((n_subjects, n_cue_levels, n_image_types, n_kinds))
    
    for si, ss in enumerate(subjects):

        trial_data = trial_data_all[trial_data_all['subject']==ss]
        cbi = np.array(trial_data['which_cb'])[0] - 1

        # define super-category of verbal cue
        cued_supercateg = np.array(trial_data['cue_name'])
        shown_supercateg = np.array(trial_data['super_name'])
        is_basic = np.array(trial_data['cue_level'])=='basic'
        # for basic-level trials, super-categ of cue is always same as shown
        cued_supercateg[is_basic] = shown_supercateg[is_basic]

        for cc, cue in enumerate(cue_level_names):

            for ii, imtype in enumerate(image_type_names):

                for kk in range(n_kinds):

                    supcats_use = super_names[(is_natural==kk) & (super_cbinds==cbi)]
                    # print(kind_names[kk])
                    # print(supcats_use)

                    inds = (trial_data['cue_level']==cue) & \
                            (trial_data['image_type']==imtype) & \
                            np.isin(cued_supercateg, supcats_use)

#                     print(np.sum(inds))

                    predlabs = np.array(trial_data['resp'])[inds]
                    reallabs = np.array(trial_data['correct_resp'])[inds]

                    acc_by_condition[si,cc,ii, kk] = np.mean(predlabs==reallabs)

                    did_respond = predlabs>-1

                    predlabs = predlabs[did_respond]
                    reallabs = reallabs[did_respond]

                    dprime_by_condition[si,cc,ii,kk] = stats_utils.get_dprime(predlabs, reallabs)

                    propyes_by_condition[si,cc,ii,kk] = np.mean(predlabs==1)

                    # get accuracy for only target-absent trials
                    # this isolates effect of cue 
                    inds = (trial_data['cue_level']==cue) & \
                            (trial_data['image_type']==imtype) & \
                            np.isin(cued_supercateg, supcats_use) & \
                            (trial_data['target_present']==False)

                    predlabs = np.array(trial_data['resp'])[inds]
                    reallabs = np.array(trial_data['correct_resp'])[inds]

                    acc_absent_by_condition[si, cc, ii, kk] = np.mean(predlabs==reallabs)

    return acc_by_condition, dprime_by_condition, propyes_by_condition, acc_absent_by_condition


def get_perf_by_run():
    
    run_nums = np.unique(trial_data_all['run_number'])
    n_runs = len(run_nums)
    acc_by_run = np.zeros((n_subjects, n_runs))
    rt_by_run = np.zeros((n_subjects, n_runs))
    
    for si, ss in enumerate(subjects):

        trial_data = trial_data_all[trial_data_all['subject']==ss]
        
        for ri, rr in enumerate(run_nums):
            run_inds = trial_data['run_number']==rr
            acc_by_run[si,ri] = np.mean(np.array(trial_data['correct'])[run_inds])
            rt_by_run[si,ri] = np.nanmean(np.array(trial_data['rt'])[run_inds])
        
    return acc_by_run, rt_by_run


def choose_category_subsets():

    """
    # choosing subsets of categories to use in expt3
    # divide each image set of 10 supercategories in half, 
    # to get 4 image sets of 5 supercategories each.
    # use the behavioral performance on each supercategory in expt1 
    # to try and split evenly.
    # this should give:
    [['fruit', 'bird', 'home_decor', 'weapon', 'vehicle'],
     ['body_part','drink','electronic_device','sports_equipment','medical_equipment'],
     ['vegetable', 'dessert', 'clothing', 'part_of_car', 'toy'],
     ['plant', 'insect', 'furniture', 'kitchen_tool', 'office_supply']]
    Note these are chosen based on behavioral performance for expt1 subjects
    as of 4/19/23, so if you run it again based on different subjects, might 
    output something different.
    """
    
    acc_by_supcat, dprime_by_supcat = get_perf_by_supercateg()

    groups = []
    
    for cbi, cb in enumerate([1,2]):

        vals = acc_by_supcat[cbi][:,:,:,:]

        supnames = super_names[super_cbinds==cbi]

        plot_vals = np.mean(np.mean(vals, axis=1), axis=1)
        meanvals = np.mean(plot_vals, axis=0)

        # best to worst
        order = np.flip(np.argsort(meanvals))

        group1 = order[np.arange(0,10,2)]
        group2 = order[np.arange(1,10,2)]

        groups += [supnames[group1], supnames[group2]]
        
    save_filename = os.path.join(project_root, 'make_expt_designs', 'category_subsets_4sets.npy')
    np.save(save_filename, groups)
    return 
import os, sys
import numpy as np
import pandas as pd

project_root = '/user_data/mmhender/featsynth/'
ecoset_info_path = '/user_data/mmhender/stimuli/ecoset_info/'
sys.path.append(os.path.join(project_root, 'code'))

from utils import stats_utils

def get_categ_info():
   
    # load list of 64 categories to use here.
    # after running this expt, we changed some of the musical instrument categories.
    # so this file has the "old version" of categories, used in this expt.
    # fn = os.path.join(ecoset_info_path, 'categ_use_ecoset.npy')
    fn = os.path.join(ecoset_info_path, 'categ_use_ecoset_OLDVERSION.npy')
    info = np.load(fn, allow_pickle=True).item()
    
    bnames = np.array(list(info['binfo'].keys()))
    bnames = [b.replace(' ', '_') for b in bnames]

    snames = np.array(list(info['sinfo'].keys()))
    snames = [s.replace(' ', '_') for s in snames]

    super_names = snames
    super_cbinds = np.zeros(np.shape(super_names)).astype(int)

    basic_names = bnames
    basic_inds = np.arange(len(bnames))

    n_basic = len(basic_names)
    n_super = len(super_names)
    n_basic_each_super  = int(n_basic/n_super)

    super_names_long = np.repeat(super_names, n_basic_each_super)
    super_inds_long = np.repeat(np.arange(n_super), n_basic_each_super)
    
    return super_names, super_cbinds, basic_names, basic_inds, \
        super_names_long, super_inds_long, n_basic, n_super, n_basic_each_super

super_names, super_cbinds, basic_names, basic_inds, \
        super_names_long, super_inds_long, n_basic, n_super, n_basic_each_super = \
            get_categ_info()

def save_categ_info():
    
    super_names, super_cbinds, basic_names, basic_inds, \
        super_names_long, super_inds_long, n_basic, n_super, n_basic_each_super = \
            get_categ_info()

    save_filename = os.path.join(project_root, 'code', 'make_expt_designs', 'expt3_categ_info.npy')
    np.save(save_filename, {'super_names': super_names, \
                           'super_cbinds': super_cbinds, \
                           'basic_names': basic_names, \
                           'basic_inds': basic_inds, \
                           'super_names_long': super_names_long, \
                           'super_inds_long': super_inds_long})
    
def load_data(expt_name = 'expt3'):

    preproc_folder = os.path.join(project_root, 'online_data', expt_name, 'preproc')
    data_filename = os.path.join(preproc_folder, 'preproc_data_all.csv')
    trial_data_all = pd.read_csv(data_filename)
    subjects = np.unique(trial_data_all['subject'])
    n_subjects = len(subjects)
   
    cue_level_names = np.unique(np.array(trial_data_all['cue_level']))
    
    image_type_names = np.unique(np.array(trial_data_all['image_type']))
    type_order_plot = [1,2,3,4,0]
    image_type_names = np.array(image_type_names)[type_order_plot]

    return trial_data_all, subjects, cue_level_names, image_type_names

trial_data_all, subjects, cue_level_names, image_type_names = load_data()
 
n_subjects = len(subjects)
n_cue_levels = len(cue_level_names)
n_image_types = len(image_type_names)


def get_perf_by_supercateg():

    # break trials down by conditions (cue level, image type) and
    # each super-category group
    acc_by_supcat = np.full(fill_value = np.nan, \
                             shape = (n_subjects, n_cue_levels, \
                                      n_image_types, n_super))
    dprime_by_supcat = np.full(fill_value = np.nan, \
                             shape = (n_subjects, n_cue_levels, \
                                      n_image_types, n_super))
    
    rt_by_supcat = np.full(fill_value = np.nan, \
                             shape = (n_subjects, n_cue_levels, \
                                      n_image_types, n_super))
    
    for si, ss in enumerate(subjects):

        trial_data = trial_data_all[trial_data_all['subject']==ss]

        for sc, supcat in enumerate(super_names):

            for cc, cue in enumerate(cue_level_names):

                for ii, imtype in enumerate(image_type_names):

                    inds = (trial_data['cue_level']==cue) & \
                            (trial_data['image_type']==imtype) & \
                            (trial_data['super_name']==supcat)
                    # print([supcat, cue, imtype])
                    # print(np.sum(inds))
                    assert(np.sum(inds)==8)

                    predlabs = np.array(trial_data['resp'])[inds]
                    reallabs = np.array(trial_data['correct_resp'])[inds]

                    acc = np.mean(predlabs==reallabs)
                    
                    did_respond = predlabs>-1

                    predlabs = predlabs[did_respond]
                    reallabs = reallabs[did_respond]
                    
                    dprime = stats_utils.get_dprime(predlabs, reallabs)
                    
                    rts = trial_data['rt'][inds]
                    rts_use = rts[did_respond]
            
                    acc_by_supcat[si,cc,ii,sc] = acc
                    dprime_by_supcat[si,cc,ii,sc] = dprime
                    rt_by_supcat[si,cc,ii,sc] = np.mean(rts_use)
                
    return acc_by_supcat, dprime_by_supcat, rt_by_supcat


def get_perf_by_cond():

    acc_by_condition = np.zeros((n_subjects, n_cue_levels, n_image_types))
    dprime_by_condition = np.zeros((n_subjects, n_cue_levels, n_image_types))
    propleft_by_condition = np.zeros((n_subjects, n_cue_levels, n_image_types))
    rt_by_condition = np.zeros((n_subjects, n_cue_levels, n_image_types))

    for si, ss in enumerate(subjects):

        trial_data = trial_data_all[trial_data_all['subject']==ss]

        for cc, cue in enumerate(cue_level_names):

            for ii, imtype in enumerate(image_type_names):

                inds = (trial_data['cue_level']==cue) & \
                        (trial_data['image_type']==imtype)
                assert(np.sum(inds)==64)

                predlabs = np.array(trial_data['resp'])[inds]
                reallabs = np.array(trial_data['correct_resp'])[inds]
                rts = np.array(trial_data['rt'])[inds]
                
                acc_by_condition[si,cc,ii] = np.mean(predlabs==reallabs)

                did_respond = predlabs>-1

                predlabs = predlabs[did_respond]
                reallabs = reallabs[did_respond]

                dprime_by_condition[si,cc,ii] = stats_utils.get_dprime(predlabs, reallabs)

                propleft_by_condition[si,cc,ii] = np.mean(predlabs==1)
                
                rt_by_condition[si,cc,ii] = np.mean(rts[did_respond])

    return acc_by_condition, dprime_by_condition, propleft_by_condition, rt_by_condition



def get_perf_by_run():
    
    run_nums = np.unique(trial_data_all['run_number'])
    n_runs = len(run_nums)
    
    # going to sort the runs by number within each task
    # the tasks alternate even/odd runs
    
    acc_by_run = np.zeros((n_subjects, n_cue_levels, int(n_runs/2)))
    dprime_by_run = np.zeros((n_subjects, n_cue_levels, int(n_runs/2)))
    rt_by_run = np.zeros((n_subjects, n_cue_levels, int(n_runs/2)))
    
    for si, ss in enumerate(subjects):
        
        for cc, cue in enumerate(cue_level_names):

            trial_data = trial_data_all[(trial_data_all['subject']==ss) & \
                                        (trial_data_all['cue_level']==cue)]
            # this is either just even runs or just odd runs, 
            # since the cue levels alternate
            run_nums_here = np.unique(trial_data['run_number'])

            # print(ss, cue, run_nums_here)
            for ri, rr in enumerate(run_nums_here):
                
                inds = trial_data['run_number']==rr
                
                predlabs = np.array(trial_data['resp'])[inds]
                reallabs = np.array(trial_data['correct_resp'])[inds]
                rts = np.array(trial_data['rt'])[inds]
                
                acc_by_run[si, cc, ri] = np.mean(predlabs==reallabs)
                
                did_respond = predlabs>-1
                
                rt_by_run[si, cc, ri] = np.mean(rts[did_respond])
                
                predlabs = predlabs[did_respond]
                reallabs = reallabs[did_respond]
                
                dprime_by_run[si, cc, ri] = stats_utils.get_dprime(predlabs, reallabs)
        
    return acc_by_run, dprime_by_run, rt_by_run

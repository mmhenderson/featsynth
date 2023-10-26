import os
import numpy as np
import pandas as pd

things_root = '/user_data/mmhender/stimuli/things/'
project_root = '/user_data/mmhender/featsynth/'

from utils import dropbox_utils, expt_utils
token_path = os.path.join(project_root, 'tokens/dbx_token.txt')
dbx = dropbox_utils.init_dropbox(token_path)

expt_name = 'pilot1'

def make_trial_info(rndseed = None):

    # decide which images we will use
    categ_sets, concept_sets, image_name_sets = choose_image_subsets()
    n_counterbalance_conds = len(categ_sets)
    
    # figure out trial conditions/counts
    n_categ_use = len(categ_sets[0])
    n_concepts_use = len(concept_sets[0][0])
    n_ex_use = len(image_name_sets[0][0][0])
    
    # set the condition (target present, etc) based on which exemplar it is
    # (same exemplars never used twice across conds)
    n_resp_conds = 4;
    n_ex_per_resp_cond = int(n_ex_use/n_resp_conds)
    
    # image types are original or "synth" from different DNN layers
    n_layers=4
    n_image_types = n_layers+1 
    
    n_trials_total = n_categ_use * n_concepts_use * n_resp_conds * n_ex_per_resp_cond * n_image_types
 
    n_runs = 10;
    assert(np.mod(n_trials_total, n_runs)==0)
    n_trials_per_run = int(n_trials_total/n_runs);
    
    rndseed = 133423
    np.random.seed(rndseed)
    
    for cb in range(n_counterbalance_conds):
        
        print('making trial info for counterbalance cond %d of %d'%(cb, n_counterbalance_conds))
        
        categ_use = categ_sets[cb]
        print(categ_use)
        concepts_use = concept_sets[cb]
        image_names_use = image_name_sets[cb]

        trial_info = pd.DataFrame({'trial_num_overall': np.zeros((n_trials_total,)), 
                                   'categ_ind': np.zeros((n_trials_total,)),
                                  'concept_ind': np.zeros((n_trials_total,)),
                                  'super_name': np.zeros((n_trials_total,)),
                                  'basic_name': np.zeros((n_trials_total,)),
                                  'ex_num': np.zeros((n_trials_total,)),
                                  'image_type_num': np.zeros((n_trials_total,)),
                                  'image_name': np.zeros((n_trials_total,)),
                                  'dropbox_url': np.zeros((n_trials_total,)),
                                  'target_present': np.zeros((n_trials_total,)),
                                  'cue_level': np.zeros((n_trials_total,)),
                                  'cue_name': np.zeros((n_trials_total,)),
                                  })

        layer_names = [None, 'pool1','pool2','pool3','pool4']

        tt=-1
        for ca in range(n_categ_use):
            for co in range(n_concepts_use):
                for ex in range(n_ex_use):
                    for typ in range(n_image_types):
                        tt+=1
                        if np.mod(tt,20)==0:
                            print('proc trial %d of %d'%(tt, n_trials_total))

                        trial_info['trial_num_overall'].iloc[tt] = tt
                        trial_info['categ_ind'].iloc[tt] = ca
                        trial_info['concept_ind'].iloc[tt] = co
                        trial_info['super_name'].iloc[tt] = categ_use[ca]
                        trial_info['basic_name'].iloc[tt] = concepts_use[ca][co]
                        trial_info['ex_num'].iloc[tt] = ex
                        trial_info['image_type_num'].iloc[tt] = typ

                        name_raw = image_names_use[ca][co][ex].split('.jpg')[0]

                        if layer_names[typ] is None:

                            trial_info['image_name'].iloc[tt] = os.path.join('%s_orig.png'%name_raw)
                            dbxpath = '/images/'+name_raw+'/orig.png'

                        else:

                            trial_info['image_name'].iloc[tt] = os.path.join('%s_grid5_1x1_upto_%s.png'%\
                                                                             (name_raw,layer_names[typ]))
                            dbxpath = '/images/'+name_raw+'/grid5_1x1_upto_%s.png'%layer_names[typ]

                        # print(dbxpath)
                        trial_info['dropbox_url'].iloc[tt] = dropbox_utils.get_shared_url(dbxpath, dbx)

                        # setting the condition (target present, etc) based on which exemplar it is
                        # (same exemplars never shown twice)
                        if np.mod(ex,4)==0:
                            trial_info['target_present'].iloc[tt] = True
                            trial_info['cue_level'].iloc[tt] = 'basic'
                        elif np.mod(ex,4)==1:
                            trial_info['target_present'].iloc[tt] = False
                            trial_info['cue_level'].iloc[tt] = 'basic'
                        elif np.mod(ex,4)==2:
                            trial_info['target_present'].iloc[tt] = True
                            trial_info['cue_level'].iloc[tt] = 'super'
                        elif np.mod(ex,4)==3:
                            trial_info['target_present'].iloc[tt] = False
                            trial_info['cue_level'].iloc[tt] = 'super'

        print('prepping cue names')
        # decide what name to use to "cue" each trial, based on which level they are being cued at
        # (these are all "correct names"; some of them get changed to incorrect in the next step)
        trial_info['cue_name'] = trial_info['super_name']
        trial_info['cue_name'][trial_info['cue_level']=='basic'] = \
            trial_info['basic_name'][trial_info['cue_level']=='basic']

        # Assign incorrect names to all the basic-level target-absent trials
        # always swapping basic-level names across trials with same superordinate 
        # level name, and same in all other attributes
        lev = 'basic'

        for ca in range(n_categ_use):
            for typ in range(n_image_types):
                for ex in range(n_ex_use):

                    group = (trial_info['target_present']==False) & \
                            (trial_info['categ_ind']==ca) & \
                            (trial_info['cue_level']==lev) & \
                            (trial_info['image_type_num']==typ) & \
                            (trial_info['ex_num']==ex)

                    actual_basic_inds = np.array(trial_info['concept_ind'][group])
                    actual_basic_names = np.array(trial_info['basic_name'][group])
                    # switch the names around pseudo-randomly, so they are 
                    # always cued with the wrong name on these trials.
                    incorrect_basic_inds = expt_utils.swap_rand_pairs(actual_basic_inds).astype(int)

                    trial_info['cue_name'].iloc[group] = actual_basic_names[incorrect_basic_inds]


        # Assign incorrect names to all the superord-level target-absent trials
        lev = 'super'

        for co in range(n_concepts_use):
            for typ in range(n_image_types):
                for ex in range(n_ex_use):

                    group = (trial_info['target_present']==False) & \
                            (trial_info['concept_ind']==co) & \
                            (trial_info['cue_level']==lev) & \
                            (trial_info['image_type_num']==typ) & \
                            (trial_info['ex_num']==ex)

                    actual_super_inds = np.array(trial_info['categ_ind'][group])
                    actual_super_names = np.array(trial_info['super_name'][group])
                    # switch the names around pseudo-randomly, so they are 
                    # always cued with the wrong name on these trials.
                    incorrect_super_inds = expt_utils.swap_rand_pairs(actual_super_inds).astype(int)

                    trial_info['cue_name'].iloc[group] = actual_super_names[incorrect_super_inds]

        # shuffle the order of everything together.
        shuff_order = np.random.permutation(np.arange(n_trials_total))
        trial_info = trial_info.iloc[shuff_order]

        # finally, assign the run numbers for each trial (these are the only non-shuffled columns).
        trial_info['run_number'] = np.repeat(np.arange(n_runs), n_trials_per_run) + 1
        trial_info['trial_in_run'] = np.tile(np.arange(n_trials_per_run), [n_runs,]) + 1

        # double check everything
        print('checking trial info')
        check_trial_info(trial_info)

        # save everything to a single CSV file
        expt_design_folder = os.path.join(project_root, 'expt_design', expt_name)
        if not os.path.exists(expt_design_folder):
            os.makedirs(expt_design_folder)
        trialinfo_filename1 =  os.path.join(expt_design_folder, 'trial_info_counterbal%d.csv'%(cb+1))
        print('saving to %s'%trialinfo_filename1)
        trial_info.to_csv(trialinfo_filename1, index=False)

        # making .js files for use in jsPsych
        # (this file holds all the runs)
        js_filename = os.path.join(expt_design_folder, 'trialseq_counterbal%d.js'%(cb+1))
        
        expt_utils.make_runs_js(trial_info, js_filename, var_name='info%d'%(cb+1))
        
    return


def choose_image_subsets():
    
    concepts_filename = os.path.join(things_root, 'concepts_use.npy')
    concepts_use = np.load(concepts_filename,allow_pickle=True).item()
    categ_names = concepts_use['categ_names']
    concept_names = concepts_use['concept_names_subsample']
    image_names = concepts_use['image_names']
    
    # we only made synthetic ims for the first 4 exemplars, so ignore the rest 
    for kk in image_names.keys():
        image_names[kk] = image_names[kk][0:4]
    
    concepts_all = np.concatenate(concept_names)
    categ_all = np.repeat(categ_names, [len(cc) for cc in concept_names])

    n_categ_use_each = 6;
    n_concepts_use_each = 6;

    n_ex_use = 4;
    assert(np.mod(n_ex_use,4)==0)

    # randomly selecting the subset of categories and concepts that will be used here
    rndseed = 456466
    np.random.seed(rndseed)

    # i am manually assigning the categories that are animals, food/drink, 
    # or human-related, so that they are balanced in the two sets
    categ_set1 = ['bird', 'drink', 'fruit', 'body_part']
    categ_set2 = ['insect','dessert', 'vegetable', 'clothing']
    # then randomly assign the remaining categories
    categ_shuffled = np.array(categ_names)[np.random.permutation(len(categ_names))]
    categ_shuffled = [c for c in categ_shuffled if (c not in categ_set1 and c not in categ_set2)]
    n_append = n_categ_use_each - len(categ_set1)
    categ_set1 += categ_shuffled[0:n_append]
    categ_set2 += categ_shuffled[n_append:2*n_append]

    categ_sets = [categ_set1, categ_set2]
    concept_sets = []
    image_name_sets = []

    for categ_use in categ_sets:

        categ_inds_use = [np.where(np.array(categ_names)==c)[0][0] for c in categ_use]

        # randomly select which individual concepts and which exemplars to use here
        concepts_use = [np.random.choice(concept_names[ca], n_concepts_use_each, replace=False) for ca in categ_inds_use]

        image_names_use = [[np.random.choice(image_names[co], n_ex_use, replace=False) for co in conc] \
                           for conc in concepts_use]

        concept_sets += [concepts_use]
        image_name_sets += [image_names_use]

    return categ_sets, concept_sets, image_name_sets

def check_trial_info(ti):

    # make sure that the target present/target absent trials line up correctly
    # (cue name should match actual name for target present, and should 
    # always mismatch for target absent)

    inds_check = (ti['cue_level']=='basic') & (ti['target_present']==True)
    assert(np.all(ti['cue_name'][inds_check]==ti['basic_name'][inds_check]))
    inds_check = (ti['cue_level']=='basic') & (ti['target_present']==False)
    assert(np.all(ti['cue_name'][inds_check]!=ti['basic_name'][inds_check]))

    inds_check = (ti['cue_level']=='super') & (ti['target_present']==True)
    assert(np.all(ti['cue_name'][inds_check]==ti['super_name'][inds_check]))
    inds_check = (ti['cue_level']=='super') & (ti['target_present']==False)
    assert(np.all(ti['cue_name'][inds_check]!=ti['super_name'][inds_check]))

    # check to make sure individual attributes are distributed evenly across trials 
    n_trials_total = ti.shape[0]

    attr_check_even = ['super_name', 'basic_name', 'ex_num', 'image_type_num', 'target_present', 'cue_level']
    for attr in attr_check_even:

        # should be an equal number of each thing 
        un, counts = np.unique(ti[attr], return_counts=True)
        assert(np.all(counts==n_trials_total/len(un)))

    # check the counterbalancing over multiple attributes

    # there should be an equal number of trials in each of the combinations of these 
    # different attribute "levels". for example each combination of category/image type. 

    attr_balanced = [ti['categ_ind'], ti['concept_ind'], ti['image_type_num'], ti['target_present'], ti['cue_level']]
    attr_balanced_inds = np.array([np.unique(attr, return_inverse=True)[1] for attr in attr_balanced]).T

    n_levels_each = [len(np.unique(attr)) for attr in attr_balanced]
    n_combs_expected = np.prod(n_levels_each)
    n_repeats_expected = n_trials_total/n_combs_expected

    un_rows, counts = np.unique(attr_balanced_inds,axis=0, return_counts=True)

    assert(un_rows.shape[0]==n_combs_expected)
    assert(np.all(counts==n_repeats_expected))

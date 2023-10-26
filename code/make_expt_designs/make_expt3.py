import os, sys
import numpy as np
import pandas as pd
import copy

things_root = '/user_data/mmhender/stimuli/things/'
project_root = '/user_data/mmhender/featsynth/'
stimuli_folder = 'images_v3'
ecoset_info_path = '/user_data/mmhender/stimuli/ecoset_info/'

from utils import dropbox_utils, expt_utils
token_path = os.path.join(project_root, 'tokens/dbx_token.txt')
dbx = dropbox_utils.init_dropbox(token_path)

expt_name = 'expt3'

def make_trial_info(rndseed=986768):

    # using here the same set of categories that we chose from ecoset.
    # 64 basic categories in 8 superordinate groups.
    fn = os.path.join(ecoset_info_path, 'categ_use_ecoset.npy')
    info = np.load(fn, allow_pickle=True).item()
    basic_names = list(info['binfo'].keys())
    super_names = list(info['sinfo'].keys())
    
    # load info about things images/names 
    fn2load = os.path.join(things_root, 'things_file_info.npy')
    tfiles = np.load(fn2load, allow_pickle=True).item()

    categ_sets = [super_names]
    concept_sets = [[info['sinfo'][sname]['basic_names'] for sname in super_names]]

    n_ex_use = 10;
    n_counterbalance_conds = 1


    # figure out trial conditions/counts
    n_categ_use = len(categ_sets[0])
    n_concepts_use = len(concept_sets[0][0])
    # n_ex_use = len(image_name_sets[0][0][0])

    # image types are original or "synth" from different DNN layers
    n_layers=4
    n_image_types = n_layers+1 

    cue_levels = ['basic', 'super']
    n_cue_levels = len(cue_levels); # basic or superordinate

    n_trials_per_concept = n_image_types * n_cue_levels
    assert(n_trials_per_concept==n_ex_use)

    n_trials_total = n_categ_use * n_concepts_use * n_trials_per_concept

    n_runs = 16;
    assert(np.mod(n_trials_total, n_runs)==0)
    n_trials_per_run = int(n_trials_total/n_runs);

    np.random.seed(rndseed)
    
    for cb in range(n_counterbalance_conds):
        
        print('making trial info for counterbalance cond %d of %d'%(cb, n_counterbalance_conds))

        categ_use = categ_sets[cb]
        categ_use = [c.replace(' ', '_') for c in categ_use]
        print(categ_use)
        concepts_use = concept_sets[cb]
        # image_names_use = image_name_sets[cb]
        
        # make 100 different randomized orders for this same image set
        # n_random_orders=2;
        n_random_orders=100;
        
        trial_info_list = []
    
        dbxpath_dict = dict()
        
        for rand in range(n_random_orders):

            trial_info = pd.DataFrame({'trial_num_overall': np.zeros((n_trials_total,)), 
                            'run_number': np.zeros((n_trials_total,)), 
                            'trial_in_run': np.zeros((n_trials_total,)), 
                            'image_set_num': np.zeros((n_trials_total,)), 
                            'random_order_number': np.zeros((n_trials_total,)), 
                           'categ_ind': np.zeros((n_trials_total,)),
                          'concept_ind': np.zeros((n_trials_total,)),
                          'super_name': np.zeros((n_trials_total,)),
                          'basic_name': np.zeros((n_trials_total,)),
                          'ex_num': np.zeros((n_trials_total,)),
                          'image_type_num': np.zeros((n_trials_total,)),
                          'image_type': np.zeros((n_trials_total,)),
                          'image_name': np.zeros((n_trials_total,)),
                          'dropbox_url': np.zeros((n_trials_total,)),
                          'cue_level_num': np.zeros((n_trials_total,)),
                          'cue_level': np.zeros((n_trials_total,)),
                          'cue_name': np.zeros((n_trials_total,)),
                          'distractor_name': np.zeros((n_trials_total,)),
                          'left_name': np.zeros((n_trials_total,)),
                          'right_name': np.zeros((n_trials_total,)),
                          'correct_resp': np.zeros((n_trials_total,)), 
                          })

            image_type_names = ['orig', 'pool1','pool2','pool3','pool4']

            tt=-1
            for ca in range(n_categ_use):

                for co in range(n_concepts_use):

                    ex = -1

                    for typ in range(n_image_types):
                        for cue in range(n_cue_levels):

                            tt+=1
                            if np.mod(tt,100)==0:
                                print('proc trial %d of %d'%(tt, n_trials_total))

                            trial_info['trial_num_overall'].iloc[tt] = tt
                            trial_info['categ_ind'].iloc[tt] = ca
                            trial_info['concept_ind'].iloc[tt] = co
                            trial_info['super_name'].iloc[tt] = categ_use[ca]
                            trial_info['basic_name'].iloc[tt] = concepts_use[ca][co].replace(' ', '_')

                            trial_info['image_type_num'].iloc[tt] = typ
                            trial_info['image_type'].iloc[tt] = image_type_names[typ]
                            trial_info['cue_level_num'].iloc[tt] = cue
                            trial_info['cue_level'].iloc[tt] = cue_levels[cue]

                            trial_info['image_set_num'].iloc[tt] = cb
                            trial_info['random_order_number'].iloc[tt] = rand

                            ex += 1
                            trial_info['ex_num'].iloc[tt] = ex

                            bname = concepts_use[ca][co]
                            name_raw = tfiles[bname][ex].split('.jpg')[0]
                            
                            # name_raw = image_names_use[ca][co][ex].split('.jpg')[0]

                            if image_type_names[typ]=='orig':

                                trial_info['image_name'].iloc[tt] = os.path.join(name_raw, 'orig.png')
                                dbxpath = '/'+os.path.join(stimuli_folder, name_raw, 'orig.png')

                            else:

                                trial_info['image_name'].iloc[tt] = os.path.join(name_raw, \
                                                                                 'scramble_upto_%s.png'%\
                                                                                 (image_type_names[typ]))
                                dbxpath = '/'+os.path.join(stimuli_folder,name_raw,\
                                                       'scramble_upto_%s.png'%image_type_names[typ])

                            # print(dbxpath)
                            sys.stdout.flush()
                            if dbxpath in dbxpath_dict.keys():
                                # don't have to re-create url, faster
                                url = dbxpath_dict[dbxpath]
                            else:
                                # url = ''
                                print('creating dropbox url')
                                print(dbxpath)
                                # print(dbx)
                                url = dropbox_utils.get_shared_url(dbxpath, dbx)
                                dbxpath_dict[dbxpath] = url
                                print(url)
                            
                            trial_info['dropbox_url'].iloc[tt] = url
              
            
            # randomly deciding whether basic or super happens on odd/even runs
            odd_even = int(np.random.normal(0,1,1)>0)
            basic_run_order = np.random.permutation(len(categ_use))*2 + odd_even
            super_run_order = np.arange(n_runs)
            super_run_order = super_run_order[~np.isin(super_run_order, basic_run_order)]


            # now group the trials into runs based on which task is being done
            
            
            # first basic-level tasks, within each superordinate   
            for si, sname in enumerate(categ_use):

                for typ in range(n_image_types):

                    trial_inds = np.array((trial_info['super_name']==sname) & \
                                          (trial_info['cue_level']=='basic') & \
                                          (trial_info['image_type_num']==typ))

                    trial_info['run_number'].iloc[trial_inds] = basic_run_order[si]

                    # name of actual basic-level category
                    cue_names_actual = np.array(trial_info['basic_name'].iloc[trial_inds])

                    # now assign the "distractor" names by shuffling real names
                    distract_names = expt_utils.shuffle_nosame(cue_names_actual)

                    trial_info['cue_name'].iloc[trial_inds] =  cue_names_actual
                    trial_info['distractor_name'].iloc[trial_inds] = distract_names

                    # randomly assign whether left or right side is the actual cue name 
                    cue_on_left = np.mod(np.random.permutation(len(cue_names_actual)),2)==0
                    left_name = copy.deepcopy(cue_names_actual)
                    right_name = copy.deepcopy(distract_names)
                    # switch half of them
                    left_name[~cue_on_left] = distract_names[~cue_on_left]
                    right_name[~cue_on_left] = cue_names_actual[~cue_on_left]

                    # correct resp: 1 for left, 2 for right
                    correct_resp = np.ones_like(cue_on_left).astype(int)
                    correct_resp[~cue_on_left] = 2

                    trial_info['left_name'].iloc[trial_inds] = left_name
                    trial_info['right_name'].iloc[trial_inds] = right_name

                    trial_info['correct_resp'].iloc[trial_inds] = correct_resp


            # now organizing the superordinate task runs
            for typ in range(n_image_types):

                trial_inds = np.array((trial_info['cue_level']=='super') & \
                                      (trial_info['image_type_num']==typ) 
                                      )

                # print(np.sum(trial_inds))

                # actual cue names (superordinate)
                cue_names_actual = np.array(trial_info['super_name'].iloc[trial_inds])

                # creating the list of "distractor" names that are pseudorandom
                distract_names = [np.array(categ_use)[np.array(categ_use)!=categ] for categ in categ_use]
                distract_names = [d[np.random.permutation(len(d))] for d in distract_names]
                extras = expt_utils.shuffle_nosame(np.array(categ_use))
                distract_names = [list(d)+[e] for d, e in zip(distract_names, extras)]
                distract_names = np.concatenate(distract_names, axis=0)

                # check they meet all the criteria
                un, counts = np.unique(distract_names, return_counts=True)
                assert(np.all(counts==counts[0]))
                assert not np.any(distract_names==cue_names_actual)

                trial_info['cue_name'].iloc[trial_inds] =  cue_names_actual
                trial_info['distractor_name'].iloc[trial_inds] = distract_names


                # randomly assign whether left or right side is the actual cue name 
                n_each = [np.sum(cue_names_actual==categ) for categ in categ_use]
                cue_on_left = np.concatenate([np.mod(np.random.permutation(n),2)==0 for n in n_each], \
                                             axis=0)
                left_name = copy.deepcopy(cue_names_actual)
                right_name = copy.deepcopy(distract_names)
                # switch half of them
                left_name[~cue_on_left] = distract_names[~cue_on_left]
                right_name[~cue_on_left] = cue_names_actual[~cue_on_left]

                # correct resp: 1 for left, 2 for right
                correct_resp = np.ones_like(cue_on_left).astype(int)
                correct_resp[~cue_on_left] = 2

                trial_info['left_name'].iloc[trial_inds] = left_name
                trial_info['right_name'].iloc[trial_inds] = right_name

                trial_info['correct_resp'].iloc[trial_inds] = correct_resp

                # break all these into equal sized runs
                super_run_seq = np.repeat(super_run_order, int(n_trials_per_run/n_image_types))
                super_run_seq = super_run_seq[np.random.permutation(len(super_run_seq))]
                trial_info['run_number'].iloc[trial_inds] = super_run_seq


            # organize the trials by run number
            trial_info.sort_values(by='run_number', inplace=True)

            # now shuffle within each run 
            for rr in np.unique(trial_info['run_number']):

                inds = np.where(np.array(trial_info['run_number']==rr))[0] 
                shuff_order = np.random.permutation(len(inds))
                trial_info.iloc[inds] = trial_info.iloc[inds[shuff_order]]
                
            # assign trial numbers 
            trial_info['trial_in_run'] = np.tile(np.arange(n_trials_per_run), [n_runs,])
            trial_info['trial_num_overall'] = np.arange(n_trials_total)

            trial_info.set_index(np.arange(n_trials_total))

            # double check everything
            print('checking trial info')
            check_trial_info(trial_info, concepts_use, categ_use)

            # save everything to a single CSV file
            expt_design_folder = os.path.join(project_root, 'expt_design', expt_name)
            if not os.path.exists(expt_design_folder):
                os.makedirs(expt_design_folder)
            trialinfo_filename1 =  os.path.join(expt_design_folder, 'trial_info_counterbal%d_randorder%d.csv'%(cb+1, rand))
            print('saving to %s'%trialinfo_filename1)
            trial_info.to_csv(trialinfo_filename1, index=False)
            
            trial_info_list += [trial_info]

        # making .js files for use in jsPsych
        # (this file holds all the runs)
        js_filename = os.path.join(expt_design_folder, 'trialseq_counterbal%d.js'%(cb+1))
        print('saving to %s'%(js_filename))
        expt_utils.make_runs_js_multiplesets(trial_info_list, js_filename, var_name='info%d'%(cb+1))
        
    return




def check_trial_info(ti, concepts_use, categ_use):

    # double checking all the trial attributes
    
    
    # check that we assigned the "correct response" correctly
    # 1 is left, 2 is right
    inds_check = np.array(ti['correct_resp']==1)
    assert(np.all(ti['cue_name'][inds_check]==ti['left_name'][inds_check]))
    inds_check = np.array(ti['correct_resp']==2)
    assert(np.all(ti['cue_name'][inds_check]==ti['right_name'][inds_check]))

    # cue and distractor name always different
    assert(not np.any(ti['cue_name']==ti['distractor_name']))

    # check that basic and super task are assigned correctly
    inds_check = np.array(ti['cue_level']=='basic')
    assert(np.all(np.isin(np.array(ti['cue_name'].iloc[inds_check]), concepts_use)))
    assert(np.all(np.isin(np.array(ti['distractor_name'].iloc[inds_check]), concepts_use)))

    # for each basic run, checking that the cue and distractor names are all part of a single superord categ
    for rr in np.unique(ti['run_number'].iloc[inds_check]):
        inds_check_run = inds_check & (np.array(ti['run_number']==rr))
        sname = np.array(ti['super_name'].iloc[inds_check_run])
        assert(np.all(sname==sname[0]))
        sind = np.where(np.array(categ_use)==sname[0])[0][0]
        assert(np.all(np.isin(np.array(ti['distractor_name'].iloc[inds_check_run]), concepts_use[sind])))
        assert(np.all(np.isin(np.array(ti['cue_name'].iloc[inds_check_run]), concepts_use[sind])))

    # check superordinate names 
    inds_check = np.array(ti['cue_level']=='super')
    assert(np.all(np.isin(np.array(ti['cue_name'].iloc[inds_check]), categ_use)))
    assert(np.all(np.isin(np.array(ti['distractor_name'].iloc[inds_check]), categ_use)))

    # check to make sure individual attributes are distributed evenly across trials 
    n_trials_total = ti.shape[0]

    attr_check_even = ['super_name', 'basic_name', 'ex_num', 'image_type_num', \
                       'correct_resp', 'cue_level', 'run_number']
    for attr in attr_check_even:

        # should be an equal number of each thing 
        un, counts = np.unique(ti[attr], return_counts=True)
        # print(un, counts)
        assert(np.all(counts==n_trials_total/len(un)))


    # check the counterbalancing over multiple attributes

    # there should be an equal number of trials in each of the combinations of these 
    # different attribute "levels". for example each combination of category/image type. 

    attr_balanced = [ti['categ_ind'], ti['concept_ind'], ti['image_type_num'], ti['cue_level']]
    attr_balanced_inds = np.array([np.unique(attr, return_inverse=True)[1] for attr in attr_balanced]).T

    n_levels_each = [len(np.unique(attr)) for attr in attr_balanced]
    n_combs_expected = np.prod(n_levels_each)
    n_repeats_expected = n_trials_total/n_combs_expected

    un_rows, counts = np.unique(attr_balanced_inds,axis=0, return_counts=True)

    assert(un_rows.shape[0]==n_combs_expected)
    assert(np.all(counts==n_repeats_expected))

    
    
    # check evenness of stuff for super task only

    inds_check = np.array(ti['cue_level']=='super')
    n_trials_total = np.sum(inds_check)

    attr_check_even = ['super_name', 'cue_name', 'image_type', 'distractor_name', 'correct_resp']

    for attr in attr_check_even:

        # should be an equal number of each thing 
        un, counts = np.unique(ti[attr].iloc[inds_check], return_counts=True)
        # print(un, counts)
        assert(np.all(counts==n_trials_total/len(un)))



    # check evenness of stuff for basic task only

    inds_check = np.array(ti['cue_level']=='basic')
    n_trials_total = np.sum(inds_check)

    attr_check_even = ['basic_name', 'cue_name', 'image_type', 'distractor_name', 'correct_resp']

    for attr in attr_check_even:

        # should be an equal number of each thing 
        un, counts = np.unique(ti[attr].iloc[inds_check], return_counts=True)
        # print(un, counts)
        assert(np.all(counts==n_trials_total/len(un)))


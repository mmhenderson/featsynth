import os, sys
import numpy as np
import pandas as pd
import copy

project_root = '/user_data/mmhender/featsynth/'
stimuli_folder = 'images_comb64_grayscale/'
stimuli_folder_full = '/user_data/mmhender/stimuli/featsynth/images_comb64/'
ecoset_info_path = '/user_data/mmhender/stimuli/ecoset_info/'

sys.path.insert(0, '/user_data/mmhender/featsynth/code/')
from utils import dropbox_utils, expt_utils
token_path = os.path.join(project_root, 'tokens/dbx_token.txt')
dbx = dropbox_utils.init_dropbox(token_path)

expt_name = 'expt5'

def make_trial_info(rndseed=234453):

    # using here the same set of categories that we chose from ecoset.
    # 64 basic categories in 8 superordinate groups.
    fn = os.path.join(ecoset_info_path, 'categ_use_ecoset.npy')
    info = np.load(fn, allow_pickle=True).item()
    basic_names = list(info['binfo'].keys())
    super_names = list(info['sinfo'].keys())

    # load info about which images to use
    # made this in make_featsynth_images_comb64.py
    fn = os.path.join(stimuli_folder_full, 'synth_losses_all.npy')
    print(fn)
    l = np.load(fn, allow_pickle=True).item()

    # in this code, categ = superordinate, concept = basic
    # (following from things paper)
    categ_sets = [super_names]
    concept_sets = [[info['sinfo'][sname]['basic_names'] for sname in super_names]]

    
    # figure out trial conditions/counts
    n_categ_use = len(categ_sets[0])
    n_concepts_use = len(concept_sets[0][0])
    # n_ex_use = len(image_name_sets[0][0][0])

    # image types are original or "synth" from different DNN layers
    n_layers=4
    n_image_types = n_layers+1 

    cue_levels = ['basic', 'super']
    n_cue_levels = len(cue_levels); # basic or superordinate

    # this is per session
    n_trials_per_concept = n_image_types * n_cue_levels
    # assert(n_trials_per_concept==n_ex_use)

    # one session behavioral experiment
    n_sessions = 1;

    # how many unique exemplars will we need for all sessions?
    n_ex_use = n_trials_per_concept * n_sessions 
   
    n_trials_per_session = n_categ_use * n_concepts_use * n_trials_per_concept

    # this is all per session
    n_runs = 16;
    assert(np.mod(n_trials_per_session, n_runs)==0)
    n_trials_per_run = int(n_trials_per_session/n_runs);

    # each run will have 4 mini-blocks of 10 trials each
    n_trials_mini = 10

    assert(np.mod(n_trials_per_run, n_trials_mini)==0)
    n_mini_per_run = int(n_trials_per_run/n_trials_mini)

    n_mini_total = n_mini_per_run * n_runs

    n_trials_per_categ = n_trials_per_run # this just happens to work out this way
    n_mini_per_categ = int(n_trials_per_categ / n_trials_mini)

    np.random.seed(rndseed)

    categ_use = categ_sets[0]
    categ_use = [c.replace(' ', '_') for c in categ_use]
    print(categ_use)
    concepts_use = concept_sets[0]
    
    # make 100 different randomized orders for this same image set
    # n_random_orders=2;
    n_random_orders=100;

    # this will be a list over all randomized orders
    trial_info_list = []

    dbxpath_dict = dict()

    for rand in range(n_random_orders):

        # this will be a list for this random order, all sessions
        trial_info_allsess = []

        # this is a coinflip that will be used to pick which task to start with
        odd_even = int(np.random.normal(0,1,1)>0)
        print(odd_even)

        ex_each_sess = dict()
        for bname in basic_names:
            # this is the order of the exemplars to use, sorted by synth losses
            # picking the best ones to use for the experiment 
            order = l[bname]['order'][0:n_ex_use]
            # each of these exemplars can be used exactly once in whole experiment 
            # (across all image types).
            # going to randomize how these exemplars are assigned to sessions. 
            rndorder = order[np.random.permutation(len(order))]
            ex_each_sess[bname] = np.reshape(rndorder, [n_trials_per_concept, n_sessions])
                
        for sess in range(n_sessions):

            print('making rand order %d, session %d'%(rand, sess))
                
    
            trial_info = pd.DataFrame({'trial_num_overall': np.zeros((n_trials_per_session,)), 
                           'trial_in_run': np.zeros((n_trials_per_session,)), 
                           'trial_in_miniblock': np.zeros((n_trials_per_session,)), 
                            'run_number': np.zeros((n_trials_per_session,)), 
                            'miniblock_number_overall': np.zeros((n_trials_per_session,)), 
                            'miniblock_number_in_run': np.zeros((n_trials_per_session,)), 
                            'session_number': np.zeros((n_trials_per_session,)), 
                            'image_set_num': np.zeros((n_trials_per_session,)), 
                            'random_order_number': np.zeros((n_trials_per_session,)), 
                           'categ_ind': np.zeros((n_trials_per_session,)),
                          'concept_ind': np.zeros((n_trials_per_session,)),
                          'super_name': np.zeros((n_trials_per_session,)),
                          'basic_name': np.zeros((n_trials_per_session,)),
                          'ex_num': np.zeros((n_trials_per_session,)),
                          'ex_num_actual': np.zeros((n_trials_per_session,)),
                          'image_type_num': np.zeros((n_trials_per_session,)),
                          'image_type': np.zeros((n_trials_per_session,)),
                          'image_name': np.zeros((n_trials_per_session,)),
                          'dropbox_url': np.zeros((n_trials_per_session,)),
                          'cue_level_num': np.zeros((n_trials_per_session,)),
                          'cue_level': np.zeros((n_trials_per_session,)),
                          'cue_name': np.zeros((n_trials_per_session,)),
                          'distractor_name': np.zeros((n_trials_per_session,)),
                          'left_name': np.zeros((n_trials_per_session,)),
                          'right_name': np.zeros((n_trials_per_session,)),
                          'correct_resp': np.zeros((n_trials_per_session,)), 
                          })
    
            image_type_names = ['orig', 'pool1','pool2','pool3','pool4']
    
            tt=-1
            for ca in range(n_categ_use):
    
                for co in range(n_concepts_use):
    
                    bname = concepts_use[ca][co]
                    
                    ex = -1

                    # this is the set of exemplars assigned to this session.
                    # they will be randomly assigned to trial types
                    # for example ex01 might be used for intact or pool4, but never both
                    ex_use_list = ex_each_sess[bname][:,sess]
                   
                    for typ in range(n_image_types):
                        for cue in range(n_cue_levels):
    
                            tt+=1
                            if np.mod(tt,100)==0:
                                print('proc trial %d of %d'%(tt, n_trials_per_session))
    
                            trial_info['trial_num_overall'].iloc[tt] = tt
                            trial_info['categ_ind'].iloc[tt] = ca
                            trial_info['concept_ind'].iloc[tt] = co
                            trial_info['super_name'].iloc[tt] = categ_use[ca]
                            trial_info['basic_name'].iloc[tt] = concepts_use[ca][co].replace(' ', '_')
    
                            trial_info['image_type_num'].iloc[tt] = typ
                            trial_info['image_type'].iloc[tt] = image_type_names[typ]
                            trial_info['cue_level_num'].iloc[tt] = cue
                            trial_info['cue_level'].iloc[tt] = cue_levels[cue]
    
                            trial_info['image_set_num'].iloc[tt] = 0
                            trial_info['random_order_number'].iloc[tt] = rand
                            trial_info['session_number'].iloc[tt] = sess
    
                            ex += 1
                            # this is the exemplar index within the experiment
                            trial_info['ex_num'].iloc[tt] = ex
    
                            # this is the actual number of the exemplar, 0-40
                            ex_num_actual = ex_use_list[ex]
                            trial_info['ex_num_actual'].iloc[tt] = ex_num_actual
    
                            ex_name = 'ex%02d'%(ex_num_actual)
                            
                            if image_type_names[typ]=='orig':
    
                                trial_info['image_name'].iloc[tt] = os.path.join(bname, ex_name, 'orig.png')
                                dbxpath = '/'+os.path.join(stimuli_folder, bname, ex_name, 'orig.png')
    
                            else:
    
                                trial_info['image_name'].iloc[tt] = os.path.join(bname, ex_name,
                                                                                 'scramble_upto_%s.png'%\
                                                                                 (image_type_names[typ]))
                                dbxpath = '/'+os.path.join(stimuli_folder, bname, ex_name,\
                                                       'scramble_upto_%s.png'%image_type_names[typ])

                            # print(dbxpath)
                            sys.stdout.flush()
                            if dbxpath in dbxpath_dict.keys():
                                # don't have to re-create url, faster
                                url = dbxpath_dict[dbxpath]
                            else:
                                print('creating dropbox url')
                                print(dbxpath)
                                print(dbx)
                                # url = ''
                                url = dropbox_utils.get_shared_url(dbxpath, dbx)
                                dbxpath_dict[dbxpath] = url
                                print(url)
                            
                            trial_info['dropbox_url'].iloc[tt] = url   
              
            
            # figure out the sequence of which task happens when

            # this value is always alternating across sessions
            odd_even += 1
            odd_even = np.mod(odd_even,2)
            print(odd_even)
            
            miniblock_list = np.arange(n_mini_total)
            miniblock_run = np.repeat(np.arange(n_runs), n_mini_per_run)
    
            # miniblock tasks will always alternate, within a run
            miniblock_task = np.roll(np.tile(np.arange(n_cue_levels), [int(n_mini_total/2),]), odd_even)
    
            # flip the ordering of task within the "odd" runs
            # so that every other run starts with coarse/fine miniblock
            for rr in np.arange(1, n_runs, 2):
                miniblock_task[miniblock_run==rr] = np.flipud(miniblock_task[miniblock_run==rr])
    
            # decide which super-category is being tested on each run of the basic task
            # for the super task, this isn't relevant
            miniblock_supcat = np.zeros_like(miniblock_list)
            miniblock_supcat[miniblock_task==1] = -1
    
            # want to find a way of assigning super-categories to miniblocks
            # so that we don't have the same super-category twice in a run
            # use brute force randomization to find good sequence
            good_seq = False
            max_iter = 100
            ii = 0;
            while (not good_seq) and (ii < max_iter):
    
                supcat_seq = np.repeat(np.arange(n_categ_use), n_mini_per_categ)
                supcat_seq = supcat_seq[np.random.permutation(len(supcat_seq))]
                miniblock_supcat[miniblock_task==0] = supcat_seq
    
                un_cats = [np.unique(miniblock_supcat[(miniblock_task==0) & (miniblock_run==rr)]) \
                           for rr in range(n_runs)]
                good_seq = np.all([len(u)==2 for u in un_cats])
    
                ii+=1
                
                
            # now group the trials into runs based on which task is being done
                        
            # first basic-level tasks, within each superordinate   
            for si, sname in enumerate(categ_use):
    
                for typ in range(n_image_types):
    
                    trial_inds = np.array((trial_info['super_name']==sname) & \
                                          (trial_info['cue_level']=='basic') & \
                                          (trial_info['image_type_num']==typ))
    
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
    
    
                # now grab the trials for all image types, within this superordinate categ
                trial_inds = np.array((trial_info['super_name']==sname) & \
                                          (trial_info['cue_level']=='basic'))
    
                # break all these into equal sized "mini-blocks" randomly
                # assigning to the block numbers that we already decided on previously
                miniblock_inds_use = np.repeat(miniblock_list[miniblock_supcat==si], n_trials_mini)
                run_inds_use = np.repeat(miniblock_run[miniblock_supcat==si], n_trials_mini)
    
                rand_order = np.random.permutation(len(miniblock_inds_use))
                miniblock_inds_use = miniblock_inds_use[rand_order]
                run_inds_use = run_inds_use[rand_order]
    
                # print(miniblock_inds_use )
                trial_info['run_number'].iloc[trial_inds] = run_inds_use
                trial_info['miniblock_number_overall'].iloc[trial_inds] = miniblock_inds_use
                trial_info['miniblock_number_in_run'].iloc[trial_inds] = np.mod(miniblock_inds_use, n_mini_per_run)
    
    
            # now organizing the superordinate task runs
            # looping over image types, things will be shuffled within image type
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
    
            # now grab all the trials in superordinate task, all image types
            trial_inds = np.array(trial_info['cue_level']=='super')
    
            # break all these into equal sized "mini-blocks" randomly
            # assigning to the block numbers that we already decided on previously
            miniblock_inds_use = np.repeat(miniblock_list[miniblock_task==1], n_trials_mini)
            run_inds_use = np.repeat(miniblock_run[miniblock_task==1], n_trials_mini)
    
            rand_order = np.random.permutation(len(miniblock_inds_use))
            miniblock_inds_use = miniblock_inds_use[rand_order]
            run_inds_use = run_inds_use[rand_order]
    
            # print(miniblock_inds_use)
    
            trial_info['run_number'].iloc[trial_inds] = run_inds_use
            trial_info['miniblock_number_overall'].iloc[trial_inds] = miniblock_inds_use
            trial_info['miniblock_number_in_run'].iloc[trial_inds] = np.mod(miniblock_inds_use, n_mini_per_run)
    
    
            # organize the trials by mini-block number
            trial_info.sort_values(by='miniblock_number_overall', inplace=True)
    
            # now shuffle within each mini-block
            for mb in np.unique(trial_info['miniblock_number_overall']):
    
                inds = np.where(np.array(trial_info['miniblock_number_overall']==mb))[0] 
                shuff_order = np.random.permutation(len(inds))
                trial_info.iloc[inds] = trial_info.iloc[inds[shuff_order]]
    
            # assign trial numbers 
            trial_info['trial_in_miniblock'] = np.tile(np.arange(n_trials_mini), [n_mini_total,])
            trial_info['trial_in_run'] = np.tile(np.arange(n_trials_per_run), [n_runs,])
            trial_info['trial_num_overall'] = np.arange(n_trials_per_session)
    
            trial_info.set_index(np.arange(n_trials_per_session))
    
    
            # double check everything
            print('checking trial info')
            check_trial_info(trial_info, concepts_use, categ_use)

            trial_info_allsess += [trial_info]


        # concatenating the trials for all sessions together
        trial_info_allsess = pd.concat(trial_info_allsess)

        # fix the indexing
        trial_info_allsess.set_index(np.arange(n_trials_per_session * n_sessions))

        # making a variable that tracks all runs across all sessions
        trial_info_allsess['run_number_overall'] = trial_info_allsess['run_number'] + trial_info_allsess['session_number']*n_runs

        # checking that all exemplars are unique across whole expt
        
        names = trial_info_allsess['image_name']
        bnames = [n.split('/')[0] for n in names]
        enames = [n.split('/')[1] for n in names]
        
        un = np.unique(np.array([bnames, enames]), axis=1)
        assert(un.shape[1]==len(bnames))


        # save everything to a single CSV file
        # this has all sessions included
        expt_design_folder = os.path.join(project_root, 'expt_design', expt_name)
        if not os.path.exists(expt_design_folder):
            os.makedirs(expt_design_folder)
        trialinfo_filename1 =  os.path.join(expt_design_folder, 'trial_info_randorder%d.csv'%(rand))
        print('saving to %s'%trialinfo_filename1)
        sys.stdout.flush()
        
        trial_info_allsess.to_csv(trialinfo_filename1, index=False)

        # putting into bigger list here
        trial_info_list += [trial_info_allsess]

        
    # making .js files for use in jsPsych
    # (this file holds all the runs)
    js_filename = os.path.join(expt_design_folder, 'trialseq.js')
    print('saving to %s'%(js_filename))
    expt_utils.make_runs_js_multiplesets(trial_info_list, js_filename, var_name='info')
    
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

    # for each basic block checking that the cue and distractor names are all part of a single superord categ
    for mb in np.unique(ti['miniblock_number_overall'].iloc[inds_check]):
        inds_check_mb = inds_check & (np.array(ti['miniblock_number_overall']==mb))
        sname = np.array(ti['super_name'].iloc[inds_check_mb])
        assert(np.all(sname==sname[0]))
        sind = np.where(np.array(categ_use)==sname[0])[0][0]
        assert(np.all(np.isin(np.array(ti['distractor_name'].iloc[inds_check_mb]), concepts_use[sind])))
        assert(np.all(np.isin(np.array(ti['cue_name'].iloc[inds_check_mb]), concepts_use[sind])))

    # check superordinate names 
    inds_check = np.array(ti['cue_level']=='super')
    assert(np.all(np.isin(np.array(ti['cue_name'].iloc[inds_check]), categ_use)))
    assert(np.all(np.isin(np.array(ti['distractor_name'].iloc[inds_check]), categ_use)))

    # check to make sure individual attributes are distributed evenly across trials 
    n_trials_per_session = ti.shape[0]

    attr_check_even = ['super_name', 'basic_name', 'ex_num', 'image_type_num', \
                       'correct_resp', 'cue_level', 'run_number', \
                       'miniblock_number_overall', 'miniblock_number_in_run']
    for attr in attr_check_even:

        # should be an equal number of each thing 
        un, counts = np.unique(ti[attr], return_counts=True)
        # print(un, counts)
        assert(np.all(counts==n_trials_per_session/len(un)))

    
    # check the counterbalancing over multiple attributes

    # there should be an equal number of trials in each of the combinations of these 
    # different attribute "levels". for example each combination of category/image type. 

    attr_balanced = [ti['categ_ind'], ti['concept_ind'], ti['image_type_num'], ti['cue_level']]
    attr_balanced_inds = np.array([np.unique(attr, return_inverse=True)[1] for attr in attr_balanced]).T

    n_levels_each = [len(np.unique(attr)) for attr in attr_balanced]
    n_combs_expected = np.prod(n_levels_each)
    n_repeats_expected = n_trials_per_session/n_combs_expected

    un_rows, counts = np.unique(attr_balanced_inds,axis=0, return_counts=True)

    assert(un_rows.shape[0]==n_combs_expected)
    assert(np.all(counts==n_repeats_expected))

    
    
    # check evenness of stuff for super task only

    inds_check = np.array(ti['cue_level']=='super')
    n_trials_per_session = np.sum(inds_check)

    attr_check_even = ['super_name', 'cue_name', 'image_type', 'distractor_name', 'correct_resp']

    for attr in attr_check_even:

        # should be an equal number of each thing 
        un, counts = np.unique(ti[attr].iloc[inds_check], return_counts=True)
        # print(un, counts)
        assert(np.all(counts==n_trials_per_session/len(un)))



    # check evenness of stuff for basic task only

    inds_check = np.array(ti['cue_level']=='basic')
    n_trials_per_session = np.sum(inds_check)

    attr_check_even = ['basic_name', 'cue_name', 'image_type', 'distractor_name', 'correct_resp']

    for attr in attr_check_even:

        # should be an equal number of each thing 
        un, counts = np.unique(ti[attr].iloc[inds_check], return_counts=True)
        # print(un, counts)
        assert(np.all(counts==n_trials_per_session/len(un)))

    # making sure that for the different superordinate tasks, 
    # the miniblocks all fall on different runs
    for si, sname in enumerate(categ_use):
        
        inds = (ti['cue_level']=='basic') & (ti['super_name']==sname)
        assert(len(np.unique(ti['run_number'][inds])) == 4)



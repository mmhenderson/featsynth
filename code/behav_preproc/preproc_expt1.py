import numpy as np
import pandas as pd
import os, sys
from utils import date_utils

project_root = '/user_data/mmhender/featsynth/'
expt_name = 'expt1'

stimulus_dur_ms = 500 # need this to get accurate RTs
intact_acc_threshold = 0.75 # use to filter out very bad subjects
n_runs_expected = 10;

def preproc_all():

    """
    Load/process all the data from this version of experiment.
    """
    
    # find the data
    data_folder = os.path.join(project_root, 'online_data', expt_name)
  
    # store preprocessed files in this folder
    preproc_folder = os.path.join(data_folder, 'preproc')
    if not os.path.exists(preproc_folder):
        os.makedirs(preproc_folder)

    # there might be multiple folders to look through, diff expt versions.
    folders = os.listdir(data_folder)
    folders = [f for f in folders if os.path.isdir(os.path.join(data_folder,f))]
    folders = [f for f in folders if 'data' in f and 'old' not in f]

    # load all raw data files
    raw_data = pd.DataFrame()
    raw_worker_ids = pd.DataFrame()

    for ff, folder in enumerate(folders):

        # this is the main task data
        string1 = 'task-sq8b'
        subfolder = os.path.join(data_folder, folders[ff])
        files = os.listdir(subfolder)
        files = [f for f in files if string1 in f and '.csv' in f]
        task_filename = os.path.join(subfolder,files[0])
        print(task_filename)

        r = pd.read_csv(task_filename)
        raw_data = pd.concat([raw_data, r])

        # this is the worker-id collection task
        string2 = '89gk'
        subfolder = os.path.join(data_folder, folders[ff])
        files = os.listdir(subfolder)
        files = [f for f in files if string2 in f and '.csv' in f]
        if len(files)>0:
            worker_id_filename = os.path.join(subfolder,files[0])
            r = pd.read_csv(worker_id_filename)
            raw_worker_ids = pd.concat([raw_worker_ids, r])

    raw_data.set_index(np.arange(raw_data.shape[0]));
    raw_worker_ids.set_index(np.arange(raw_worker_ids.shape[0]));

    # define "good subjects" (those that finished whole experiment and got completion codes)
    public_ids = np.array(raw_data['Participant Public ID'])
    public_ids = [p if isinstance(p,str) else '' for p in public_ids]
    public_ids = np.unique(public_ids)
    public_ids = np.array([p for p in public_ids if len(p)>1])

    is_good = []
    for pub_id in public_ids:
        inds = np.where(raw_data['Participant Public ID']==pub_id)[0]
        d = raw_data.iloc[inds]

        code = np.array(d['Participant Completion Code'])
        has_code = np.any([isinstance(c,str) for c in code])

        is_good += [has_code]
        
    # get ids of all subjects that finished
    good_ids = public_ids[np.array(is_good)]
    
    # cross-check this list of completed subjects against the csv file of participants from gorilla
    fn = os.path.join(data_folder, 'participants_all.csv')
    p = pd.read_csv(fn)
    pubids_check = p['PublicID'][p['Status']=='Complete']
    pubids_check = np.sort(list(pubids_check))

    # make sure all ids from 'participants_all.csv' are in the actual data
    assert(np.all([p in good_ids for p in pubids_check]))
    
    # make .csv files listing the good subjects only (finished all trials)
    good_df = make_subject_df(raw_data, raw_worker_ids, good_ids)
    fn2save = os.path.join(preproc_folder,'good_sub_list.csv')
    print('saving to %s'%fn2save)
    good_df.to_csv(fn2save)
    good_worker_ids = np.array(good_df['Worker ID mTurk'])
    
    # make .csv for all subjects (including incomplete)
    all_df = make_subject_df(raw_data, raw_worker_ids, public_ids)
    fn2save = os.path.join(preproc_folder,'all_sub_list.csv')
    print('saving to %s'%fn2save)
    all_df.to_csv(fn2save)
    
    # now gathering the actual trial-by-trial info for each subject, 
    # which will be saved to disk for the good subjects.
    trial_data_all = pd.DataFrame()
    good_subject_count = 0 # keep track of how many have > threshold accuracy
    
    print(good_ids)
    
    for si, ss in enumerate(good_ids):
        
        
        # get data for this participant
        inds = np.where(raw_data['Participant Public ID']==ss)[0]
        d = raw_data.iloc[inds,:]
        
        pub_id = np.array(d['Participant Public ID'])[0]
        print('\nsubject %d, id: %s'%(si, pub_id))
        date = np.array(good_df['Expt Start Time (PST)'])[good_df['Gorilla public ID']==ss]
        print('date = %s'%date)
        
        # check all runs are done
        rnums = np.array(d['run_number'])
        rnums = rnums[~np.isnan(rnums)]
        runs_done = np.unique(rnums).astype(int)
        runs_expected = np.arange(1,n_runs_expected+1)
        all_runs_done = np.all(np.isin(runs_expected, runs_done))
        if not all_runs_done:
            # this happened once where there is a run missing because gorilla accidentally 
            # showed the same run number (same images) to a subject twice and skipped
            # another one. shouldn't happen. 
            run_missing = runs_expected[~np.isin(runs_expected, runs_done)]
            print('subject is missing runs:')
            print(run_missing)
            print('skipping subject')
            continue

        # sort by time. sometimes they are out of order, sometimes not...
        t = d['Local Timestamp']
        d = d.iloc[np.argsort(t)]

        trial_data = preproc_data(d)

        # as quality control, check accuracy on the intact condition. subjects
        # should all be high on this since it's easy.
        intact_data = trial_data[trial_data['image_type']=='orig']
        intact_acc = np.mean(intact_data['resp']==intact_data['correct_resp'])

        print('intact acc: %.2f'%intact_acc)
        if intact_acc<intact_acc_threshold:

            # don't save the low-performing subjects
            print('skipping this subject')
            continue

        else:

            good_subject_count += 1
            trial_data['subject']=np.full(fill_value=good_subject_count, shape=[trial_data.shape[0],])
            trial_data['gorilla_pub_id']=np.full(fill_value=ss, shape=[trial_data.shape[0],])
            trial_data['worker_id']=np.full(fill_value=good_worker_ids[si], shape=[trial_data.shape[0],])
            
            trial_data_all = pd.concat([trial_data_all, trial_data])

    trial_data_all.set_index(np.arange(trial_data_all.shape[0]));
    
    # get some basic counts here
    d = trial_data_all
    subinds = np.unique(d['subject'])
    sinds = [np.where(np.array(d['subject'])==ss)[0][0] for ss in subinds]
    cbinds = [np.array(d['which_cb'])[ii] for ii in sinds]
    u, counts = np.unique(cbinds, return_counts=True)
    print('\nnumber of good subjects (above acc %.2f): %d'%(intact_acc_threshold, len(subinds)))
    print('set 1, set 2: [%d, %d]'%(counts[0],counts[1]))
    
    fn2save = os.path.join(preproc_folder, 'preproc_data_all.csv')
    print('saving to %s'%fn2save)
    trial_data_all.to_csv(fn2save)
    
    

def make_subject_df(raw_data, raw_worker_ids, public_ids):

    """ 
    Making list of all the subjects who participated, and some basic stats for each.
    """
    
    colnames = ['Gorilla public ID', 'Survey Completion Code', \
                'Worker ID mTurk',\
                'Expt Start Time (PST)', 'Expt End Time (PST)', 'Total number of trials', \
                'Avg Acc', 'Avg RT (ms)',\
               'Experiment version num', 'Task version num', \
               'Stim height pix',]

    df = pd.DataFrame(columns=colnames)

    for ii, pub_id in enumerate(public_ids):

        inds = np.where(raw_data['Participant Public ID']==pub_id)[0]

        d = raw_data.iloc[inds]

        start_run_inds = np.where(~np.isnan(d['subject_id_rnd']))[0][0::2]
        if len(start_run_inds)==0:
            print(pub_id)
            continue
        stop_run_inds = np.where(~np.isnan(d['subject_id_rnd']))[0][1::2]
        finish_inds = stop_run_inds+1
        run_acc = np.array(d['total_acc'])[finish_inds]
        run_avg_rts = np.array(d['average_rt'])[finish_inds]
        acc = np.nanmean(run_acc).round(2)
        rt = np.nanmean(run_avg_rts).round(2)
        
        stim_height_pix = d['stim_height_pix'].iloc[start_run_inds[0]]
        
        
        trial_nums = np.array(d['trial_in_run'])
        trial_nums = trial_nums[~np.isnan(trial_nums)]
        n_trials = int(len(trial_nums)/3)

        code = np.array(d['Participant Completion Code'])[0]

        # converting to PST here
        start = date_utils.adjust_datetime_str(np.min(d['UTC Date']), -8)
        end = date_utils.adjust_datetime_str(np.max(d['UTC Date']), -8)

        subject_inds = [pub_id==pid for pid in raw_worker_ids['Participant Public ID']]
        if sum(subject_inds)>0:
            inds = np.where(subject_inds & (raw_worker_ids['Question Key']=='worker-id'))[0][0]
            wid = raw_worker_ids['Response'].iloc[inds]
        else:
            wid = 'nan'

        task_version = np.array(d['Task Version'])[0]
        expt_version = np.array(d['Experiment Version'])[0]

        vals = np.array([[pub_id, code, wid, start, end, n_trials, \
                          acc, rt, \
                          expt_version, task_version, stim_height_pix]])
        df = pd.concat([df, pd.DataFrame(vals, columns=colnames, index=[ii])],\
                        axis=0)
    return df


def preproc_data(data):

    """
    Preprocess trial-by-trial attributes for a single subject at a time.
    Returns trial_data, [n_trials x n_attributes]
    """
    
    start_run_inds = np.where(~np.isnan(data['subject_id_rnd']))[0][0::2]
    stop_run_inds = np.where(~np.isnan(data['subject_id_rnd']))[0][1::2]
    finish_inds = stop_run_inds+1

    # get some basic stats here
    run_acc = np.array(data['total_acc'])[finish_inds]
    run_avg_rts = np.array(data['average_rt'])[finish_inds]
    print('run accuracies, avg RTs:')
    print(run_acc, run_avg_rts)
    
    run_numbers = np.array(data['run_number'])[start_run_inds].astype(int)
   
    # more info about what sequence this subject did
    which_cb = int(np.array(data['which_counterbal'])[start_run_inds[0]])
    random_order_number = int(np.array(data['random_order_number'])[start_run_inds[0]])
  
    print('counterbalance cond: %d'%which_cb)
    print('random order number: %d'%random_order_number)
    # from disk, load the sequence for the trials that were shown to this participant.
    # this was pre-made before experiment was run.
    # can double check that everything saved from gorilla matches what was in here.
    expt_design_folder = os.path.join(project_root, 'expt_design', expt_name)
    info_filename = os.path.join(expt_design_folder, 'trial_info_counterbal%d_randorder%d.csv'%(which_cb, random_order_number))
    info = pd.read_csv(info_filename)

    n_trials = np.sum(data['is_stim'])
    assert(info.shape[0]==n_trials)

    # make the trial_data df, starting with what is in info, and adding specific 
    # data from this subject's performance.
    trial_data = info
    rts = np.array(data['Reaction Time'])
    stim_rts = rts[data['is_stim']==True]
    iti_rts = rts[data['is_iti']==True]

    trial_data['run_number'] = np.repeat(run_numbers, int(n_trials/len(run_numbers)))
    
    rts = stim_rts
    # if they responded during ITI, we adjust their RT based on stimulus duration.
    rts[np.isnan(stim_rts)] = iti_rts[np.isnan(stim_rts)] + stimulus_dur_ms

    trial_data['rt'] = rts

    resp = np.array(proc_resp_strs(data['Response']))

    stim_resp = resp[data['is_stim']==True]
    iti_resp = resp[data['is_iti']==True]
    resp = stim_resp
    resp[np.isnan(stim_resp)] = iti_resp[np.isnan(stim_resp)]

    resp[np.isnan(resp)] = -1
    trial_data['resp'] = resp.astype(int)

    correct_resp = np.array(data['correct_response'])
    correct_resp = correct_resp[data['is_stim']==True].astype(int)

    # double check that trial sequence is correct between data and trial info csv
    correct_resp_check = 2-info['target_present'].astype(int)
    if not np.all(correct_resp_check==correct_resp):
        print(info_filename)
        print(correct_resp, correct_resp_check)

    assert(np.all(correct_resp_check==correct_resp))

    trial_data['correct_resp'] = correct_resp

    trial_data['correct'] = trial_data['correct_resp']==trial_data['resp']

    # add some more info for this subject, same values for all trials
    trial_data['which_cb'] = np.full(fill_value=which_cb, shape=[trial_data.shape[0],])
    trial_data['random_order_number'] = np.full(fill_value=random_order_number, \
                                                shape=[trial_data.shape[0],])
    ppd = np.array(data['pixels_per_degree'])
    ppd = ppd[~np.isnan(ppd)][0]
    trial_data['pixels_per_degree'] = np.full(fill_value=ppd, \
                                                shape=[trial_data.shape[0],])
    
    return trial_data


def proc_resp_strs(resp, poss_resp = ['1','2']):

    resp = [r if isinstance(r, str) else str(r) for r in resp]
    proc_resp = np.array([int(r) if r in poss_resp else np.nan for r in resp])

    return proc_resp

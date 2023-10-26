import numpy as np

from itertools import permutations
import warnings
warnings.filterwarnings('ignore')

def make_runs_js(df, js_filename, var_name='info'):

    # the .js file will contain a var "info", 
    # an array where each element is one run. 
    # info[run_number] will be the exact trial sequence that jsPsych needs to do the run.
    
    with open(js_filename,'w') as f:

        f.write('var %s = [\n'%var_name)

        run_numbers = np.unique(np.array(df['run_number']))
        
        for rr in run_numbers:

            ti_thisrun = df.iloc[np.array(df['run_number'])==rr]
            n_trials = ti_thisrun.shape[0]

            f.write('    [\n')
            for tt in range(n_trials):

                trial_info = dict(ti_thisrun.iloc[tt])

                f.write('        {\n')
                for key,val in zip(trial_info.keys(), trial_info.values()):

                    if isinstance(val, str):
                        f.write('        \'%s\': \'%s\',\n'%(key, val))
                    else:
                        f.write('        \'%s\': %d,\n'%(key, val))

                f.write('        },\n')
            f.write('    ],\n')

        f.write('];\n')
        
def make_runs_js_multiplesets(df, js_filename, var_name='info'):

    # the .js file will contain a var "info", 
    # an array or arrays where each element is one run. 
    # info[random_order_number][run_number] will be the exact trial sequence that jsPsych needs to do the run.
    # [df] is a list of the randomized trial orders.
    
    n_random_orders = len(df)
    
    with open(js_filename,'w') as f:

        # open list of random orders
        f.write('var %s = [\n'%var_name)
        
        for rand in range(n_random_orders):
            
            # open list of runs for this random order
            f.write('    [\n')

            run_numbers = np.unique(np.array(df[rand]['run_number']))

            for rr in run_numbers:

                ti_thisrun = df[rand].iloc[np.array(df[rand]['run_number'])==rr]
                n_trials = ti_thisrun.shape[0]

                # open list of trials for this run
                f.write('    [\n')
                for tt in range(n_trials):

                    trial_info = dict(ti_thisrun.iloc[tt])

                    # open trial
                    f.write('        {\n')
                    for key,val in zip(trial_info.keys(), trial_info.values()):

                        if isinstance(val, str):
                            f.write('        \'%s\': \'%s\',\n'%(key, val))
                        else:
                            f.write('        \'%s\': %d,\n'%(key, val))

                    # close trial
                    f.write('        },\n')
                    
                # close run    
                f.write('    ],\n')
                
            # close list of runs
            f.write('    ],\n')

        # close list of random orders
        f.write('];\n')
        
        
def make_trial_js(df, js_filename, var_name='info'):
    
    n_trials = df.shape[0]

    with open(js_filename,'w') as f:

        f.write('var %s = [\n'%var_name)

        for tt in range(n_trials):
            
            trial_info = dict(df.iloc[tt])

            f.write('    {\n')
            for key,val in zip(trial_info.keys(), trial_info.values()):

                if isinstance(val, str):
                    f.write('    \'%s\': \'%s\',\n'%(key, val))
                else:
                    f.write('    \'%s\': %d,\n'%(key, val))

            f.write('    },\n')

        f.write('];\n')
        
        

def make_runs_for_matlab(df, js_filename, var_name='info'):

    # the .json file will contain a var "info", 
    # an array or arrays where each element is one run. 
    # info[random_order_number][run_number] will be the exact trial sequence that jsPsych needs to do the run.
    # [df] is a list of the randomized trial orders.
    
    n_random_orders = len(df)
    
    with open(js_filename,'w') as f:

        # open list of random orders
        f.write('[\n')
        
        for rand in range(n_random_orders):
            
            # open list of runs for this random order
            f.write('    [\n')

            run_numbers = np.unique(np.array(df[rand]['run_number_overall']))

            print(run_numbers)

            for rr in run_numbers:

                ti_thisrun = df[rand].iloc[np.array(df[rand]['run_number_overall'])==rr]
                n_trials = ti_thisrun.shape[0]

                # open list of trials for this run
                f.write('    [\n')
                for tt in range(n_trials):

                    trial_info = dict(ti_thisrun.iloc[tt])

                    # open trial
                    f.write('        {\n')
                    n_keys = len(trial_info.keys())
                    for ki, [key,val] in enumerate(zip(trial_info.keys(), trial_info.values())):

                        if isinstance(val, str):
                            f.write('        \"%s\": \"%s\"'%(key, val))
                        else:
                            f.write('        \"%s\": %d'%(key, val))

                        if ki==(n_keys-1):
                            f.write('\n')
                        else:
                            f.write(',\n')
                            
                    # close trial
                    if tt==(n_trials-1):
                        f.write('        }\n')
                    else:
                        f.write('        },\n')
                    
                    
                # close run   
                if rr==(run_numbers[-1]):
                    f.write('    ]\n')
                else:
                    f.write('    ],\n')

            # close list of runs
            if rand==(n_random_orders-1):   
               f.write('    ]\n')
            else:
                f.write('    ],\n')

        # close list of random orders
        f.write(']\n')
        


def swap_rand_pairs(sequence):
    
    n = len(sequence)
    assert(np.mod(n, 2)==0)
    
    randpairs = np.floor(np.random.permutation(np.arange(n))/2).astype(int)
    new_sequence = np.zeros_like(sequence)
    
    for ii in range(int(n/2)):
        
        pair_inds = randpairs==ii
        new_sequence[pair_inds] = np.flip(sequence[pair_inds])
        
    return new_sequence

def shuffle_nosame(sequence, rndseed = None):
    
    """
    Shuffle a list pseudo-randomly, making sure that no element has the same 
    value in the new and old list.
    All list values must be unique.
    """
    
    unique_vals, uncounts = np.unique(sequence, return_counts=True)
    assert(len(unique_vals)==len(sequence))
    n_vals = len(sequence)
    
    if rndseed is not None:
        np.random.seed(rndseed)
        
    if np.mod(n_vals, 2)==0:
        # for even number, can use this algorithm
        new_sequence = swap_rand_pairs(sequence)
    elif n_vals<=7:
        n_total = len(sequence)
        # list all the ways to randomly permute groups of elements without having any in 
        # their original position.
        all_perms = list(permutations(np.arange(n_vals)))
        good_perms = [perm for perm in all_perms if np.all(np.array(perm)!=(np.arange(n_vals)))]
        # pick at random one of the orders that works
        perm_order = np.array(good_perms[np.random.choice(np.arange(len(good_perms)))])
        new_sequence = sequence[perm_order]
    else:
        # split in half and recurse
        rand_order = np.random.permutation(n_vals)
        groups = (rand_order<np.floor(n_vals/2)).astype(int)
        new_sequence = np.zeros(np.shape(sequence), sequence.dtype)
        for gg in [0,1]:
            inds = groups==gg
            new_sequence[inds] = shuffle_nosame(sequence[inds])
     
    assert(not np.any(sequence==new_sequence))
    assert(np.all(np.unique(sequence)==np.unique(new_sequence)))
    
    return new_sequence
    
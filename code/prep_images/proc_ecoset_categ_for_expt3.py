import os
import sys
import numpy as np
import pandas as pd
import scipy.io
import PIL.Image


things_stim_path = '/user_data/mmhender/stimuli/things/'
ecoset_path = '/lab_data/tarrlab/common/datasets/Ecoset/'
ecoset_info_path = '/user_data/mmhender/stimuli/ecoset_info/'

# these are just the categories we used in behav expt 3
# slightly diff from final set of 64 categs

def choose_ecoset_categs_step1():

    """
    Choosing the set of 8 superordinate categories to use, and listing 
    all the basic-level categories available in each.
    Only using basic-level categories that appear in both ecoset and things.
    Need to run choose_ecoset_categs() after this to choose final 64 categs.
    """
    
    # load the list of all concepts (basic-level) in THINGS dataset
    filename = os.path.join(things_stim_path,'things_concepts.tsv')
    df = pd.read_csv(filename, sep='\t')
    things_names = np.array(df['Word'])
    
    # load names of all categories (superord) in THINGS dataset
    info_folder = os.path.join(things_stim_path,'27 higher-level categories')
    categ_names = scipy.io.loadmat(os.path.join(info_folder, 'categories.mat'))['categories'][0]
    categ_names = np.array([categ_names[ii][0] for ii in range(len(categ_names))])
    
    # correspondence between things concepts (basic) and categories (superord)
    # this matrix goes [basic x super]
    dat = scipy.io.loadmat(os.path.join(info_folder, 'category_mat_manual.mat'))
    cmat = dat['category_mat_manual']
    cmat_use = cmat==1
    
    # i am manually adding some extra types of "furniture" here...this is because the existing
    # category "hammock" doesn't have that many images in ecoset. trying to find a bigger one.
    furniture_extras = ['lamp', 'radiator', 'refrigerator', 'sink', 'stove', 'television']
    sind = np.where(categ_names=='furniture')[0][0]
    for name in furniture_extras:
        bind = np.where(things_names==name)[0]
        cmat_use[bind,sind] = True
        
    dessert_extras = ['candy']
    sind = np.where(categ_names=='dessert')[0][0]
    for name in dessert_extras:
        bind = np.where(things_names==name)[0]
        cmat_use[bind,sind] = True

    # make a list of names to skip...

    # skip any that are the same name as superordinate, because that won't make sense
    things_inds_skip = np.isin(things_names, categ_names) 

    # removing any duplicate concept names here (these are ambiguous meaning words like bat)
    un, counts = np.unique(things_names, return_counts=True)
    duplicate_conc = un[counts>1]
    duplicate_conc_inds = [conc in duplicate_conc for conc in things_names]
    things_inds_skip = things_inds_skip | duplicate_conc_inds

    # skip some other ones that I found confusing/uncommon 
    #(cheese was included as dessert, avocado was fruit)
    conc_skip = ['earwig', 'cheese', 'avocado', 'anvil', 'okra']
    things_inds_skip = things_inds_skip | np.isin(things_names, conc_skip)

    things_names_skip = things_names[things_inds_skip]

    # load names of all categories in ecoset dataset
    ecoset_folders = np.load(os.path.join(ecoset_info_path, 'ecoset_names.npy'), \
                             allow_pickle=True).item()
    ecoset_names = np.array(list(ecoset_folders.keys()))

    # get overlap
    is_in_ecoset = [np.any([(t==e) for e in ecoset_names]) for t in things_names]

    
    
    # this is the list of 8 superord categories to use here
    # chosen based on which ones had at least 8 basic-level members in ecoset, 
    # and also trying to minimize overlap between categories. 
    superord_use = ['insect','vegetable','fruit','dessert', \
                 'tool','musical instrument','furniture', 'vehicle']
    superord_inds_use = np.array([np.where(categ_names==superord)[0][0] for superord in superord_use])

    # going to exclude any basic-level that occur in multiple superord categ
    used_once = np.sum(cmat_use[:,superord_inds_use], axis=1)==1

    inds_each = [(cmat_use[:,ci] & is_in_ecoset & ~things_inds_skip & used_once) \
                            for ci in superord_inds_use]

    counts_each = np.array([np.sum(inds) for inds in inds_each])

    # this is the resulting list of basic categ in each superord categ.
    # there are still too many here so we will further pare down based on file counts
    basic_names_each = [things_names[inds] for inds in inds_each]

    
    return superord_use, basic_names_each



def choose_ecoset_categs_oldversion():

    """
    Choosing the final set of 64 categories, 8 basic-level in each of 8 superordinate.
    OLD VERSION before we switched a few of the musical instrument categories.
    So this includes ukulele/mandolin instead of trumpet/cymbal.
    This is only used for one behavior experiment (expt4)
    """
    
    # load names of all categories in ecoset dataset
    ecoset_folders = np.load(os.path.join(ecoset_info_path, 'ecoset_names.npy'), \
                             allow_pickle=True).item()

    superord_use, basic_names_each = choose_ecoset_categs_step1()

    counts_each = np.array([len(bnames) for bnames in basic_names_each])

    # will subsample 8 basic from each superord
    n_basic_min = 8;
    assert(np.all(counts_each>=n_basic_min))

    # list of all files in each category
    fn = os.path.join(ecoset_info_path, 'ecoset_file_info.npy')
    efiles = np.load(fn, allow_pickle=True).item()

    # going to exclude a few categories, to help prevent ambiguity. 
    # these are all similar to another category in their superordinate.
    # HERE we are excluding the new music instrument categories, so we'll get the
    # older version of the musical instruments (include ukulele, manodolin).
    exclude = ['wasp', 'razor', 'melon', 'chocolate','cake','pudding','radish', \
          'trumpet', 'cymbal','kazoo']

    # choose whichever basic categories have the most images available
    basic_names_subsample = []
    for si in range(len(superord_use)):
        bnames = np.array(basic_names_each[si])
        bnames = bnames[~np.isin(bnames, exclude)]
        
        basic_counts = np.array([efiles[bname]['good_ims'] for bname in bnames])
        print(si, superord_use[si])
        # print(basic_names_each[si])
        # print(basic_counts)
        inds_use = np.flipud(np.argsort(basic_counts))[0:n_basic_min]
        bnames_use = np.array(bnames)[inds_use]
        print(bnames_use)
        print(basic_counts[inds_use])
        basic_names_subsample += [bnames_use]
    
    
    # organize into dict for saving
    sinfo = dict()
    binfo = dict()
    for si, sname in enumerate(superord_use):
        sinfo[sname] = dict()
        sinfo[sname]['super_name'] = sname
        sinfo[sname]['basic_names'] = basic_names_subsample[si]
        for bi, bname in enumerate(basic_names_subsample[si]):
            binfo[bname] = dict()
            binfo[bname]['basic_name'] = bname
            binfo[bname]['super_name'] = sname
            binfo[bname]['ecoset_folder'] = ecoset_folders[bname]

    # save
    fn2save = os.path.join(ecoset_info_path, 'categ_use_ecoset_OLDVERSION.npy')
    print('saving to %s'%fn2save)
    np.save(fn2save, {'binfo': binfo, 'sinfo': sinfo})
    

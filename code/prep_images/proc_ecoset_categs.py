import os
import sys
import numpy as np
import pandas as pd
import scipy.io
import PIL.Image


things_stim_path = '/user_data/mmhender/stimuli/things/'
ecoset_path = '/lab_data/tarrlab/common/datasets/Ecoset/'
ecoset_info_path = '/user_data/mmhender/stimuli/ecoset_info/'

def get_ecoset_names():

    """
    make a dict of names/folder names for all categories in ecoset.
    """
    ecoset_folder_names = os.listdir(os.path.join(ecoset_path, 'train'))
    ecoset_names = [e.split('_')[1] for e in ecoset_folder_names]

    # EDIT to make a few categories better match the names in THINGS
    # changing bugle to trumpet, and cymbals to cymbal
    # changed 10/17/23
    ind = np.where(np.array(ecoset_names)=='bugle')[0][0]
    ecoset_names[ind] = 'trumpet'
    
    ind = np.where(np.array(ecoset_names)=='cymbals')[0][0]
    ecoset_names[ind] = 'cymbal'

    # now make the dict of names to folders
    ecoset_folders = {k: v for k, v in zip(ecoset_names, ecoset_folder_names)}

    save_path = os.path.join(ecoset_info_path, 'ecoset_names.npy')
    np.save(save_path, ecoset_folders)
    
    
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


def get_ecoset_file_info(debug=0):

    """
    Gather info about what files are included in ecoset database.
    This will speed things up later on when figuring out what files to use.
    Also going to choose categories to use based on which have most images.
    """
    
    # which superordinate categories are we using?
    # for speed, only going to process images belonging to one of these.
    superord_use, basic_names_each = choose_ecoset_categs_step1()
    bnames = np.concatenate(basic_names_each, axis=0)

    print(bnames)
    
    debug = debug==1
    
    # this has the mapping from ecoset name to folder names
    fn = os.path.join(ecoset_info_path, 'ecoset_names.npy')
    info = np.load(fn, allow_pickle=True).item()
    
    fn2save = os.path.join(ecoset_info_path, 'ecoset_file_info.npy')
    
    if os.path.exists(fn2save):
        efiles = np.load(fn2save, allow_pickle=True).item()
    else:
        efiles = dict()
    
    for bi, bname in enumerate(bnames):
    
        if debug & (bi>0):
            continue
    
        print(bname)
        
        if bname in efiles.keys():
            print('done with %s, skipping'%bname)
            continue
            
        
        sys.stdout.flush()
        
        efiles[bname] = dict()

        folder = os.path.join(ecoset_path, 'train', info[bname])
        efiles[bname]['train'] = dict()
        imlist = os.listdir(folder)
        efiles[bname]['train']['images'] = imlist
        efiles[bname]['train']['size'] = [[] for ii in range(len(efiles[bname]['train']['images']))]
        efiles[bname]['train']['mode'] = ['' for ii in range(len(efiles[bname]['train']['images']))]

        for ii, imfile in enumerate(imlist):
            
            imfn = os.path.join(os.path.join(folder, imfile))
            im = PIL.Image.open(imfn)
            efiles[bname]['train']['size'][ii] = im.size
            efiles[bname]['train']['mode'][ii] = im.mode
            
        folder = os.path.join(ecoset_path, 'val', info[bname])
        efiles[bname]['val'] = dict()
        imlist = os.listdir(folder)
        efiles[bname]['val']['images'] = imlist
        efiles[bname]['val']['size'] = [[] for ii in range(len(efiles[bname]['val']['images']))]
        efiles[bname]['val']['mode'] = ['' for ii in range(len(efiles[bname]['val']['images']))]

        for ii, imfile in enumerate(imlist):
            
            imfn = os.path.join(os.path.join(folder, imfile))
            im = PIL.Image.open(imfn)
            efiles[bname]['val']['size'][ii] = im.size
            efiles[bname]['val']['mode'][ii] = im.mode
            
            
        folder = os.path.join(ecoset_path, 'test', info[bname])
        efiles[bname]['test'] = dict()
        imlist = os.listdir(folder)
        efiles[bname]['test']['images'] = imlist
        efiles[bname]['test']['size'] = [[] for ii in range(len(efiles[bname]['test']['images']))]
        efiles[bname]['test']['mode'] = ['' for ii in range(len(efiles[bname]['test']['images']))]

        for ii, imfile in enumerate(imlist):
            
            imfn = os.path.join(os.path.join(folder, imfile))
            im = PIL.Image.open(imfn)
            efiles[bname]['test']['size'][ii] = im.size
            efiles[bname]['test']['mode'][ii] = im.mode
            
    
        # now computing some basic stats/counts about the files
        sizes = efiles[bname]['train']['size'] + \
                efiles[bname]['val']['size'] + \
                efiles[bname]['test']['size']

        is_rgb = np.array(efiles[bname]['train']['mode'] + \
                         efiles[bname]['val']['mode'] + \
                         efiles[bname]['test']['mode'])=='RGB'

        # want images that are big enough and in RGB
        pix_thresh = 256
        abv_thresh = np.array([(s1>=pix_thresh) & (s2>=pix_thresh) for s1, s2 in sizes])
        ims_use_all = np.where(abv_thresh & is_rgb)[0]

        efiles[bname]['total_ims'] = len(sizes)
        efiles[bname]['good_ims'] = len(ims_use_all)
        
        
    print('saving to %s'%fn2save)
    np.save(fn2save, efiles)


def choose_ecoset_categs():

    """
    Choosing the final set of 64 categories, 8 basic-level in each of 8 superordinate.
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
    # also skipping kazoo because too many of the kazoo images have people
    exclude = ['wasp', 'razor', 'melon', 'chocolate','cake','pudding','radish', \
          'ukulele', 'mandolin','kazoo']

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
    fn2save = os.path.join(ecoset_info_path, 'categ_use_ecoset.npy')
    print('saving to %s'%fn2save)
    np.save(fn2save, {'binfo': binfo, 'sinfo': sinfo})



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
    


def get_ecoset_files_good():

    """
    Make a list of which files to use in each basic-level category, 
    excluding those that have "person" detected.
    This is about 500 per category (big), and not quality checked. 
    To get the smaller list of ~40 images that we use in experiment, 
    use choose_extra_ecoset_ims.py
    """
    
    # info about which categories to use
    fn = os.path.join(ecoset_info_path, 'categ_use_ecoset.npy')
    info = np.load(fn, allow_pickle=True).item()
    bnames = np.array(list(info['binfo'].keys()))

    
    # list of all files in each category
    fn = os.path.join(ecoset_info_path, 'ecoset_file_info.npy')
    efiles = np.load(fn, allow_pickle=True).item()

    # list of which images have person
    fn_person = os.path.join(ecoset_info_path, 'ecoset_files_detect_person.npy')
    has_person = np.load(fn_person, allow_pickle=True).item()


    pix_thresh = 256

    # where to save result 
    fn2save = os.path.join(ecoset_info_path, 'ecoset_files_use.npy')

    import choose_extra_ecoset_ims
    trumpet_fns = choose_extra_ecoset_ims.get_trumpet_filenames()
    trumpet_fns = [t.split('bugle/')[1] for t in trumpet_fns]
    
    efiles_adj = dict()

    for bi, bname in enumerate(bnames):

        efiles_adj[bname] = dict()

        # choose images to analyze here
        imfiles_all = efiles[bname]['train']['images']

        sizes = efiles[bname]['train']['size']
        is_rgb = np.array(efiles[bname]['train']['mode'])=='RGB'

        # want images that are big enough and in RGB
        abv_thresh = np.array([(s1>=pix_thresh) & (s2>=pix_thresh) for s1, s2 in sizes])
        good = abv_thresh & is_rgb

        # if bname in has_person.keys():
        # for each image in training list, figure out whether it was flagged as 
        # having a person
        hp = np.array([has_person[bname][im] for im in efiles[bname]['train']['images']])

        print('%s: %.2f of images have person'%(bname, np.nanmean(hp)))
        # the person detection was only run for the "good" images, so make sure there are
        # no nans left here (nans are for ones we skipped)
        assert(not np.any(np.isnan(hp[good])))

        # keep the person-free images now
        ims_use_all = np.where(good & (hp==0))[0]
        # else:
        #     ims_use_all = np.where(good)[0]

        if bname=='trumpet':
            # hacky solution to make sure the ones i hand-picked are here
           np.random.seed(234345+66) 
        else:
           np.random.seed(234345+bi)
            
        n_ex_each = 500
        print('%s: there are %d images in total after QC, choosing %d'%(bname, len(ims_use_all), n_ex_each))
        try:
            ims_use = np.random.choice(ims_use_all, n_ex_each, replace=False)
            # quick check for any duplicate file names
            assert(len(np.unique(np.array(imfiles_all)[ims_use]))==len(np.array(imfiles_all)[ims_use]))

        except:
        
            print('Sampling WITH replacement for %s'%bname)
            ims_use = np.random.choice(ims_use_all, n_ex_each, replace=True)
            
        # if bname=='trumpet':

        #     imfiles = np.array(imfiles_all)[ims_use]
        #     print(trumpet_fns[0])
        #     print(imfiles[0])
        #     print(len(trumpet_fns))
        #     print(len(imfiles))
        #     print(np.sum(np.isin(trumpet_fns, imfiles)))

        efiles_adj[bname]['images'] = np.array(imfiles_all)[ims_use]
        efiles_adj[bname]['size'] = np.array(sizes)[ims_use]
        efiles_adj[bname]['mode'] = np.array(efiles[bname]['train']['mode'])[ims_use]
        # if bname in has_person.keys():
        efiles_adj[bname]['has_person'] = hp[ims_use]


    print('saving to %s'%fn2save)
    np.save(fn2save, efiles_adj)

    

def get_ecoset_files_includeperson():

    # this is a test to see what happens if we do NOT exclude person category
    # for image similarity analyses
    
    # info about which categories to use
    fn = os.path.join(ecoset_info_path, 'categ_use_ecoset.npy')
    info = np.load(fn, allow_pickle=True).item()
    bnames = np.array(list(info['binfo'].keys()))

    # list of all files in each category
    fn = os.path.join(ecoset_info_path, 'ecoset_file_info.npy')
    efiles = np.load(fn, allow_pickle=True).item()

    # # list of which images have person
    fn_person = os.path.join(ecoset_info_path, 'ecoset_files_detect_person.npy')
    has_person = np.load(fn_person, allow_pickle=True).item()


    pix_thresh = 256

    # where to save result 
    fn2save = os.path.join(ecoset_info_path, 'ecoset_files_includeperson.npy')


    efiles_adj = dict()

    for bi, bname in enumerate(bnames):

        efiles_adj[bname] = dict()

        # choose images to analyze here
        imfiles_all = efiles[bname]['train']['images']

        sizes = efiles[bname]['train']['size']
        is_rgb = np.array(efiles[bname]['train']['mode'])=='RGB'

        # want images that are big enough and in RGB
        abv_thresh = np.array([(s1>=pix_thresh) & (s2>=pix_thresh) for s1, s2 in sizes])
        good = abv_thresh & is_rgb

        # if bname in has_person.keys():
        # for each image in training list, figure out whether it was flagged as 
        # having a person
        hp = np.array([has_person[bname][im] for im in efiles[bname]['train']['images']])

        # don't remove person-having images
        ims_use_all = np.where(good)[0]

        np.random.seed(765777+bi)
        n_ex_each = 500
        print('%s: there are %d images in total after QC, choosing %d'%(bname, len(ims_use_all), n_ex_each))
        ims_use = np.random.choice(ims_use_all, n_ex_each, replace=False)

        print('%s: %.2f of selected images have person'%(bname, np.nanmean(hp[ims_use])))
        
        # quick check for any duplicate file names
        assert(len(np.unique(np.array(imfiles_all)[ims_use]))==len(np.array(imfiles_all)[ims_use]))


        efiles_adj[bname]['images'] = np.array(imfiles_all)[ims_use]
        efiles_adj[bname]['size'] = np.array(sizes)[ims_use]
        efiles_adj[bname]['mode'] = np.array(efiles[bname]['train']['mode'])[ims_use]
        # if bname in has_person.keys():
        efiles_adj[bname]['has_person'] = hp[ims_use]


    print('saving to %s'%fn2save)
    np.save(fn2save, efiles_adj)


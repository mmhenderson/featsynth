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


def get_ecoset_file_info(debug=0):

    """
    Gather info about what files are included in ecoset database.
    This will speed things up later on when figuring out what files to use.
    Also going to choose categories to use based on which have most images.
    """

    # this has the mapping from ecoset name to folder names
    # (computed in above function)
    names_path = os.path.join(ecoset_info_path, 'ecoset_names.npy')
    info = np.load(names_path, allow_pickle=True).item()

    bnames = list(info.keys())
    
    # # which superordinate categories are we using?
    # # for speed, only going to process images belonging to one of these.
    # superord_use, basic_names_each = choose_ecoset_categs_step1()
    # bnames = np.concatenate(basic_names_each, axis=0)

    print(bnames)
    
    debug = debug==1
    
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
            try:
                im = PIL.Image.open(imfn)
                efiles[bname]['train']['size'][ii] = im.size
                efiles[bname]['train']['mode'][ii] = im.mode
            except:
                print('%s failed to load'%imfn)
                efiles[bname]['train']['size'][ii] = (0,0)
                efiles[bname]['train']['mode'][ii] = ''
                
            
        folder = os.path.join(ecoset_path, 'val', info[bname])
        efiles[bname]['val'] = dict()
        imlist = os.listdir(folder)
        efiles[bname]['val']['images'] = imlist
        efiles[bname]['val']['size'] = [[] for ii in range(len(efiles[bname]['val']['images']))]
        efiles[bname]['val']['mode'] = ['' for ii in range(len(efiles[bname]['val']['images']))]

        for ii, imfile in enumerate(imlist):
            
            imfn = os.path.join(os.path.join(folder, imfile))
            try:
                im = PIL.Image.open(imfn)
                efiles[bname]['val']['size'][ii] = im.size
                efiles[bname]['val']['mode'][ii] = im.mode
            except:
                print('%s failed to load'%imfn)
                efiles[bname]['val']['size'][ii] = (0,0)
                efiles[bname]['val']['mode'][ii] = ''
            
            
        folder = os.path.join(ecoset_path, 'test', info[bname])
        efiles[bname]['test'] = dict()
        imlist = os.listdir(folder)
        efiles[bname]['test']['images'] = imlist
        efiles[bname]['test']['size'] = [[] for ii in range(len(efiles[bname]['test']['images']))]
        efiles[bname]['test']['mode'] = ['' for ii in range(len(efiles[bname]['test']['images']))]

        for ii, imfile in enumerate(imlist):
            
            imfn = os.path.join(os.path.join(folder, imfile))
            try:
                im = PIL.Image.open(imfn)
                efiles[bname]['test']['size'][ii] = im.size
                efiles[bname]['test']['mode'][ii] = im.mode
            except:
                print('%s failed to load'%imfn)
                efiles[bname]['test']['size'][ii] = (0,0)
                efiles[bname]['test']['mode'][ii] = ''
            
    
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


def get_ecoset_files_good():

    """
    Make a list of which files to use in each basic-level category, 
    excluding those that have "person" detected.
    Have to run "detect_person_ecoset" to get this
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

        # hacky solutions to make sure the ones i hand-picked are here
        if bname=='trumpet':
           np.random.seed(234345+66) 
        elif (info['binfo'][bname]['super_name']=='fruit') or \
             (info['binfo'][bname]['super_name']=='vegetable'):
            bi_use = bi-8
            print([bname, bi, bi_use])
            np.random.seed(234345+bi_use) 
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
      
        efiles_adj[bname]['images'] = np.array(imfiles_all)[ims_use]
        efiles_adj[bname]['size'] = np.array(sizes)[ims_use]
        efiles_adj[bname]['mode'] = np.array(efiles[bname]['train']['mode'])[ims_use]
        # if bname in has_person.keys():
        efiles_adj[bname]['has_person'] = hp[ims_use]

        

    print('saving to %s'%fn2save)
    np.save(fn2save, efiles_adj)

    
def choose_ecoset_categs_step1():
        
    """
    For the 8 superordinate categories we're using in experiment:
    ['insect', 'mammal', 'vegetable', 'fruit', 
                 'tool','musical instrument','furniture','vehicle']
                 
    This code will list all the basic-level categories available in each.
    This is only including those that are in both ecoset and things,
    and excluding some weird/bad ones.
    From these possible basic options, we're going to further pare down to
    the 8 best in each superordinate, to get the final 64 categories.
    """
    
    # this is the list of 8 superord categories to use here
    superord_use = ['insect', 'mammal', 'vegetable', 'fruit', \
                 'tool','musical instrument','furniture','vehicle']
    
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
    
    # adding new category here - "mammal"
    # this is not actually labeled in 'things' originally, but adding it
    # so we have another good animal category...
    categ_names = np.array(list(categ_names)+['mammal'])
    cmat_use = np.concatenate([cmat_use, np.zeros([cmat_use.shape[0],1], dtype=bool)], axis=1)
    mammal_extras = ['aardvark', 'alpaca', 'anteater', 'antelope', 'badger', 'bat', \
                     'bear', 'beaver', 'bison', 'boar', 'bull', 'calf', 'camel', 'cat', \
                     'cheetah', 'chihuahua', 'chinchilla', 'chipmunk', 'cougar', 'cow', \
                     'coyote', 'dalmatian', 'deer', 'dog', 'dolphin', 'donkey', \
                     'elephant', 'ferret', 'fox', 'gazelle', 'giraffe', 'goat', 'gopher', \
                     'gorilla', 'groundhog', 'guinea pig', 'hamster', 'hedgehog', \
                     'hippopotamus', 'horse', 'hyena', 'kangaroo', 'kitten', 'koala', \
                     'lamb', 'leopard', 'lion', 'llama', 'manatee', 'meerkat', 'mole', \
                     'mongoose', 'monkey', 'moose', 'mouse', 'orangutan', 'otter', 'panda',\
                     'panther', 'pig', 'piglet', 'platypus', 'polar bear', 'pony', 'poodle', \
                     'porcupine', 'possum', 'pug', 'puppy', 'rabbit', 'raccoon', 'ram', 'rat', \
                     'reindeer', 'rhinoceros', 'seal', 'sheep', 'skunk', 'sloth', 'squirrel', \
                     'tiger', 'walrus', 'warthog', 'weasel', 'whale', 'wolf', 'yak', 'zebra']
    sind = np.where(categ_names=='mammal')[0][0]
    for name in mammal_extras:
        bind = np.where(things_names==name)[0]
        cmat_use[bind,sind] = True
    
    cmat_use.shape
    
    # make a list of names to skip...
    
    # skip any that are the same name as superordinate, because that won't make sense
    things_inds_skip = np.isin(things_names, categ_names) 
    
    # removing any duplicate concept names here (these are ambiguous meaning words like bat)
    un, counts = np.unique(things_names, return_counts=True)
    duplicate_conc = un[counts>1]
    duplicate_conc_inds = [conc in duplicate_conc for conc in things_names]
    things_inds_skip = things_inds_skip | duplicate_conc_inds
    
    # skip some other ones that I found confusing/uncommon 
    # (cheese was included as dessert, avocado was fruit)
    conc_skip = ['earwig', 'cheese', 'avocado', 'anvil', 'okra',\
                 'dolphin','bat','otter','platypus','walrus','whale']
    things_inds_skip = things_inds_skip | np.isin(things_names, conc_skip)
    
    things_names_skip = things_names[things_inds_skip]
    
    
    # load names of all categories in ecoset dataset
    ecoset_folders = np.load(os.path.join(ecoset_info_path, 'ecoset_names.npy'), \
                             allow_pickle=True).item()
    ecoset_names = np.array(list(ecoset_folders.keys()))
    
    # get overlap
    is_in_ecoset = np.array([np.any([(t==e) for e in ecoset_names]) for t in things_names])
    
    superord_inds_use = np.array([np.where(categ_names==superord)[0][0] \
                                  for superord in superord_use])
    
    # going to exclude any basic-level that occur in multiple superord categ
    used_once = np.sum(cmat_use[:,superord_inds_use], axis=1)==1
    
    inds_each = [(cmat_use[:,ci] & is_in_ecoset & ~things_inds_skip & used_once) \
                            for ci in superord_inds_use]
    
    counts_each = np.array([np.sum(inds) for inds in inds_each])
    
    # this is the resulting list of basic categ in each superord categ.
    # there are still too many here so we will further pare down based on file counts
    basic_names_each = [things_names[inds] for inds in inds_each]

    return superord_use, basic_names_each


def choose_ecoset_categs_step2():

    """
    Defining the final set of 64 categories, 8 basic-level in each of 8 superordinate.
    These are chosen by using the above "step1" then choosing 8 best from each.
    """

    # these are manually selected
    # Based on total number of ecoset images available
    # and also trying to minimize ambiguity between categs.
    categ_use = dict({'insect':
                    ['beetle', 'bee', 'butterfly', 'grasshopper', 'caterpillar', 'ant', 'moth', 'mosquito'],
                      'mammal': 
                    ['dog','lion','horse','squirrel','elephant','cow','pig','rabbit'],
                     'vegetable':
                    ['pea', 'corn', 'pumpkin', 'onion', 'cabbage', 'lettuce', 'beet', 'asparagus'],
                     'fruit': 
                    ['grape', 'cherry', 'raspberry', 'apple', 'pear', 'banana', 'pomegranate', 'coconut'], 
                     'tool': 
                    ['pencil', 'knife', 'axe', 'broom', 'hammer', 'shovel', 'spoon', 'scissors'], 
                     'musical instrument':
                    ['bell', 'guitar', 'piano', 'drum', 'violin', 'trumpet', 'clarinet', 'cymbal'], 
                     'furniture':
                    ['table', 'bench', 'couch', 'television', 'bed', 'chair', 'refrigerator', 'lamp'], 
                     'vehicle':
                    ['ship', 'train', 'airplane', 'truck', 'car', 'bus', 'motorcycle', 'canoe']})
    
    # load names of all categories in ecoset dataset
    ecoset_folders = np.load(os.path.join(ecoset_info_path, 'ecoset_names.npy'), \
                             allow_pickle=True).item()

    superord_use, basic_names_each = choose_ecoset_categs_step1()

    # organize into dict for saving
    sinfo = dict()
    binfo = dict()
    for si, sname in enumerate(superord_use):
        sinfo[sname] = dict()
        sinfo[sname]['super_name'] = sname
        sinfo[sname]['basic_names'] = categ_use[sname]
        for bi, bname in enumerate(categ_use[sname]):
            binfo[bname] = dict()
            binfo[bname]['basic_name'] = bname
            binfo[bname]['super_name'] = sname
            binfo[bname]['ecoset_folder'] = ecoset_folders[bname]

    # save
    fn2save = os.path.join(ecoset_info_path, 'categ_use_ecoset.npy')
    print('saving to %s'%fn2save)
    np.save(fn2save, {'binfo': binfo, 'sinfo': sinfo})




# def get_ecoset_files_includeperson():

#     # this is a test to see what happens if we do NOT exclude person category
#     # for image similarity analyses
    
#     # info about which categories to use
#     fn = os.path.join(ecoset_info_path, 'categ_use_ecoset.npy')
#     info = np.load(fn, allow_pickle=True).item()
#     bnames = np.array(list(info['binfo'].keys()))

#     # list of all files in each category
#     fn = os.path.join(ecoset_info_path, 'ecoset_file_info.npy')
#     efiles = np.load(fn, allow_pickle=True).item()

#     # # list of which images have person
#     fn_person = os.path.join(ecoset_info_path, 'ecoset_files_detect_person.npy')
#     has_person = np.load(fn_person, allow_pickle=True).item()


#     pix_thresh = 256

#     # where to save result 
#     fn2save = os.path.join(ecoset_info_path, 'ecoset_files_includeperson.npy')


#     efiles_adj = dict()

#     for bi, bname in enumerate(bnames):

#         efiles_adj[bname] = dict()

#         # choose images to analyze here
#         imfiles_all = efiles[bname]['train']['images']

#         sizes = efiles[bname]['train']['size']
#         is_rgb = np.array(efiles[bname]['train']['mode'])=='RGB'

#         # want images that are big enough and in RGB
#         abv_thresh = np.array([(s1>=pix_thresh) & (s2>=pix_thresh) for s1, s2 in sizes])
#         good = abv_thresh & is_rgb

#         # if bname in has_person.keys():
#         # for each image in training list, figure out whether it was flagged as 
#         # having a person
#         hp = np.array([has_person[bname][im] for im in efiles[bname]['train']['images']])

#         # don't remove person-having images
#         ims_use_all = np.where(good)[0]

#         np.random.seed(765777+bi)
#         n_ex_each = 500
#         print('%s: there are %d images in total after QC, choosing %d'%(bname, len(ims_use_all), n_ex_each))
#         ims_use = np.random.choice(ims_use_all, n_ex_each, replace=False)

#         print('%s: %.2f of selected images have person'%(bname, np.nanmean(hp[ims_use])))
        
#         # quick check for any duplicate file names
#         assert(len(np.unique(np.array(imfiles_all)[ims_use]))==len(np.array(imfiles_all)[ims_use]))


#         efiles_adj[bname]['images'] = np.array(imfiles_all)[ims_use]
#         efiles_adj[bname]['size'] = np.array(sizes)[ims_use]
#         efiles_adj[bname]['mode'] = np.array(efiles[bname]['train']['mode'])[ims_use]
#         # if bname in has_person.keys():
#         efiles_adj[bname]['has_person'] = hp[ims_use]


#     print('saving to %s'%fn2save)
#     np.save(fn2save, efiles_adj)



# def get_ecoset_file_info_fornoise(debug=0):

#     """
#     Gather info about what files are included in ecoset database.
#     This will speed things up later on when figuring out what files to use.
#     Also going to choose categories to use based on which have most images.
    
#     % this version (2/12/24) is intended for choosing images that we will estimate noise
#     % distributions from.
#     % only using categories NOT in the comb64 set
    
#     """

#     debug = debug==1
    
#     # this has the mapping from ecoset name to folder names
#     fn = os.path.join(ecoset_info_path, 'ecoset_names.npy')
#     info = np.load(fn, allow_pickle=True).item()

#     # this is the file for the categories used in actual experiments
#     finfo = np.load(os.path.join(ecoset_info_path, \
#                                  'ecoset_file_info.npy'), allow_pickle=True).item()

#     non_overlap = [e for e in list(info.keys()) if e not in list(finfo.keys())]
#     bnames = non_overlap

#     print(bnames)
    
    
#     fn2save = os.path.join(ecoset_info_path, 'ecoset_file_info_fornoise.npy')
#     print(fn2save)
    
#     if os.path.exists(fn2save):
#         efiles = np.load(fn2save, allow_pickle=True).item()
#     else:
#         efiles = dict()
    
#     for bi, bname in enumerate(bnames):
    
#         if debug & (bi>0):
#             continue
    
#         print(bname)
        
#         if bname in efiles.keys():
#             print('done with %s, skipping'%bname)
#             continue
            
        
#         sys.stdout.flush()
        
#         efiles[bname] = dict()

#         folder = os.path.join(ecoset_path, 'train', info[bname])
#         efiles[bname]['train'] = dict()
#         imlist = os.listdir(folder)
#         efiles[bname]['train']['images'] = imlist
#         efiles[bname]['train']['size'] = [[] for ii in range(len(efiles[bname]['train']['images']))]
#         efiles[bname]['train']['mode'] = ['' for ii in range(len(efiles[bname]['train']['images']))]

#         for ii, imfile in enumerate(imlist):
            
#             imfn = os.path.join(os.path.join(folder, imfile))
#             try:
#                 im = PIL.Image.open(imfn)
#                 efiles[bname]['train']['size'][ii] = im.size
#                 efiles[bname]['train']['mode'][ii] = im.mode
#             except:
#                 print('%s failed to load'%imfn)
#                 efiles[bname]['train']['size'][ii] = (0,0)
#                 efiles[bname]['train']['mode'][ii] = ''
            
#         # now computing some basic stats/counts about the files
#         sizes = efiles[bname]['train']['size'] 

#         is_rgb = np.array(efiles[bname]['train']['mode'])=='RGB'

#         # want images that are big enough and in RGB
#         pix_thresh = 256
#         abv_thresh = np.array([(s1>=pix_thresh) & (s2>=pix_thresh) for s1, s2 in sizes])
#         ims_use_all = np.where(abv_thresh & is_rgb)[0]

#         efiles[bname]['total_ims'] = len(sizes)
#         efiles[bname]['good_ims'] = len(ims_use_all)
        
        
#     print('saving to %s'%fn2save)
#     np.save(fn2save, efiles)



# def get_ecoset_files_fornoise():

#     """
#     Make a list of which files to use in each basic-level category, 
#     excluding those that have "person" detected.
#     This is about 500 per category (big), and not quality checked. 
#     To get the smaller list of ~40 images that we use in experiment, 
#     use choose_extra_ecoset_ims.py

#     % this version (2/12/24) is intended for choosing images that we will estimate noise
#     % distributions from.
#     % only using categories NOT in the comb64 set
    
#     """
  
#     # list of all files in each category
#     fn = os.path.join(ecoset_info_path, 'ecoset_file_info_fornoise.npy')
#     efiles = np.load(fn, allow_pickle=True).item()

#     bnames = efiles.keys()
    
#     pix_thresh = 256

#     # where to save result 
#     fn2save = os.path.join(ecoset_info_path, 'ecoset_files_use_fornoise.npy')

#     efiles_adj = dict()

#     for bi, bname in enumerate(bnames):

#         efiles_adj[bname] = dict()

#         # choose images to analyze here
#         imfiles_all = efiles[bname]['train']['images']

#         sizes = efiles[bname]['train']['size']
#         is_rgb = np.array(efiles[bname]['train']['mode'])=='RGB'

#         # want images that are big enough and in RGB
#         abv_thresh = np.array([(s1>=pix_thresh) & (s2>=pix_thresh) for s1, s2 in sizes])
#         good = abv_thresh & is_rgb

#         # keep the person-free images now
#         ims_use_all = np.where(good)[0]
        
#         np.random.seed(132434+bi)
            
#         n_ex_each = 500
#         print('%s: there are %d images in total after QC, choosing %d'%(bname, len(ims_use_all), n_ex_each))
#         try:
#             ims_use = np.random.choice(ims_use_all, n_ex_each, replace=False)
#             # quick check for any duplicate file names
#             assert(len(np.unique(np.array(imfiles_all)[ims_use]))==len(np.array(imfiles_all)[ims_use]))

#         except:
        
#             print('Sampling WITH replacement for %s'%bname)
#             ims_use = np.random.choice(ims_use_all, n_ex_each, replace=True)
            
#         efiles_adj[bname]['images'] = np.array(imfiles_all)[ims_use]
#         efiles_adj[bname]['size'] = np.array(sizes)[ims_use]
#         efiles_adj[bname]['mode'] = np.array(efiles[bname]['train']['mode'])[ims_use]
       

#     print('saving to %s'%fn2save)
#     np.save(fn2save, efiles_adj)


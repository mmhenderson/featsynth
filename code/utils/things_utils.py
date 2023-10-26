import numpy as np
import PIL
import pandas as pd
import os
import copy
import scipy.io

things_stim_path = '/user_data/mmhender/stimuli/things/'
things_images_root = os.path.join(things_stim_path, 'Images')


def process_concepts():

    """
    Processing THINGS categories and concepts to use in experiment.
    
    This pertains to the set of 200 categories that we used in first 
    behavioral experiments. 
    Tagged with "things200" in filenames, or "images_v1".
    
    To choose the 64 categories we used in MRI expts, see proc_ecoset_categs.py
    """
    
    filename = os.path.join(things_stim_path,'things_concepts.tsv')

    df = pd.read_csv(filename, sep='\t')
    concept_list = np.array(df['Word'])
    ids_list = np.array(df['uniqueID'])
    concept_list = [c.replace(' ', '_') for c in concept_list]
    n_concepts = len(concept_list)
    # concepts are the fine-grained/basic level names

    info_folder = os.path.join(things_stim_path,'27 higher-level categories')
    categ_names = scipy.io.loadmat(os.path.join(info_folder, 'categories.mat'))['categories'][0]
    categ_names = [categ_names[ii][0] for ii in range(len(categ_names))]
    categ_names = [categ.replace(' ', '_') for categ in categ_names]
    n_categ = len(categ_names)
    # categories are the high-level/superordinate names

    # load the "bottom-up" (human-generated) groupings
    dat = scipy.io.loadmat(os.path.join(info_folder, 'category_mat_bottom_up.mat'))
    cmat = dat['category_mat_bottom_up']

    # there is a swap in this labeling betweeen "hot-air balloon" and "hot chocolate"
    # (maybe a typo?)
    # i am manually switching them here
    cmat_fixed = copy.deepcopy(cmat)
    tmp = copy.deepcopy(cmat[801,:])
    cmat_fixed[801,:] = cmat[803,:]
    cmat_fixed[803,:] = tmp

    # now going to fix these a bit to get rid of anything ambiguous
    cmat_adjusted = copy.deepcopy(cmat_fixed).astype(bool) 

    # removing any duplicate concept names here (these are ambiguous meaning words like bat)
    un, counts = np.unique(concept_list, return_counts=True)
    duplicate_conc = un[counts>1]
    duplicate_conc_inds = np.where([conc in duplicate_conc for conc in concept_list])
    cmat_adjusted[duplicate_conc_inds,:] = False

    # remove any concepts that have the same name as one of the categories (for example "fruit")
    duplicate_inds = np.where([conc in categ_names for conc in concept_list])[0]
    cmat_adjusted[duplicate_inds,:] = False

    # remove first the "food" and "animal" categories, because these share a lot of members with 
    # other categories like "dessert", "bird"
    categ_exclude_first = ['food','animal']
    categ_names_new = [cc for cc in categ_names if cc not in categ_exclude_first]
    categ_inds_keep = [cc for cc in range(n_categ) if categ_names[cc] not in categ_exclude_first]
    n_categ = len(categ_inds_keep)
    categ_names = categ_names_new
    cmat_adjusted = cmat_adjusted[:,categ_inds_keep]

    # now decide which concepts and categories to exclude from this remaining set.
    # exclusion criteria:
    # exclude any categories that are supersets of other categories
    # (for example exclude food, keep dessert)
    # exclude concepts that are supersets of other concepts
    # (for example exclude berry, keep strawberry).
    # also excluding some concepts that are not very well-known 
    # (for example spark plug).

    categ_exclude = list(pd.read_csv(os.path.join(things_stim_path, 'categ_exclude.csv'))['categ_exclude'])
    concepts_exclude = list(pd.read_csv(os.path.join(things_stim_path, 'conc_exclude.csv'))['concepts_exclude'])

    # from set of all categories, find the ones that are overlapping for any categories
    conc_inds_overlapping = np.sum(cmat_adjusted, axis=1)>1
    conc_inds_exclude = np.array([conc in concepts_exclude for conc in concept_list])

    categ_names_new = [cc for cc in categ_names if cc not in categ_exclude]
    categ_inds_keep = [cc for cc in range(n_categ) if categ_names[cc] not in categ_exclude]
    n_categ = len(categ_inds_keep)
    categ_names = categ_names_new
    cmat_adjusted = cmat_adjusted[:,categ_inds_keep]

    conc_inds_notcategorized = np.sum(cmat_adjusted, axis=1)==0
    conc_inds_keep = ~conc_inds_overlapping & ~conc_inds_notcategorized & ~conc_inds_exclude
    cmat_adjusted = cmat_adjusted[conc_inds_keep,:]

    conc_use = np.array(concept_list)[conc_inds_keep]
    ids_use = np.array(ids_list)[conc_inds_keep]
    n_concepts = len(conc_use)
    
    image_names = dict()

    for categ_ind in range(n_categ):
        ids = ids_use[cmat_adjusted[:,categ_ind]]
        for conc in ids:
            files = os.listdir(os.path.join(things_images_root, conc))
            files.sort()
            image_names[conc] = files
        
        
    concepts_each_categ = [conc_use[cmat_adjusted[:,ca]] for ca in range(n_categ)]
    ids_each_categ = [ids_use[cmat_adjusted[:,ca]] for ca in range(n_categ)]
    
    # save these concept names to file
    filename_save = os.path.join(things_stim_path, 'concepts_removeoverlap.npy')
    print('saving to %s'%filename_save)
    np.save(filename_save, \
            {'categ_names': categ_names, 
             'concept_names': concepts_each_categ, 
             'concept_ids': ids_each_categ, 
             'image_names': image_names}, \
            allow_pickle=True)
    
def subsample_concepts():

    """
    This is the next step of sub-sampling to choose the 200 THINGS categories
    that are used in behavior experiments (runs after above function)
    They get sub-sampled more when creating the actual experiment design.
    """
    
    concept_info = np.load(os.path.join(things_stim_path, 'concepts_removeoverlap.npy'), \
                           allow_pickle=True).item()
    categ_names = concept_info['categ_names']
    concept_names = concept_info['concept_names']
    image_names = concept_info['image_names']
    concept_ids = concept_info['concept_ids']

    # now sub-sample the same number of concepts from each category
    n_each = [len(conc) for conc in concept_names]
    n_concepts_each = np.min(n_each)

    concept_names_subsample = []
    concept_ids_subsample = []
    
    # always using same seed so we get same result
    rndseed = 243545
    np.random.seed(rndseed)

    for cat in range(len(categ_names)):
        conc = concept_names[cat]
        conc_inds_use = np.random.choice(np.arange(len(conc)), n_concepts_each, replace=False)
        
        conc_use = conc[conc_inds_use]
        ids_use = concept_ids[cat][conc_inds_use]
        
        concept_names_subsample.append(conc_use)
        concept_ids_subsample.append(ids_use)
        
   
    filename_save = os.path.join(things_stim_path, 'concepts_use.npy')
    print('saving to %s'%filename_save)
    np.save(filename_save, \
            {'categ_names': categ_names, 
             'concept_names_subsample': concept_names_subsample,
             'concept_ids_subsample': concept_ids_subsample,
             'image_names': image_names, 
             'rndseed': rndseed}, \
            allow_pickle=True)
    
    

def get_filename(categ_name, concept_name, image_ind=0):
    
    concept_info = np.load(os.path.join(things_stim_path, 'concepts_removeoverlap.npy'), \
                           allow_pickle=True).item()
    
    categ_ind = np.where([categ_name==cat for cat in concept_info['categ_names']])[0][0]
    concept_ind = np.where(concept_name==concept_info['concept_names'][categ_ind])[0][0]
    
    subfolder = concept_info['concept_ids'][categ_ind][concept_ind]
    image_name = concept_info['image_names'][subfolder][image_ind]
    
    filename = os.path.join(things_images_root, subfolder, image_name)
    
    return filename


def get_things_files():

    """
    Making a list of all the things concepts and files included in each.
    """
    
    filename = os.path.join(things_stim_path,'things_concepts.tsv')

    df = pd.read_csv(filename, sep='\t')
    concept_list = np.array(df['Word'])
    ids_list = np.array(df['uniqueID'])
    concept_list = [c.replace(' ', '_') for c in concept_list]
    n_concepts = len(concept_list)

    tfiles = dict()

    for concept in concept_list:

        try:
            files = os.listdir(os.path.join(things_stim_path, 'Images', concept))
            files = [f for f in files if '.jpg' in f]
            files = np.sort(files)
            tfiles[concept] = files
        except:
            continue
        
    fn2save = os.path.join(things_stim_path, 'things_file_info.npy')
    print('saving to %s'%fn2save)
    np.save(fn2save, tfiles)
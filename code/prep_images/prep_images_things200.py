import os, sys
import numpy as np
import pandas as pd
import PIL
from PIL import Image

project_root = '/user_data/mmhender/featsynth/'

def prep(image_set_name = 'images_things200'):
    
    # these are the things images used in experiment 1 
    # colored images
    # 200 total basic-level categories
    
    expt_name = 'expt1'
    
    # images are in same folder used for experiment 1
    stims_root = '/user_data/mmhender/stimuli/featsynth/images_v1'
   
    print(expt_name, image_set_name, stims_root)
    
    image_list = make_image_list(stims_root, expt_name)

    # get intact images only
    image_list = image_list[image_list['image_type']=='orig']
    print(image_list.shape)
        
    folder_save = os.path.join(project_root, 'features','raw')
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)

    image_list_filename = os.path.join(folder_save, '%s_list.csv'%(image_set_name))
    print('saving to %s'%image_list_filename)
    sys.stdout.flush()
    image_list.to_csv(image_list_filename)
    
    image_data = load_image_data(image_list, n_pix=256)
    image_data_filename = os.path.join(folder_save, '%s_preproc.npy'%(image_set_name))
    print('saving to %s'%(image_data_filename))
    sys.stdout.flush()
    np.save(image_data_filename, image_data)
    
    
def make_image_list(stims_root, expt_name):

    super_names_all = []
    basic_names_all = []
    
    for cb in [1,2]:
        design_file = os.path.join(project_root, 'expt_design', expt_name, \
                                   'trial_info_counterbal%d_randorder0.csv'%cb)
        design = pd.read_csv(design_file)
        super_names = np.unique(np.array(design['super_name']))
        super_names_all += [super_names]
        basic_names = []
        for sname in super_names:
            bnames = np.unique(np.array(design['basic_name'][design['super_name']==sname]))
            basic_names += [bnames]
        basic_names_all += [basic_names]

    basic_names_list = list(np.reshape(np.array(basic_names_all), \
                              [np.prod(np.array(basic_names_all).shape),]))
    super_names_list = list(np.repeat(np.reshape(np.array(super_names_all), \
           [np.prod(np.array(super_names_all).shape),]), 10))

    # make a list of all the images we want to analyze here 
    image_list = pd.DataFrame(columns=['super_name', 'super_index', \
                                       'basic_name', 'basic_index', \
                                       'image_type', \
                                       'exemplar_number','image_filename'])
    n_ims_each = 10;
    ex_nums_use = np.arange(1,n_ims_each+1)
    image_type_strs = ['pool1','pool2','pool3','pool4','orig']

    folders_all = os.listdir(stims_root)

    ii=-1

    for bi, bname in enumerate(basic_names_list):

        sname = super_names_list[bi]
        si = int(np.floor(bi/10))

        folders_this_cat = [f for f in folders_all if f[0:(len(bname)+1)]=='%s_'%bname]
        assert(len(folders_this_cat)==n_ims_each)
        folders_use = [[f for f in folders_this_cat if '%02d'%ii in f][0] for ii in ex_nums_use]

        for fi, folder in enumerate(folders_use):

            ex_num = int(folder.split('_')[-1][0:2])
            assert(ex_num==ex_nums_use[fi])

            for ti, imtype in enumerate(image_type_strs):
                if imtype=='orig':
                    image_name = os.path.join(stims_root, folder, 'orig.png')
                else:
                    image_name = os.path.join(stims_root, folder, 'grid5_1x1_upto_%s.png'%imtype)
                assert(os.path.exists(image_name))
                ii+=1
                image_list = pd.concat([image_list, \
                                             pd.DataFrame({'super_name': sname,\
                                                             'super_index': si, \
                                                             'basic_name': bname,\
                                                             'basic_index': bi, \
                                                             'image_type': imtype, \
                                                            'exemplar_number': ex_num, \
                                                            'image_filename': image_name}, index=[ii]) \
                                             ])

    return image_list

def load_image_data(image_list, n_pix=256):
    
    n_images = len(image_list)

    image_data = np.zeros([n_images, 3, n_pix, n_pix],dtype=int)
    
    for ii in range(n_images):
        filename = np.array(image_list['image_filename'])[ii]
        if np.mod(ii, 100)==0:
            print('loading from %s'%filename)
            sys.stdout.flush()
        im = PIL.Image.open(filename)

        assert(im.size[0]==im.size[1])
        # im_resized = im.resize([n_pix, n_pix], resample=PIL.Image.BILINEAR)
        im_resized = im.resize([n_pix, n_pix], resample=PIL.Image.Resampling.LANCZOS)
        
        if im_resized.mode!='RGB':
            im_resized = im_resized.convert('RGB')
        image_array = np.reshape(np.array(im_resized.getdata()), [n_pix, n_pix, 3])
        image_data[ii,:,:,:] = np.moveaxis(image_array, [2],[0])

    return image_data
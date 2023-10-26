import os, sys
import numpy as np
import pandas as pd
import PIL
from PIL import Image

project_root = '/user_data/mmhender/featsynth/'
ecoset_info_path = '/user_data/mmhender/stimuli/ecoset_info/'
things_stim_path = '/user_data/mmhender/stimuli/things/'

# this is for the THINGS images but using same 64 categs as used for ecoset in v2

def prep(image_set_name = 'images_things64', bilinear_resize=0):

    bilinear_resize = bilinear_resize==1
    
    # info about which categories to use
    fn = os.path.join(ecoset_info_path, 'categ_use_ecoset.npy')
    info = np.load(fn, allow_pickle=True).item()
    bnames = np.array(list(info['binfo'].keys()))

    # if image_set_name == 'images_things64_music2':
    #     # adding some extra music categories here
    #     new_instr = ['kazoo','cymbal','trumpet']
    #     bnames = new_instr
    #     for b in new_instr:
    #         info['binfo'][b] = dict()
    #         info['binfo'][b]['super_name'] = 'musical instrument'

    print(bnames)
    # list of all files in each category
    fn2load = os.path.join(things_stim_path, 'things_file_info.npy')
    tfiles = np.load(fn2load, allow_pickle=True).item()

    folder_save = os.path.join(project_root, 'features','raw')
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
           
    # make a list of all the images we want to analyze here 
    image_list = pd.DataFrame(columns=['super_name','super_index', \
                                       'basic_name', 'basic_index', \
                                       'image_type', \
                                       'exemplar_number','image_filename'])
    
    n_ex_use = 12
    
    last_sname = ''
    si = -1;
    
    ii = -1
    
    for bi, bname in enumerate(bnames):
        
        folder = os.path.join(things_stim_path,'Images', bname)
        
        # choose images to analyze here
        ex_use = np.arange(0, n_ex_use)

        # for a few categories, we're going to manually skip images that had people or faces
        if bname=='drum':
            ex_use = [0,1,2,3,4,5,6,7,8,9, 11,12]
        elif bname=='violin':
            ex_use = [0,1,2,3,4,5,6,7, 9,10,11,12]
        elif bname=='clarinet':
            # 13 here is a cropped version of image that i created manually
            ex_use = [0,1,2,3,4,13,6,7,8,9,10,11]
        elif bname=='cymbal':
            ex_use = [0,1,2, 4,5,6,7, 9,10,11, 13,14]

        ims_use = tfiles[bname][ex_use]
        print(ims_use)
        
        for ee, im in enumerate(ims_use):

            target_image_filename = os.path.join(folder,im)
            
            assert(os.path.exists(target_image_filename))
           
            sname = info['binfo'][bname]['super_name']
            if sname!=last_sname:
                si+=1
                last_sname = sname
                
            ii+=1
            image_list = pd.concat([image_list, \
                                         pd.DataFrame({'super_name': sname, 'super_index': si, \
                                                         'basic_name': bname, 'basic_index': bi, \
                                                         'image_type': 'orig', \
                                                        'exemplar_number': ee, \
                                                        'image_filename': target_image_filename}, index=[ii]) \
                                        ])


    image_list_filename = os.path.join(folder_save, '%s_list.csv'%(image_set_name))
    print('saving to %s'%image_list_filename)
    sys.stdout.flush()
    image_list.to_csv(image_list_filename)
    
    image_data = load_image_data(image_list, n_pix=256, bilinear_resize=bilinear_resize)
    if bilinear_resize:
        # only use this for getting clip embeddings (image selection)
        image_data_filename = os.path.join(folder_save, '%s_bilinear_preproc.npy'%(image_set_name))
    else:
        image_data_filename = os.path.join(folder_save, '%s_preproc.npy'%(image_set_name))
    
    print('saving to %s'%(image_data_filename))
    sys.stdout.flush()
    np.save(image_data_filename, image_data)
    
    

def load_image_data(image_list, n_pix=256, bilinear_resize=False):
    
    n_images = len(image_list)

    image_data = np.zeros([n_images, 3, n_pix, n_pix],dtype=int)
    
    for ii in range(n_images):
        filename = np.array(image_list['image_filename'])[ii]
        if np.mod(ii, 100)==0:
            print('loading from %s'%filename)
            sys.stdout.flush()
        im = PIL.Image.open(filename)

        assert(im.size[0]==im.size[1])
        if bilinear_resize:
            # this is really not the best way, but i used this version when
            # getting the clip embeddings, so keeping it as option so things are 
            # consistent.
            if ii==0:
                print('using BILINEAR resampling')
            im_resized = im.resize([n_pix, n_pix], resample=PIL.Image.BILINEAR)
        else:
            # this is better for antialiasing
            if ii==0:
                print('using LANCZOS resampling')
            im_resized = im.resize([n_pix, n_pix], resample=PIL.Image.Resampling.LANCZOS)
        
        if im_resized.mode!='RGB':
            im_resized = im_resized.convert('RGB')
        image_array = np.reshape(np.array(im_resized.getdata()), [n_pix, n_pix, 3])
        image_data[ii,:,:,:] = np.moveaxis(image_array, [2],[0])

    return image_data
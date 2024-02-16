import os, sys
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import copy

project_root = '/user_data/mmhender/featsynth/'
ecoset_path = '/lab_data/tarrlab/common/datasets/Ecoset/'
ecoset_info_path = '/user_data/mmhender/stimuli/ecoset_info/'

sys.path.append('/user_data/mmhender/featsynth/texture_synthesis/code/')
import utilities, segmentation_utils

# this is for the ecoset images, 64 categs

def prep(image_set_name = 'images_ecoset64'):
    
    # if image_set_name == 'images_ecoset64':
    # list of all files in each category
    # in this list, images with person have been removed
    fn = os.path.join(ecoset_info_path, 'ecoset_files_use.npy')

    # elif image_set_name == 'images_ecoset64_music2':
    #     # this is the one with different music categories included
    #     fn = os.path.join(ecoset_info_path, 'ecoset_files_use.npy')
        
    # elif image_set_name == 'images_ecoset64_includeperson':
    #     # list of all files in each category
    #     # version of the files list where person-having images have not been removed
    #     fn = os.path.join(ecoset_info_path, 'ecoset_files_includeperson.npy')
   
    print(fn)
    efiles = np.load(fn, allow_pickle=True).item()

    fn = os.path.join(ecoset_info_path, 'ecoset_names.npy')
    print(fn)
    efolders = np.load(fn, allow_pickle=True).item()
    
    # info about which categories to use
    fn = os.path.join(ecoset_info_path, 'categ_use_ecoset.npy')
    print(fn)
    info = np.load(fn, allow_pickle=True).item()
    bnames = np.array(list(info['binfo'].keys()))

    print(bnames)

    # if image_set_name == 'images_ecoset64_music2':
    #     # adding some extra music categories here
    #     new_instr = ['kazoo','cymbals','bugle']
    #     bnames = new_instr
    #     for b in new_instr:
    #         info['binfo'][b] = dict()
    #         info['binfo'][b]['super_name'] = 'musical instrument'
    #         info['binfo'][b]['ecoset_folder'] = efolders[b]

    print(bnames)
    sys.stdout.flush()
    
    folder_save = os.path.join(project_root, 'features','raw')
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
           
    # make a list of all the images we want to analyze here 
    image_list = pd.DataFrame(columns=['super_name','super_index', \
                                       'basic_name', 'basic_index', \
                                       'image_type', \
                                       'exemplar_number','image_filename'])
    
    last_sname = ''
    si = -1;
    
    ii = -1
    
    for bi, bname in enumerate(bnames):
        
        folder = os.path.join(ecoset_path,'train',info['binfo'][bname]['ecoset_folder'])
        
        # choose images to analyze here
        ims_use = efiles[bname]['images']
        
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
    
    image_data = load_image_data(image_list, n_pix=256)
    image_data_filename = os.path.join(folder_save, '%s_preproc.npy'%(image_set_name))
    print('saving to %s'%(image_data_filename))
    sys.stdout.flush()
    np.save(image_data_filename, image_data)
    
    

def load_image_data(image_list, n_pix=256):
    
    n_images = len(image_list)

    image_data = np.zeros([n_images, 3, n_pix, n_pix],dtype=int)
    
    for ii in range(n_images):
        filename = np.array(image_list['image_filename'])[ii]
        # if np.mod(ii, 100)==0:
        print('loading from %s'%filename)
        sys.stdout.flush()
        im = PIL.Image.open(filename)

        print('orig size: %d by %d'%(im.size[0], im.size[1]))
        
        imdat = copy.deepcopy(np.array(im))

        # crop to a square (if not already square)
        # this always takes center of the longer side
        imdat_cropped, bbox = segmentation_utils.crop_to_square(imdat)

        # back to PIL format
        im_cropped = PIL.Image.fromarray(imdat_cropped.astype(np.uint8))

        print('size after crop: %d by %d'%(im_cropped.size[0], im_cropped.size[1]))
        
        assert(im_cropped.size[0]==im_cropped.size[1])
        # im_resized = im.resize([n_pix, n_pix], resample=PIL.Image.BILINEAR)
        im_resized = im_cropped.resize([n_pix, n_pix], resample=PIL.Image.Resampling.LANCZOS)

        if im_resized.mode!='RGB':
            im_resized = im_resized.convert('RGB')

        print('size after resize: %d by %d'%(im_resized.size[0], im_resized.size[1]))
        
        image_array = np.reshape(np.array(im_resized.getdata()), [n_pix, n_pix, 3])
        image_data[ii,:,:,:] = np.moveaxis(image_array, [2],[0])

    return image_data
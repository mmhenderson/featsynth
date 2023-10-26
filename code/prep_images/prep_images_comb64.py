import os, sys
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import copy

image_path = '/user_data/mmhender/stimuli/featsynth/'
project_root = '/user_data/mmhender/featsynth/'

sys.path.append('/user_data/mmhender/featsynth/texture_synthesis/code/')
import utilities, segmentation_utils


def prep(image_set_name = 'images_comb64'):

    # for these images, the full list to use is already made.
    # made this in choose_extra_ecoset_ims.py
    folder = os.path.join(project_root, 'features','raw')
    image_list_filename = os.path.join(folder, 'images_comb64_list.csv')
    print(image_list_filename)
    image_list = pd.read_csv(image_list_filename, index_col=0)

    n_images = image_list.shape[0]

    # for these images i am going to save them as individual files, 
    # within folders organized by category.
    # first figuring out what the final names of the files will be
    bname_list = np.array(image_list['basic_name'])
    fn_list = list(np.array(image_list['image_filename']))
    fn_list = [f.split('/')[-1] for f in fn_list]
    # changing extensions to .png here, this preserves the image values
    # more accurately than .jpg
    fn_list = [f.split('.')[0]+'.png' for f in fn_list]
    
    image_save_folder = os.path.join(image_path, 'images_comb64_orig')
    if not os.path.exists(image_save_folder):
        os.makedirs(image_save_folder)
    for bname in np.unique(bname_list):
        if not os.path.exists(os.path.join(image_save_folder, bname)):
            os.makedirs(os.path.join(image_save_folder, bname))
                
    image_save_names = [os.path.join(image_save_folder, \
                                     bname_list[ii], fn_list[ii]) \
                        for ii in range(n_images)]

    # making new column here with the new names
    image_list['image_filename_new'] = image_save_names

    # going to choose random seeds to use with each image when we do texture synth
    allseeds = get_seeds()
    for x in range(allseeds.shape[1]):
        image_list['seed%d'%(x)] = allseeds[:,x]
    
    print('saving to %s'%image_list_filename)
    sys.stdout.flush()
    image_list.to_csv(image_list_filename)  

    # saving this also to the folder where images live
    image_list_filename2 = os.path.join(image_save_folder, 'images_comb64_list.csv')
    print('saving to %s'%image_list_filename2)
    sys.stdout.flush()
    image_list.to_csv(image_list_filename2) 
    
    image_data = load_image_data(image_list, n_pix=256)

    # also going to save a big brick file with all the images
    folder_save = os.path.join(project_root, 'features','raw')
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

        # save the individual preprocessed image to disk here.
        imfn_save = image_list['image_filename_new'][ii]
        print(imfn_save)
        im_resized.save(imfn_save)
        
        image_array = np.reshape(np.array(im_resized.getdata()), [n_pix, n_pix, 3])
        image_data[ii,:,:,:] = np.moveaxis(image_array, [2],[0])

    
    return image_data


def get_seeds():
    
    n_categ = 64
    n_ex = 40
    n_imtypes = 5
    n_each = 10

    # for each image, creating a bunch of seeds in case we need several ims
    n_seeds = n_categ * n_ex * n_imtypes * n_each
    
    s = 234534
    allseeds = np.array([s + i*3 for i in range(n_seeds)])
    
    np.random.seed(353545)
    
    allseeds = allseeds[np.random.permutation(n_seeds)]
    
    allseeds = np.reshape(allseeds, [n_categ * n_ex, n_imtypes*n_each])

    return allseeds
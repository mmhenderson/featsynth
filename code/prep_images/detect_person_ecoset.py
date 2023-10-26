import argparse
import os
import sys
import numpy as np
import time

import PIL.Image

import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


ecoset_path = '/lab_data/tarrlab/common/datasets/Ecoset/'
ecoset_info_path = '/user_data/mmhender/stimuli/ecoset_info/'
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

    
def get_person_labels(debug=False, batch_size=10, score_thresh=0.90):
    
    # info about the ecoset categories of interest for our experiment
    fn = os.path.join(ecoset_info_path, 'categ_use_ecoset.npy')
    info = np.load(fn, allow_pickle=True).item()
    basic_names = list(info['binfo'].keys())

    
    fn = os.path.join(ecoset_info_path, 'ecoset_names.npy')
    efolders = np.load(fn, allow_pickle=True).item()
    
    # list of all files in each category
    fn = os.path.join(ecoset_info_path, 'ecoset_file_info.npy')
    efiles = np.load(fn, allow_pickle=True).item()

    
    basic_names_test = basic_names
    
    
    if debug:
        fn2save = os.path.join(ecoset_info_path, 'ecoset_files_detect_person_DEBUG.npy')
    else:
        fn2save = os.path.join(ecoset_info_path, 'ecoset_files_detect_person.npy')
    print('will save to %s'%fn2save)
    
    if os.path.exists(fn2save):
        print('loading from %s'%fn2save)
        has_person_all = np.load(fn2save, allow_pickle=True).item()
    else:
        has_person_all = dict()
    
    pix_thresh = 256
    
    # initialize the pre-trained model to use
    # trained on COCO dataset
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    
    model.eval()
    
    model.to(device)
    
    # figure out which label is "person"
    categ_names = weights.meta["categories"]
    categ_num = np.where(['person' in c for c in categ_names])[0][0]
        
    
    for bi, bname in enumerate(basic_names):
        
        
        if debug and bi>1:
            continue
            
        print('\ncategory %d of %d: %s'%(bi, len(basic_names), bname))
        
        if bname in has_person_all.keys():
            print('done already, skipping')
            continue
            
        cst = time.time()    
        
        has_person = dict()
    
        # choose images to analyze here
        # folder = os.path.join(ecoset_path, 'train', info['binfo'][bname]['ecoset_folder'])
        folder = os.path.join(ecoset_path, 'train', efolders[bname])
        imfiles_all = np.array(efiles[bname]['train']['images'])
        sizes = efiles[bname]['train']['size']
        abv_thresh = np.array([(s1>=pix_thresh) & (s2>=pix_thresh) for s1, s2 in sizes])
        is_rgb = np.array(efiles[bname]['train']['mode'])=='RGB'
        
        # only look for people in the images that met other criteria
        do_test = abv_thresh & is_rgb

        imfiles_test = imfiles_all[do_test]
        imfiles_skip = imfiles_all[~do_test]
        
        for file in imfiles_skip:
            
            has_person[file] = np.nan
            
        n_ims_test = len(imfiles_test)
        n_batches = int(np.ceil(n_ims_test/batch_size))
        
        for ba in range(n_batches):
            
            if debug and ba>1:
                continue
            
            batch_inds = np.arange(batch_size*ba, np.min([batch_size*(ba+1), n_ims_test]))
            # print(batch_inds)
            
            imfiles_batch = imfiles_test[batch_inds]
            
            print('processing batch %d of %d, images:'%(ba, n_batches))
            print(imfiles_batch)
            sys.stdout.flush()
            
            # loading images and getting into correct format
            imfiles_batch_full = [os.path.join(folder, imfile) for imfile in imfiles_batch]
            
            im_list = [PIL.Image.open(fn) \
                       for fn in imfiles_batch_full]
            im_array_list = [np.reshape(np.array(im.getdata()), [im.size[1], im.size[0], 3]) \
                             for im in im_list]
            im_array_list = [im_array.astype(np.float32)/255 \
                             for im_array in im_array_list]
            im_tensor_list = [torch.Tensor(np.moveaxis(im_array, [2], [0])).to(device) \
                              for im_array in im_array_list]

            # run the images through the model here
            st = time.time()
            out_batch = model(im_tensor_list)
            elapsed = time.time() - st
            print('took %.5f sec for batch of size %d'%(elapsed, len(imfiles_batch)))
            sys.stdout.flush()
            
            for out, imfile in zip(out_batch, imfiles_batch):
                
                # looking for labels of the person category
                # with score above our threshold
                inds = np.where((out['labels'].detach().cpu().numpy()==categ_num) & \
                                (out['scores'].detach().cpu().numpy()>score_thresh))[0]

                has_person[imfile] = len(inds)>0
                
        celapsed = time.time() - cst
        print('took %.5f sec (%.5f min) to process %s'%(celapsed, celapsed/60, bname))
        has_person_all[bname] = has_person
     
    
    print('saving to %s'%fn2save)
    np.save(fn2save, has_person_all, allow_pickle=True)
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--score_thresh", type=float,default=0.90,
                    help="threshold for score assigned to each object")
    
    parser.add_argument("--batch_size", type=int,default=10,
                    help="batch size for passing images into model")
    
    args = parser.parse_args()

    get_person_labels(debug=args.debug==1, \
                      batch_size=args.batch_size, \
                      score_thresh=args.score_thresh)
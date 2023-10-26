import sys, os
import numpy as np
import time
import torch

import PIL

# clip implemented in this package, from:
# https://github.com/openai/CLIP
import clip

project_root = '/user_data/mmhender/featsynth/'


def init_cuda():
    
    # Set up CUDA stuff
    
    print ('#device:', torch.cuda.device_count())
    print ('device#:', torch.cuda.current_device())
    print ('device name:', torch.cuda.get_device_name(torch.cuda.current_device()))

    torch.manual_seed(time.time())
    device = torch.device("cuda:0") #cuda
    torch.backends.cudnn.enabled=True

    print ('\ntorch:', torch.__version__)
    print ('cuda: ', torch.version.cuda)
    print ('cudnn:', torch.backends.cudnn.version())
    print ('dtype:', torch.get_default_dtype())

    return device

try:
    device = init_cuda()
except:
    device = 'cpu:0'


def get_embed(debug=False, image_set_name = 'images_ecoset64', grayscale=False):

    debug = debug==1
    grayscale = grayscale==1

    print('debug=%s, grayscale=%s'%(debug, grayscale))
    
    feat_path = os.path.join(project_root, 'features', 'clip')
        
    if debug:
        feat_path = os.path.join(feat_path,'DEBUG')

    if not os.path.exists(feat_path):
        os.makedirs(feat_path)

    # load images to process
    folder_images = os.path.join(project_root, 'features','raw')
    image_data_filename = os.path.join(folder_images, '%s_preproc.npy'%(image_set_name))
    print('loading images from %s'%image_data_filename)
    image_data = np.load(image_data_filename)
    n_images = image_data.shape[0]

    n_feat = 1024;

    f = np.zeros((n_images, n_feat), dtype=np.float32)

    # get the model ready 
    model_architecture='RN50'
    
    print('Using CLIP model')
    model, preprocess = clip.load(model_architecture, device=device)


    for ii in range(n_images):

        if debug and ii>1:
            continue
            
        print('proc image %d of %d'%(ii, n_images))
        sys.stdout.flush()
        
        st = time.time()
        # get image into format the model is expecting
        im = PIL.Image.fromarray(np.moveaxis(image_data[ii].astype(np.uint8),[0],[2]))

        im_prep = preprocess(im).unsqueeze(0).to(device)
        
        if grayscale:
            print('convert image to grayscale')
            # converting to to grayscale
            # L = R * 299/1000 + G * 587/1000 + B * 114/1000
            # then back to RGB to get shape right
            im = im.convert('L').convert('RGB')
            # print(len(np.unique(im.getdata())))

        out = model.encode_image(im_prep)
        
        out = out.detach().cpu().numpy()

        # if ii==0:
        #     print(image_data[ii,:,0:10,0:10])
        #     print(out)
        #     print(out.shape)
            
        f[ii,:] = out

        if ii==0:
            print(f[ii,:])

        elapsed = time.time() - st
        print('took %.5f s to proc image'%elapsed)
        sys.stdout.flush()


    if grayscale:
        feat_file_name = os.path.join(feat_path, \
                                      '%s_grayscale_clip_embed.npy'%(image_set_name))
    else:
        feat_file_name = os.path.join(feat_path, \
                                      '%s_clip_embed.npy'%(image_set_name))
        
    print('size of f is:')
    print(f.shape)
    print(f[0,:])
    print('saving to %s'%feat_file_name)
    np.save(feat_file_name, f)
    print('done saving')
    print('loading from %s'%feat_file_name)
    f = np.load(feat_file_name)
    print(f[0,:])
    
    
    
import argparse
import os
import sys

import torch
import matplotlib.pyplot as plt     # type: ignore

import utilities
import model
import optimize
import pandas as pd
import time

from utils import default_paths

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

root = os.getcwd()

def make_rfs(args):
    
    """
    Estimate the spatial RF of each unit in specified layers of VGG-19.
    RFs are binary and in pixel space.
    """
    model = net(device)

    n_pix = 256
    
    pixel_centers_x, pixel_centers_y = np.meshgrid(np.arange(n_pix), np.arange(n_pix))
    pixel_centers = list(zip(np.ravel(pixel_centers_y), np.ravel(pixel_centers_x)))

    vgg19_layer_names = model.layer_names

    layer_names_use = ['Conv1', 'MaxPool1','MaxPool2','MaxPool3','MaxPool4']
    layer_inds = [np.where([n==name for n in vgg19_layer_names])[0][0] \
                                                  for name in layer_names_use]
    print(layer_names_use, layer_inds)
    
    template_batch = torch.Tensor(np.zeros((1,3,n_pix, n_pix))).to(device)
    out = model(template_batch)
    activs = [model.activs[ll] for ll in layer_inds]
    res_each_layer = [aa.shape[2] for aa in activs]
    model.init_activs()
    print(res_each_layer)
    
    spatunits_each_layer = [res**2 for res in res_each_layer]

    n_total_pix = n_pix**2

    layer_unit_centers = []
    for n_units in res_each_layer:

        unit_centers_x, unit_centers_y = np.meshgrid(np.arange(n_units), np.arange(n_units)) 
        centers = list(zip(unit_centers_y.ravel(), unit_centers_x.ravel())) 
        layer_unit_centers.append(centers)
        print(len(centers))
    
    batch_size=1;

    n_batches = int(np.ceil(n_total_pix/batch_size))

    rfs_each_layer = [np.zeros([n_total_pix, n], dtype=bool) for n in spatunits_each_layer]

    for bb in range(n_batches):
        
        if args.debug and bb>100:
            continue

        if np.mod(bb,100)==0:
            print('processing pixel batch %d of %d'%(bb, n_batches))

        st = time.time()
        pixels_batch = np.arange(bb*batch_size, np.minimum((bb+1)*batch_size, n_total_pix))

        # make an "image" with one pixel at 1 and rest at 0 
        image_batch = torch.Tensor(np.zeros((len(pixels_batch),3,n_pix, n_pix))).to(device)                  
        for ii, pix in enumerate(pixels_batch):
            center = pixel_centers[pix]
            image_batch[ii,:,center[0], center[1]] = 1.0

        with torch.no_grad():

            out = model(image_batch)
            activs = model.activs
            model.init_activs()

        for ii, ll in enumerate(layer_inds):
            
            act = activs[ll]

            a = act[:,0,:,:]>0

            rfs_each_layer[ii][pixels_batch,:] = np.reshape(a, [len(pixels_batch), -1])

        if np.mod(bb,100)==0:
            elapsed = time.time() - st;
            print('batch took %.5f s total'%elapsed)
    
    for ll, rfs in enumerate(rfs_each_layer):
        if not os.path.exists(os.path.join(root,'rfs')):
            os.makedirs(os.path.join(root,'rfs'))
        fn2save_rfs = os.path.join(root,'rfs','vgg19_unit_rfs_%s.npy'%layer_names_use[ll])
        print('saving to %s'%fn2save_rfs)
        np.save(fn2save_rfs, rfs_each_layer[ll])

    fn2save_info = os.path.join(root,'rfs','vgg19_rfs_info.npy')
    print('saving to %s'%fn2save_info)
    np.save(fn2save_info, { 'pixel_centers': pixel_centers, 
                     'layer_unit_centers': layer_unit_centers, 
                     'layer_names_use': layer_names_use, 
                     'layer_inds':layer_inds}, 
                     allow_pickle=True)

class net(torch.nn.Module):
    
    """
    This is a simplified implementation of VGG-19. It is used for estimating 
    the receptive field coverage for units in each layer of VGG-19. 
    """
    def __init__(self, device):
        
        super().__init__()
    
        self.device = device
        self.n_chunks = 5;
        self.n_convs_per_chunk = [2, 2, 4, 4, 4]
        
        self._get_layer_names()
        self.n_layers = len(self.layer_names)
        
        self._get_modules()
        
        self.init_activs()
        
        
    def init_activs(self):
        
        self.activs = [[] for ll in range(self.n_layers)]
    
    def _get_layer_names(self):
        
        names = []
        conv_counter = 0;
        for ch in range(self.n_chunks):
            for cv in range(self.n_convs_per_chunk[ch]):   
                conv_counter+=1  
                names+= ['Conv%d'%conv_counter]
                # names+= ['ReLU%d'%conv_counter]   
            names+= ['MaxPool%d'%(ch+1)]

        self.layer_names = names
        
    def _get_modules(self):
        
        modules = []
        conv_counter = -1;
        in_channels_conv = [3, 64,64, 128,128, 256,256,256,256, 512,512,512,512,512,512,512]
        out_channels_conv = [64, 64,128, 128,256, 256,256,256,512, 512,512,512,512,512,512,512]
        for ch in range(self.n_chunks):
            for cv in range(self.n_convs_per_chunk[ch]):   
                conv_counter+=1  
                mod = torch.nn.Conv2d(in_channels = in_channels_conv[conv_counter],
                                          out_channels = out_channels_conv[conv_counter], 
                                          kernel_size=(3,3), stride=(1,1), padding=(1,1), 
                                          bias = False)
                mod.weight.data.fill_(1.0)
                modules.append(mod.to(self.device))
            # mod = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            mod = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            modules.append(mod.to(self.device))
            
        self.modules = modules
        
    def forward(self, image_batch):
        
        current = image_batch
        
        for ii, mod in enumerate(self.modules):
            
            a = mod(current)
            clamp_min = torch.Tensor([0.0]).to(self.device)
            clamp_max = torch.Tensor([1.0]).to(self.device)
            a = torch.minimum(torch.maximum(a, clamp_min), clamp_max)
            self.activs[ii] = a.detach().cpu().numpy()
            
            current = a
            
        return current
    
      
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
   
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
   
    args = parser.parse_args()

    args.debug=args.debug==1
    
    make_rfs(args)

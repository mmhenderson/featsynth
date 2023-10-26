import argparse
import os
import sys

import torch
import matplotlib.pyplot as plt     # type: ignore

import pandas as pd
import time

import numpy as np

device='cpu:0'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

root = os.path.dirname(os.getcwd())


def get_overlap(args):
    
    fn_info = os.path.join(root,'rfs','vgg19_rfs_info.npy')
    rfs_info = np.load(fn_info, allow_pickle=True).item()    
    
    pixel_centers = rfs_info['pixel_centers']
    layer_unit_centers = rfs_info['layer_unit_centers']
    res_each_layer = [np.max(l)+1 for l in layer_unit_centers]
    layer_names = rfs_info['layer_names_use']
    layer_inds = rfs_info['layer_inds']
    n_layers = len(layer_names)
    n_pix = np.max(pixel_centers)+1
    
    rfs_each_layer = []
    for ll, layer in enumerate(['Conv1','MaxPool1','MaxPool2','MaxPool3','MaxPool4']):
        fn_rfs = os.path.join(root,'rfs','vgg19_unit_rfs_%s.npy'%layer)
        rfs = np.load(fn_rfs).astype(np.float32)
        rfs_each_layer.append(rfs)

        
    n_grid_eachside=args.n_grid_eachside
    
    print(n_grid_eachside)
     
    # create the grid
    if args.which_grid==1:
        spatial_weights_pixel = make_square_grid_smooth_grid1(n_grid_eachside=n_grid_eachside, n_pix=n_pix)
    elif args.which_grid==2:
        spatial_weights_pixel = make_square_grid_smooth_grid2(n_grid_eachside=n_grid_eachside, n_pix=n_pix)
    elif args.which_grid==3:
        spatial_weights_pixel = make_square_grid_smooth_grid3(n_grid_eachside=n_grid_eachside, n_pix=n_pix)
    elif args.which_grid==4:
        spatial_weights_pixel = make_square_grid_smooth_grid2(n_grid_eachside=n_grid_eachside, n_pix=n_pix, ramp_size=40)
    elif args.which_grid==5:
        spatial_weights_pixel = make_square_grid_smooth_grid5(n_grid_eachside=n_grid_eachside, n_pix=n_pix)
        
    if n_grid_eachside==1:
        # if only 1 grid pos, make sure it's constant (no blurred edges)
        spatial_weights_pixel = np.ones(np.shape(spatial_weights_pixel), dtype=np.float32)
        
    overlap_each_layer = []
    grid_weights = np.moveaxis(spatial_weights_pixel,[2],[0])
    grid_weights = np.reshape(grid_weights, [n_grid_eachside**2, -1])
    grid_weights = torch.Tensor(grid_weights).to(device)

    for ll in range(n_layers):

        r = torch.Tensor(rfs_each_layer[ll]).to(device)
        # multiply the pixel-representation of each spatial pooling field, times
        # the pixel-representation of each unit's RF.
        # overlap is [n_grid_total x n_spat_units]
        overlap = grid_weights @ r
        # normalize each unit's overlap by the sum of the unit's RF
        overlap /= torch.sum(r, dim=0, keepdims=True)

        overlap_each_layer.append(overlap.detach().cpu().numpy())

    if not os.path.exists(os.path.join(root,'grid_overlap')):
        os.makedirs(os.path.join(root,'grid_overlap'))
            
    for ll in range(n_layers):
        
        fn2save = os.path.join(root,'grid_overlap','vgg19_gridoverlap_grid%d_%dx%d_%s.npy'%(args.which_grid, n_grid_eachside, n_grid_eachside, layer_names[ll]))
        print('saving to %s'%fn2save)
        np.save(fn2save, overlap_each_layer[ll])
        

def make_square_grid_smooth_grid1(n_grid_eachside=2, n_pix=256, ramp_size=20):

    n_grid_total = n_grid_eachside**2
    grid_bounds = np.floor(np.linspace(0,n_pix, n_grid_eachside+1)).astype(int)

    # define ramping edge function (squared cosine)
    xvals_ramp = np.linspace(-np.pi/2,0, ramp_size)
    ramp_vals = np.cos(xvals_ramp)**2
    
    # for the corners, use distance from corner to get round edge
    center = [0,0]
    x, y = np.meshgrid(np.arange(ramp_size), np.arange(ramp_size))
    distance = np.sqrt((x-center[0])**2 + (y-center[1])**2)
    distance /= (ramp_size-1)
    distance = -distance * np.pi/2
    distance = np.maximum(distance, -np.pi/2)
    ramp_corner = np.cos(distance)**2

    spatial_weights_pixel = np.zeros((n_pix, n_pix, n_grid_total))

    gt = -1;
    for gx in range(n_grid_eachside):
        for gy in range(n_grid_eachside):

            gt+=1
            # xrange = np.arange(grid_bounds[gx], grid_bounds[gx+1])
            # yrange = np.arange(grid_bounds[gy], grid_bounds[gy+1])
            xrange = [grid_bounds[gx], grid_bounds[gx+1]]
            yrange = [grid_bounds[gy], grid_bounds[gy+1]]

            spatial_weights_pixel[xrange[0]:xrange[1], yrange[0]:yrange[1], gt] = 1

            # add in ramp for each edge region 
            # (there is probably a more efficient way to do this...)
            if xrange[0]>0:

                ydist = yrange[1]-yrange[0]
                ramp_tiled = np.tile(ramp_vals[:,None], [1, int(ydist)])
                spatial_weights_pixel[xrange[0]-ramp_size: xrange[0], \
                                      yrange[0]:yrange[1], gt] = \
                                        ramp_tiled

            if xrange[1]<n_pix:

                ydist = yrange[1]-yrange[0]
                ramp_tiled = np.tile(np.flipud(ramp_vals[:,None]), [1, int(ydist)])
                spatial_weights_pixel[xrange[1]: xrange[1]+ramp_size, \
                                      yrange[0]:yrange[1], gt] = \
                                        ramp_tiled


            if yrange[0]>0:

                xdist = xrange[1]-xrange[0]
                ramp_tiled = np.tile(ramp_vals[None,:], [int(xdist),1])
                spatial_weights_pixel[xrange[0]:xrange[1], \
                                      yrange[0]-ramp_size:yrange[0], gt] = \
                                        ramp_tiled

            if yrange[1]<n_pix:

                xdist = xrange[1]-xrange[0]
                ramp_tiled = np.tile(np.fliplr(ramp_vals[None,:]), [int(xdist),1])
                spatial_weights_pixel[xrange[0]:xrange[1], \
                                      yrange[1]:yrange[1]+ramp_size, gt] = \
                                        ramp_tiled

            # now add in the ramps for corner regions
            if xrange[0]>0 and yrange[0]>0:

                spatial_weights_pixel[xrange[0]-ramp_size: xrange[0], 
                                      yrange[0]-ramp_size: yrange[0], gt] = \
                                        np.flipud(np.fliplr(ramp_corner))

            if xrange[0]>0 and yrange[1]<n_pix:

                spatial_weights_pixel[xrange[0]-ramp_size: xrange[0], 
                                      yrange[1]: yrange[1]+ramp_size, gt] = \
                                        np.flipud(ramp_corner)

            if xrange[1]<n_pix and yrange[0]>0:

                spatial_weights_pixel[xrange[1]: xrange[1]+ramp_size, 
                                      yrange[0]-ramp_size: yrange[0], gt] = \
                                        np.fliplr(ramp_corner)

            if xrange[1]<n_pix and yrange[1]<n_pix:

                spatial_weights_pixel[xrange[1]: xrange[1]+ramp_size, 
                                      yrange[1]: yrange[1]+ramp_size, gt] = \
                                        ramp_corner


    return spatial_weights_pixel
        
def make_square_grid_smooth_grid2(n_grid_eachside=2, n_pix=256, ramp_size=20):

    n_grid_total = n_grid_eachside**2
    grid_bounds = np.floor(np.linspace(0,n_pix, n_grid_eachside+1)).astype(int)

    # define ramping edge function (squared cosine)
    xvals_ramp = np.linspace(-np.pi/2,0, ramp_size)
    ramp_vals = np.cos(xvals_ramp)**2
    ramp_size_half = int(np.ceil(ramp_size/2))
    pad_by = ramp_size_half
    
    # for the corners, use distance from corner to get round edge
    center = [0,0]
    x, y = np.meshgrid(np.arange(ramp_size), np.arange(ramp_size))
    distance = np.sqrt((x-center[0])**2 + (y-center[1])**2)
    distance /= (ramp_size-1)
    distance = -distance * np.pi/2
    distance = np.maximum(distance, -np.pi/2)
    ramp_corner = np.cos(distance)**2

    # spatial_weights_pixel = np.zeros((n_pix, n_pix, n_grid_total))
    
    # make the array with some padding on each side, this will make it easier to fill 
    # in the ramping regions.
    spatial_weights_pixel = np.zeros((n_pix+pad_by*2, n_pix+pad_by*2, n_grid_total))

    gt = -1;
    for gx in range(n_grid_eachside):
        for gy in range(n_grid_eachside):

            gt+=1
            
            xrange = [grid_bounds[gx]+ramp_size_half+pad_by, grid_bounds[gx+1]-ramp_size_half+pad_by]
            yrange = [grid_bounds[gy]+ramp_size_half+pad_by, grid_bounds[gy+1]-ramp_size_half+pad_by]

            spatial_weights_pixel[xrange[0]:xrange[1], yrange[0]:yrange[1], gt] = 1

            # add in ramp for each edge region 
            ydist = yrange[1]-yrange[0]
            ramp_tiled = np.tile(ramp_vals[:,None], [1, int(ydist)])
            spatial_weights_pixel[xrange[0]-ramp_size: xrange[0], \
                                  yrange[0]:yrange[1], gt] = \
                                    ramp_tiled
            
            ydist = yrange[1]-yrange[0]
            ramp_tiled = np.tile(np.flipud(ramp_vals[:,None]), [1, int(ydist)])
            spatial_weights_pixel[xrange[1]: xrange[1]+ramp_size, \
                                  yrange[0]:yrange[1], gt] = \
                                    ramp_tiled
            
            xdist = xrange[1]-xrange[0]
            ramp_tiled = np.tile(ramp_vals[None,:], [int(xdist),1])
            spatial_weights_pixel[xrange[0]:xrange[1], \
                                  yrange[0]-ramp_size:yrange[0], gt] = \
                                    ramp_tiled
                
            xdist = xrange[1]-xrange[0]
            ramp_tiled = np.tile(np.fliplr(ramp_vals[None,:]), [int(xdist),1])
            spatial_weights_pixel[xrange[0]:xrange[1], \
                                  yrange[1]:yrange[1]+ramp_size, gt] = \
                                    ramp_tiled

            spatial_weights_pixel[xrange[0]-ramp_size: xrange[0], 
                                  yrange[0]-ramp_size: yrange[0], gt] = \
                                    np.flipud(np.fliplr(ramp_corner))
                
            spatial_weights_pixel[xrange[0]-ramp_size: xrange[0], 
                                  yrange[1]: yrange[1]+ramp_size, gt] = \
                                    np.flipud(ramp_corner)

            spatial_weights_pixel[xrange[1]: xrange[1]+ramp_size, 
                                  yrange[0]-ramp_size: yrange[0], gt] = \
                                    np.fliplr(ramp_corner)

            spatial_weights_pixel[xrange[1]: xrange[1]+ramp_size, 
                                  yrange[1]: yrange[1]+ramp_size, gt] = \
                                    ramp_corner
             
    # un-pad the array here
    spatial_weights_pixel = spatial_weights_pixel[pad_by:pad_by+n_pix, pad_by:pad_by+n_pix,:]
    
    return spatial_weights_pixel


def make_square_grid_smooth_grid3(n_grid_eachside=2, n_pix=256, ramp_size=20):

    n_grid_total = n_grid_eachside**2
    grid_bounds = np.floor(np.linspace(0,n_pix, n_grid_eachside+1)).astype(int)

    grid_space = np.diff(grid_bounds)[0]
    assert(grid_space>=ramp_size*2) # can't have too fine a grid relative to the ramp size
    
    # define ramping edge function (squared cosine)
    xvals_ramp = np.linspace(-np.pi/2,0, ramp_size)
    ramp_vals = np.cos(xvals_ramp)**2
    # ramp_size_half = int(np.ceil(ramp_size/2))
    pad_by = 0
    
    # for the corners, use distance from corner to get round edge
    center = [0,0]
    x, y = np.meshgrid(np.arange(ramp_size), np.arange(ramp_size))
    distance = np.sqrt((x-center[0])**2 + (y-center[1])**2)
    distance /= (ramp_size-1)
    distance = -distance * np.pi/2
    distance = np.maximum(distance, -np.pi/2)
    ramp_corner = np.cos(distance)**2

    # spatial_weights_pixel = np.zeros((n_pix, n_pix, n_grid_total))
    
    # make the array with some padding on each side, this will make it easier to fill 
    # in the ramping regions.
    spatial_weights_pixel = np.zeros((n_pix+pad_by*2, n_pix+pad_by*2, n_grid_total))

    gt = -1;
    for gx in range(n_grid_eachside):
        for gy in range(n_grid_eachside):

            gt+=1
            
            xrange = [grid_bounds[gx]+ramp_size+pad_by, grid_bounds[gx+1]-ramp_size+pad_by]
            yrange = [grid_bounds[gy]+ramp_size+pad_by, grid_bounds[gy+1]-ramp_size+pad_by]

            spatial_weights_pixel[xrange[0]:xrange[1], yrange[0]:yrange[1], gt] = 1

            # add in ramp for each edge region 
            ydist = yrange[1]-yrange[0]
            ramp_tiled = np.tile(ramp_vals[:,None], [1, int(ydist)])
            spatial_weights_pixel[xrange[0]-ramp_size: xrange[0], \
                                  yrange[0]:yrange[1], gt] = \
                                    ramp_tiled
            
            ydist = yrange[1]-yrange[0]
            ramp_tiled = np.tile(np.flipud(ramp_vals[:,None]), [1, int(ydist)])
            spatial_weights_pixel[xrange[1]: xrange[1]+ramp_size, \
                                  yrange[0]:yrange[1], gt] = \
                                    ramp_tiled
            
            xdist = xrange[1]-xrange[0]
            ramp_tiled = np.tile(ramp_vals[None,:], [int(xdist),1])
            spatial_weights_pixel[xrange[0]:xrange[1], \
                                  yrange[0]-ramp_size:yrange[0], gt] = \
                                    ramp_tiled
                
            xdist = xrange[1]-xrange[0]
            ramp_tiled = np.tile(np.fliplr(ramp_vals[None,:]), [int(xdist),1])
            spatial_weights_pixel[xrange[0]:xrange[1], \
                                  yrange[1]:yrange[1]+ramp_size, gt] = \
                                    ramp_tiled

            spatial_weights_pixel[xrange[0]-ramp_size: xrange[0], 
                                  yrange[0]-ramp_size: yrange[0], gt] = \
                                    np.flipud(np.fliplr(ramp_corner))
                
            spatial_weights_pixel[xrange[0]-ramp_size: xrange[0], 
                                  yrange[1]: yrange[1]+ramp_size, gt] = \
                                    np.flipud(ramp_corner)

            spatial_weights_pixel[xrange[1]: xrange[1]+ramp_size, 
                                  yrange[0]-ramp_size: yrange[0], gt] = \
                                    np.fliplr(ramp_corner)

            spatial_weights_pixel[xrange[1]: xrange[1]+ramp_size, 
                                  yrange[1]: yrange[1]+ramp_size, gt] = \
                                    ramp_corner
             
    # un-pad the array here
    spatial_weights_pixel = spatial_weights_pixel[pad_by:pad_by+n_pix, pad_by:pad_by+n_pix,:]
    
    return spatial_weights_pixel

      
def make_square_grid_smooth_grid5(n_grid_eachside=2, n_pix=256, ramp_size=20):

    n_grid_total = n_grid_eachside**2
    grid_bounds = np.floor(np.linspace(0,n_pix, n_grid_eachside+1)).astype(int)

    # define ramping edge function (squared cosine)
    xvals_ramp = np.linspace(-np.pi/2,0, ramp_size)
    ramp_vals = np.cos(xvals_ramp)**2
    ramp_size_half = int(np.ceil(ramp_size/2))
    pad_by = ramp_size_half
    
    # for the corners, use distance from corner to get round edge
    center = [0,0]
    x, y = np.meshgrid(np.arange(ramp_size), np.arange(ramp_size))
    distance = np.sqrt((x-center[0])**2 + (y-center[1])**2)
    distance /= (ramp_size-1)
    distance = -distance * np.pi/2
    distance = np.maximum(distance, -np.pi/2)
    ramp_corner = np.cos(distance)**2

    # spatial_weights_pixel = np.zeros((n_pix, n_pix, n_grid_total))
    
    # make the array with some padding on each side, this will make it easier to fill 
    # in the ramping regions.
    spatial_weights_pixel = np.zeros((n_pix+pad_by*2, n_pix+pad_by*2, n_grid_total))

    gt = -1;
    for gx in range(n_grid_eachside):
        for gy in range(n_grid_eachside):

            gt+=1
            
            xrange = [grid_bounds[gx]+ramp_size_half+pad_by, grid_bounds[gx+1]-ramp_size_half+pad_by]
            yrange = [grid_bounds[gy]+ramp_size_half+pad_by, grid_bounds[gy+1]-ramp_size_half+pad_by]

            spatial_weights_pixel[xrange[0]:xrange[1], yrange[0]:yrange[1], gt] = 1

            # add in ramp for each edge region 
            ydist = yrange[1]-yrange[0]
            ramp_tiled = np.tile(ramp_vals[:,None], [1, int(ydist)])
            if gx==0:
                ramp_tiled = np.ones(ramp_tiled.shape)
            spatial_weights_pixel[xrange[0]-ramp_size: xrange[0], \
                                  yrange[0]:yrange[1], gt] = \
                                    ramp_tiled
            
            ydist = yrange[1]-yrange[0]
            ramp_tiled = np.tile(np.flipud(ramp_vals[:,None]), [1, int(ydist)])
            if gx==(n_grid_eachside-1):
                ramp_tiled = np.ones(ramp_tiled.shape)
            spatial_weights_pixel[xrange[1]: xrange[1]+ramp_size, \
                                  yrange[0]:yrange[1], gt] = \
                                    ramp_tiled
            
            xdist = xrange[1]-xrange[0]
            ramp_tiled = np.tile(ramp_vals[None,:], [int(xdist),1])
            if gy==0:
                ramp_tiled = np.ones(ramp_tiled.shape)
            spatial_weights_pixel[xrange[0]:xrange[1], \
                                  yrange[0]-ramp_size:yrange[0], gt] = \
                                    ramp_tiled
                
            xdist = xrange[1]-xrange[0]
            ramp_tiled = np.tile(np.fliplr(ramp_vals[None,:]), [int(xdist),1])
            if gy==(n_grid_eachside-1):
                ramp_tiled = np.ones(ramp_tiled.shape)
            spatial_weights_pixel[xrange[0]:xrange[1], \
                                  yrange[1]:yrange[1]+ramp_size, gt] = \
                                    ramp_tiled

            if gx==0 and gy==0:
                spatial_weights_pixel[xrange[0]-ramp_size: xrange[0], 
                                  yrange[0]-ramp_size: yrange[0], gt] = \
                                    1
            else:
                spatial_weights_pixel[xrange[0]-ramp_size: xrange[0], 
                                  yrange[0]-ramp_size: yrange[0], gt] = \
                                    np.flipud(np.fliplr(ramp_corner))
                
            if gx==0 and gy==(n_grid_eachside-1):
                spatial_weights_pixel[xrange[0]-ramp_size: xrange[0], 
                                      yrange[1]: yrange[1]+ramp_size, gt] = \
                                    1
            else:
                spatial_weights_pixel[xrange[0]-ramp_size: xrange[0], 
                                      yrange[1]: yrange[1]+ramp_size, gt] = \
                                        np.flipud(ramp_corner)

            if gx==(n_grid_eachside-1) and gy==0:
                spatial_weights_pixel[xrange[1]: xrange[1]+ramp_size, 
                                  yrange[0]-ramp_size: yrange[0], gt] = \
                                    1
            else:
                spatial_weights_pixel[xrange[1]: xrange[1]+ramp_size, 
                                      yrange[0]-ramp_size: yrange[0], gt] = \
                                        np.fliplr(ramp_corner)

            if gx==(n_grid_eachside-1) and gy==(n_grid_eachside-1):
                spatial_weights_pixel[xrange[1]: xrange[1]+ramp_size, 
                                  yrange[1]: yrange[1]+ramp_size, gt] = \
                                     1
            else:
                spatial_weights_pixel[xrange[1]: xrange[1]+ramp_size, 
                                      yrange[1]: yrange[1]+ramp_size, gt] = \
                                        ramp_corner

    # un-pad the array here
    spatial_weights_pixel = spatial_weights_pixel[pad_by:pad_by+n_pix, pad_by:pad_by+n_pix,:]
    
    return spatial_weights_pixel



    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
   
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--n_grid_eachside", type=int,default=4,
                    help="how many grid positions per side?")
   
    parser.add_argument("--which_grid", type=int,default=1,
                    help="which of the spatial grid methods to use?")

    args = parser.parse_args()

    args.debug=args.debug==1
    
    get_overlap(args)

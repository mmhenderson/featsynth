import os
import sys
import numpy as np
import copy

import torch
import torchvision


def filter_lowpass(image, sigma=25):
    
    # filter a single image with Gaussian low-pass filter.
    # image is [npix, npix, 3]
    # sigma is in units of pixels, standard dev of Gaussian.

    npix = image.shape[0]
    means_orig = [np.mean(image[:,:,cc]) for cc in range(3)]
    
    xfreq = np.abs(np.linspace(-0.5, 0.5, npix))
    yfreq = np.abs(np.linspace(-0.5, 0.5, npix))
    xgrid, ygrid = np.meshgrid(xfreq, yfreq)

    # Frequency domain representation of the gaussian kernel has a sigma, related to the 
    # spatial domain sigma by: sigma_spat = 1/(2*pi*sigma_freq)
    sigma_freq = 1/(2*np.pi*sigma)

    lp_gauss = np.exp(-((xgrid)**2 + (ygrid)**2)/(2*sigma_freq**2))
    lp_gauss /= np.max(lp_gauss) # make sure it spans 0-1
    # lp_gauss /= np.sum(lp_gauss)
    filter_freq = lp_gauss
    
    f = np.zeros(np.shape(image))
    
    for cc in range(3):
        vals = filter_fft(image[:,:,cc], filter_freq)
        
        # adjust colors
        vals -= np.mean(vals) # mean-center values
        vals /= (np.max(np.abs(vals))*2) # make them span [-0.5, 0.5]
        vals += means_orig[cc] # adjust mean
        
        f[:,:,cc] = vals
  
    f = f*255
    f = np.maximum(np.minimum(f, 255), 0)
    f = f.astype(np.uint8)

    return f

def filter_highpass(image, sigma=25):
    
     # image is [npix, npix, 3]
   
    npix = image.shape[0]
    means_orig = [np.mean(image[:,:,cc]) for cc in range(3)]
    
    xfreq = np.abs(np.linspace(-0.5, 0.5, npix))
    yfreq = np.abs(np.linspace(-0.5, 0.5, npix))
    xgrid, ygrid = np.meshgrid(xfreq, yfreq)

    # Frequency domain representation of the gaussian kernel has a sigma, related to the 
    # spatial domain sigma by: sigma_spat = 1/(2*pi*sigma_freq)
    sigma_freq = 1/(2*np.pi*sigma)

    lp_gauss = np.exp(-((xgrid)**2 + (ygrid)**2)/(2*sigma_freq**2))
    lp_gauss /= np.max(lp_gauss) # make sure it spans 0-1
    # highpass filter is 1-lowpass filter
    hp_gauss = 1 - lp_gauss
    # hp_gauss /= np.sum(hp_gauss)
    filter_freq = hp_gauss
    
    f = np.zeros(np.shape(image))
    
    for cc in range(3):
        vals = filter_fft(image[:,:,cc], filter_freq)
        
        # adjust colors
        vals -= np.mean(vals) # mean-center values
        vals /= (np.max(np.abs(vals))*2) # make them span [-0.5, 0.5]
        vals += means_orig[cc] # adjust mean
        
        f[:,:,cc] = vals
        
    f = f*255
    f = np.maximum(np.minimum(f, 255), 0)
    f = f.astype(np.uint8)

    return f

def filter_fft(image_2d, filter_2d):

    # image_2d is in spatial domain
    # filter_2d is in freq domain
    
    image_mat_fft = np.fft.fftshift(np.fft.fft2(image_2d))
    image_mat_fft_filt = image_mat_fft * filter_2d
    image_mat_filt = np.fft.ifft2(np.fft.fftshift(image_mat_fft_filt))
    image_mat_filt = np.real(image_mat_filt)
 
    return image_mat_filt

# below are spatial domain versions of filtering.
# these are much slower!

def filter_lowpass_spatial(image, sigma=25):
    
    # image is [npix, npix, 3]
    # image tensor is [3, npix, npix]
    image_tensor = torch.tensor(np.moveaxis(image, [2],[0]))

    means_orig = [np.mean(image[:,:,cc]) for cc in range(3)]
    
    kernel_size = sigma*4
    kernel_size += np.mod(kernel_size-1, 2)
    
    trf = torchvision.transforms.GaussianBlur(kernel_size, sigma)
    
    f_tensor = trf(image_tensor)
    
    f = f_tensor.detach().cpu().numpy()
    f = np.moveaxis(f, [0],[2])
    
    #adjust colors
    for cc in range(3):
        
        vals = copy.deepcopy(f[:,:,cc])
        vals -= np.mean(vals) # mean-center values
        vals /= (np.max(np.abs(vals))*2) # make them span [-0.5, 0.5]
        vals += means_orig[cc] # adjust mean
        
        f[:,:,cc] = vals
        
    f = f*255
    f = np.maximum(np.minimum(f, 255), 0)
    f = f.astype(np.uint8)

    return f

def filter_highpass_spatial(image, sigma=25):
    
    # image is [npix, npix, 3]
    # image tensor is [3, npix, npix]
    image_tensor = torch.tensor(np.moveaxis(image, [2],[0]))

    means_orig = [np.mean(image[:,:,cc]) for cc in range(3)]
    
    kernel_size = sigma*4
    kernel_size += np.mod(kernel_size-1, 2)
    
    trf = torchvision.transforms.GaussianBlur(kernel_size, sigma)
    
    f_lp_tensor = trf(image_tensor)
    
    # highpass = orig image - lowpass
    f_tensor = image_tensor - f_lp_tensor
    
    f = f_tensor.detach().cpu().numpy()
    f = np.moveaxis(f, [0],[2])
    
    # adjust colors
    for cc in range(3):
        vals = copy.deepcopy(f[:,:,cc])
        
        vals -= np.mean(vals) # mean-center values
        vals /= (np.max(np.abs(vals))*2) # make them span [-0.5, 0.5]
        vals += means_orig[cc] # adjust mean
        
        f[:,:,cc] = vals
        
    f = f*255
    f = np.maximum(np.minimum(f, 255), 0)
    f = f.astype(np.uint8)

    return f
import pickle
import sys
import imp
import inspect
import importlib
import copy

import PIL                  # type: ignore
import torch
import torchvision          # type: ignore
import numpy as np          # type: ignore
import scipy.interpolate    # type: ignore

import segmentation_utils

def save_model(model, path):
    """
    Saves the model(s), including the definitions in its containing module.
    Restore the model(s) with load_model. References to other modules
    are not chased; they're assumed to be available when calling load_model.
    The state of any other object in the module is not stored.
    Written by Pauli Kemppinen.
    """
    model_pickle = pickle.dumps(model)

    # Handle dicts, lists and tuples of models.
    model = list(model.values()) if isinstance(model, dict) else model
    model = (
        (model,)
        if not (isinstance(model, list) or isinstance(model, tuple))
        else model
    )

    # Create a dict of modules that maps from name to source code.
    module_names = {m.__class__.__module__ for m in model}
    modules = {
        name:
            inspect.getsource(importlib.import_module(name))
            for name in module_names
    }

    pickle.dump((modules, model_pickle), open(path, 'wb'))


def load_model(path):
    """
    Loads the model(s) stored by save_model.
    Written by Pauli Kemppinen.
    """
    modules, model_pickle = pickle.load(open(path, 'rb'))

    # Temporarily add or replace available modules with stored ones.
    sys_modules = {}
    for name, source in modules.items():
        
        source_adj = source
        source_adj_split = source_adj.split('\n')
        line_replace = source_adj_split[2]
        line_new = 'import collections.abc as container_abcs'
        source_adj_split[2] = line_new
        source_adj_split
        sep = '\n'
        source_adj = sep.join(source_adj_split)
        
        module = imp.new_module(name)
        exec(source_adj, module.__dict__)
        # exec(source, module.__dict__)
        if name in sys.modules:
            sys_modules[name] = sys.modules[name]
        sys.modules[name] = module

    # Map pytorch models to cpu if cuda is not available.
    if imp.find_module('torch'):
        import torch
        original_load = torch.load

        def map_location_cpu(*args, **kwargs):
            kwargs['map_location'] = 'cpu'
            return original_load(*args, **kwargs)
        torch.load = (
            original_load
            if torch.cuda.is_available()
            else map_location_cpu
        )

    model = pickle.loads(model_pickle)

    if imp.find_module('torch'):
        torch.load = original_load  # Revert monkey patch.

    # Revert sys.modules to original state.
    for name in modules.keys():
        if name in sys_modules:
            sys.modules[name] = sys_modules[name]
        else:
            # Just to make sure nobody else depends on these existing.
            sys.modules.pop(name)

    return model


def load_image(path: str) -> PIL.Image.Image:
    
    return PIL.Image.open(path).convert('RGB')

def get_mean_bgr_image(n_pix=256):
    
    # imagenet_mean = np.array([ 0.48501961,  0.45795686, 0.40760392 ]) # mean [RGB]
    imagenet_mean = np.array([ 0.40760392, 0.45795686, 0.48501961,]) # mean [BGR]
    
    mean_to_subtract =  np.tile(imagenet_mean[None,None,:],[n_pix, n_pix,1])
    
    return mean_to_subtract

def get_mean_bw_image(n_pix=256):
    
    # imagenet_mean = np.array([ 0.48501961,  0.45795686, 0.40760392 ]) # mean [RGB]
    imagenet_mean = np.array([ 0.40760392, 0.45795686, 0.48501961,]) # mean [BGR]
    # weighting these values same way as color channels are weighted for RGB>L conversion
    # (but this is [b,g,r])
    # L = R * 299/1000 + G * 587/1000 + B * 114/1000
    imagenet_mean_bw = imagenet_mean[0]*(114/1000) + imagenet_mean[1]*(587/1000) + imagenet_mean[2]*(299/1000)
    
    mean_to_subtract = imagenet_mean_bw * np.ones([n_pix, n_pix,3])
    
    return mean_to_subtract
    
def preprocess_image(
    image: PIL.Image.Image,
    new_size: int = 256,
) -> torch.Tensor:
    

    assert isinstance(image, PIL.Image.Image)

    # pull out numpy data from image
    image_data = copy.deepcopy(np.array(image))

    if image_data.shape[0]!=image_data.shape[1]:
        image_data, bbox = segmentation_utils.crop_to_square(image_data)

    # back to PIL format
    image_preproc = PIL.Image.fromarray(image_data.astype(np.uint8))

    # resize here
    if image_preproc.height!=new_size:
        image_preproc = image_preproc.resize([new_size, new_size], resample=PIL.Image.Resampling.LANCZOS)
    else:
        image_preproc

    assert(image_preproc.height==new_size and image_preproc.width==new_size)

    # going to switch ordering of [r,g,b] channels here to [b,g,r] (matlab compatibility)
    r, g, b = image_preproc.split()
    image_bgr = PIL.Image.merge('RGB', (b, g, r))

    # and back to numpy again 
    image_data = np.array(image_bgr).astype(np.float32) / 255.0
    mean_to_subtract = get_mean_bgr_image(new_size)
    image_data -= mean_to_subtract
    image_data *= 255.0

    # [H, W, C] -> [N, C, H, W]
    image_data = np.transpose(image_data, (2, 0, 1))[None, :, :, :]

    return torch.from_numpy(
        image_data
    ).to(torch.float32)

def preprocess_image_grayscale(
    image: PIL.Image.Image,
    new_size: int = 256,
) -> torch.Tensor:
    

    assert isinstance(image, PIL.Image.Image)

    # pull out numpy data from image
    image_data = copy.deepcopy(np.array(image))

    if image_data.shape[0]!=image_data.shape[1]:
        image_data, bbox = segmentation_utils.crop_to_square(image_data)

    # back to PIL format
    image_preproc = PIL.Image.fromarray(image_data.astype(np.uint8))

    # resize here
    if image_preproc.height!=new_size:
        image_preproc = image_preproc.resize([new_size, new_size], resample=PIL.Image.Resampling.LANCZOS)
    else:
        image_preproc

    assert(image_preproc.height==new_size and image_preproc.width==new_size)

    # converting to to grayscale
    # L = R * 299/1000 + G * 587/1000 + B * 114/1000
    # then back to RGB to get shape right
    image_preproc = image_preproc.convert('L').convert('RGB')
    
    # order of channels doesn't matter here because they are now identical
    # but calling this "bgr"
    image_bgr = image_preproc
  
    # and back to numpy again 
    image_data = np.array(image_bgr).astype(np.float32) / 255.0
    mean_to_subtract = get_mean_bw_image(new_size) # subtract same mean from all channels
    image_data -= mean_to_subtract
    image_data *= 255.0

    # [H, W, C] -> [N, C, H, W]
    image_data = np.transpose(image_data, (2, 0, 1))[None, :, :, :]

    return torch.from_numpy(
        image_data
    ).to(torch.float32)

def preprocess_image_simple(
    image: PIL.Image.Image,
    new_size: int = 256,
) -> torch.Tensor:

    assert isinstance(image, PIL.Image.Image)

    # pull out numpy data from image
    image_data = copy.deepcopy(np.array(image))

    if image_data.shape[0]!=image_data.shape[1]:
        image_data, bbox = segmentation_utils.crop_to_square(image_data)

    # back to PIL format
    image_preproc = PIL.Image.fromarray(image_data.astype(np.uint8))

    # resize here
    if image_preproc.height!=new_size:
        image_preproc = image_preproc.resize([new_size, new_size], resample=PIL.Image.Resampling.LANCZOS)
    else:
        image_preproc

    assert(image_preproc.height==new_size and image_preproc.width==new_size)

    # and back to numpy again 
    image_data = np.array(image_preproc).astype(np.float32)/255
    
    return image_data

def preprocess_image_tosave(image: PIL.Image.Image,
    new_size: int = 256):

    assert isinstance(image, PIL.Image.Image)

    # pull out numpy data from image
    image_data = copy.deepcopy(np.array(image))

    if image_data.shape[0]!=image_data.shape[1]:
        image_data, bbox = segmentation_utils.crop_to_square(image_data)

    # back to PIL format
    image_preproc = PIL.Image.fromarray(image_data.astype(np.uint8))

    # resize here
    if image_preproc.height!=new_size:
        image_preproc = image_preproc.resize([new_size, new_size], resample=PIL.Image.Resampling.LANCZOS)
    else:
        image_preproc

    assert(image_preproc.height==new_size and image_preproc.width==new_size)

    return image_preproc

def preprocess_image_tosave_grayscale(image: PIL.Image.Image,
    new_size: int = 256):

    assert isinstance(image, PIL.Image.Image)

    # pull out numpy data from image
    image_data = copy.deepcopy(np.array(image))

    if image_data.shape[0]!=image_data.shape[1]:
        image_data, bbox = segmentation_utils.crop_to_square(image_data)

    # back to PIL format
    image_preproc = PIL.Image.fromarray(image_data.astype(np.uint8))

    # resize here
    if image_preproc.height!=new_size:
        image_preproc = image_preproc.resize([new_size, new_size], resample=PIL.Image.Resampling.LANCZOS)
    else:
        image_preproc

    assert(image_preproc.height==new_size and image_preproc.width==new_size)

    # converting to to grayscale
    # L = R * 299/1000 + G * 587/1000 + B * 114/1000
    # then back to RGB to get shape right
    image_preproc = image_preproc.convert('L').convert('RGB')
    
    return image_preproc

def postprocess_image(
    img: torch.Tensor, target_img: PIL.Image.Image
) -> PIL.Image.Image:
    assert img.shape[0] == 1 and img.shape[1] == 3
    assert isinstance(target_img, PIL.Image.Image)

    proc_size = img.shape[2]
    target_img_numpy = preprocess_image_simple(target_img, \
                                                 new_size=proc_size)*255
    
    # back from BGR to RGB, so it will look normal again
    img_numpy = img.numpy().squeeze().transpose(1, 2, 0)[:, :, ::-1]

    result = histogram_matching(img_numpy, target_img_numpy)
    result_pil = PIL.Image.fromarray(result.astype(np.uint8))
    
    return result_pil

def postprocess_image_multiple_targets(
    img: torch.Tensor, target_img_list: PIL.Image.Image
) -> PIL.Image.Image:
    assert img.shape[0] == 1 and img.shape[1] == 3
    assert isinstance(target_img_list[0], PIL.Image.Image)

    proc_size = img.shape[2]
    target_img_list_numpy = [preprocess_image_simple(target_img, \
                                                 new_size=proc_size)*255 \
                        for target_img in target_img_list]

    # back from BGR to RGB, so it will look normal again
    img_numpy = img.numpy().squeeze().transpose(1, 2, 0)[:, :, ::-1]
    
    # now do the histogram matching
    # here we are using the distribution across all targets together 
    # to match
    result = histogram_matching_multiple_targets(img_numpy, target_img_list_numpy)
    
    # back to PIL format finally
    result_pil = PIL.Image.fromarray(result.astype(np.uint8))
    
    return result_pil


def postprocess_image_quick(img: torch.Tensor) -> PIL.Image.Image:
    assert img.shape[0] == 1 and img.shape[1] == 3
    img_rgb = torch.flip(img, [1])
    img_norm = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min())
    to_pil = torchvision.transforms.ToPILImage()
    return to_pil(img_norm.squeeze())


def gram_matrix(activations: torch.Tensor) -> torch.Tensor:
    b, n, x, y = activations.size()
    activation_matrix = activations.view(b * n, x * y)
    G = torch.mm(activation_matrix, activation_matrix.t())    # gram product
    return G.div(b * n * x * y)     # normalization


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def inv_sigmoid(y: torch.Tensor) -> torch.Tensor:
    return -torch.log((1.0 / y) - 1.0)


def histogram_matching(source_img, target_img, n_bins=100):
    '''Taken from https://github.com/leongatys/DeepTextures'''
    assert (
        isinstance(source_img, np.ndarray) and
        isinstance(target_img, np.ndarray)
    )

    result = np.zeros_like(target_img)
    for i in range(3):
        # find the distribution of values in target image
        hist, bin_edges = np.histogram(
            target_img[:, :, i].ravel(), bins=n_bins, density=True
        )
        # convert the hist into a cumulative density function
        cum_values = np.zeros(bin_edges.shape)
        cum_values[1:] = np.cumsum(hist * np.diff(bin_edges))
        # define function to interpolate in this distribution
        inv_cdf = scipy.interpolate.interp1d(
            cum_values, bin_edges, bounds_error=True
        )
        # modify the source image values to make them uniform.
        # this is a non-linear transformation of the image that gives
        # values between 0-1 (in "r")
        r = np.asarray(uniform_hist(source_img[:, :, i].ravel()))
        r[r > cum_values.max()] = cum_values.max()
        
        # use the "r" values as input into the cdf function
        # (this is converting into target image distribution)
        result[:, :, i] = inv_cdf(r).reshape(source_img[:, :, i].shape)

    return result


def histogram_matching_multiple_targets(source_img, target_img_list, n_bins=100):
    '''Taken from https://github.com/leongatys/DeepTextures'''
    assert (
        isinstance(source_img, np.ndarray) and
        isinstance(target_img_list[0], np.ndarray)
    )

    result = np.zeros_like(target_img_list[0])
    
    for i in range(3):
        # find the distribution of values in target images
        # here we are simply concatenating values across all the targets
        targ_vals_all = np.concatenate([t[:,:,i].ravel() \
                                        for t in target_img_list], axis=0)
        hist, bin_edges = np.histogram(
            targ_vals_all, bins=n_bins, density=True
        )
        # convert the hist into a cumulative density function
        cum_values = np.zeros(bin_edges.shape)
        cum_values[1:] = np.cumsum(hist * np.diff(bin_edges))
        # define function to interpolate in this distribution
        inv_cdf = scipy.interpolate.interp1d(
            cum_values, bin_edges, bounds_error=True
        )
        # modify the source image values to make them uniform.
        # this is a non-linear transformation of the image that gives
        # values between 0-1 (in "r")
        r = np.asarray(uniform_hist(source_img[:, :, i].ravel()))
        r[r > cum_values.max()] = cum_values.max()
        
        # use the "r" values as input into the cdf function
        # (this is converting into target image distribution)
        result[:, :, i] = inv_cdf(r).reshape(source_img[:, :, i].shape)

    return result


def histogram_matching_from_saved(source_img, \
                                  r_hist, g_hist, b_hist, \
                                  r_bin_edges, g_bin_edges, b_bin_edges):
    '''Taken from https://github.com/leongatys/DeepTextures'''
    assert (
        isinstance(source_img, np.ndarray)
    )

    result = np.zeros_like(source_img)
    for i, [hist, bin_edges] in enumerate(zip([r_hist, g_hist, b_hist], \
                                             [r_bin_edges, g_bin_edges, b_bin_edges])):
        # # find the distribution of values in target image
        # hist, bin_edges = np.histogram(
        #     target_img[:, :, i].ravel(), bins=n_bins, density=True
        # )
        # convert the hist into a cumulative density function
        cum_values = np.zeros(bin_edges.shape)
        cum_values[1:] = np.cumsum(hist * np.diff(bin_edges))
        # define function to interpolate in this distribution
        inv_cdf = scipy.interpolate.interp1d(
            cum_values, bin_edges, bounds_error=True
        )
        # modify the source image values to make them uniform.
        # this is a non-linear transformation of the image that gives
        # values between 0-1 (in "r")
        r = np.asarray(uniform_hist(source_img[:, :, i].ravel()))
        r[r > cum_values.max()] = cum_values.max()
        
        # use the "r" values as input into the cdf function
        # (this is converting into target image distribution)
        result[:, :, i] = inv_cdf(r).reshape(source_img[:, :, i].shape)

    return result

def uniform_hist(X):
    '''Taken from https://github.com/leongatys/DeepTextures'''

    # this takes the values in X and transforms them into a set 
    # of values that are uniformly distributed between 0-1
    # returns array same size as X
    Z = [(x, i) for i, x in enumerate(X)]
    Z.sort()
    n = len(Z)
    Rx = [0] * n
    start = 0
    for i in range(1, n):
        if Z[i][0] != Z[i-1][0]:
            for j in range(start, i):
                Rx[Z[j][1]] = float(start + 1 + i) / 2.0
            start = i
    for j in range(start, n):
        Rx[Z[j][1]] = float(start + 1 + n) / 2.0
    return np.asarray(Rx) / float(len(Rx))

# import basic modules
import numpy as np

def crop_to_square(image_array):
 
    # crop rectangular image to a square
    # smaller side becomes the size of final square
    # taking an even amount off each end of the longer side.  
    # image array should be [height, width, ...]
    
    orig_size = image_array.shape[0:2]
    height, width = orig_size
    
    if height>width:
        
        pct_crop = ((width/height)-1)/2
        
        crop_box_pixels = [np.abs(pct_crop) * height, \
                           height - np.abs(pct_crop) * height, \
                           0, \
                           width]
        crop_box_pixels = np.floor(crop_box_pixels).astype('int')
        
        if (crop_box_pixels[1]-crop_box_pixels[0])>width:
            crop_box_pixels[0] = crop_box_pixels[0] + 1
        elif (crop_box_pixels[1]-crop_box_pixels[0])<width:
            crop_box_pixels[1] = crop_box_pixels[1] + 1
            
    else:
        pct_crop = ((height/width)-1)/2
        
        crop_box_pixels = [0, \
                           height, \
                           np.abs(pct_crop) * width, \
                           width - np.abs(pct_crop) * width]
        
        crop_box_pixels = np.floor(crop_box_pixels).astype('int')
        
        if (crop_box_pixels[3]-crop_box_pixels[2])>height:
            crop_box_pixels[2] = crop_box_pixels[2] + 1
        elif (crop_box_pixels[3]-crop_box_pixels[2])<height:
            crop_box_pixels[3] = crop_box_pixels[3] + 1

    
    cropped = image_array[crop_box_pixels[0]:crop_box_pixels[1], \
                                crop_box_pixels[2]:crop_box_pixels[3]]

    assert(cropped.shape[0]==cropped.shape[1])
    
    return cropped, crop_box_pixels


def get_crop_box_pixels(crop_box_raw, orig_size):
    
    """
    Convert cropping box from its raw format ([bottom, top, left, right]) to a bbox in pixels.
    Returns [hmin, hmax, wmin, wmax] where h is the "height" dim and w is the "width" dim.
    """
    
    orig_height, orig_width = orig_size
    
    crop_box_pixels = crop_box_raw * np.array([orig_height, orig_height, orig_width, orig_width])
    crop_box_pixels[1] = orig_height - crop_box_pixels[1]
    crop_box_pixels[3] = orig_width - crop_box_pixels[3]    
    
    return np.floor(crop_box_pixels).astype('int')


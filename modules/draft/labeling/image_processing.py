import numpy as np 


def clip_img(img):
    """Clip image pixel values into desired range. 
    
    Parameters
    ----------
        img : Numpy array 
            Original input image which is neither normalized nor clipped.
    
    Returns 
    ----------
        clipped_img : Numpy array 
            Clipped image array. 
    """ 
    min_value = img.min()
    max_value = img.max()
    
    clipped_img = np.clip(img, min_value, max_value)
    return clipped_img


def stitch_img(tiled_imgs, tile_dims):
    """Stitch image tiles into a given shape (tile_dims).  
    Note : This function will stitch tiles in X direction (so will fill out the 
        result array in row by row), so tile_imgs must be aligned in that order. 

    Parameters
    ----------
        tiled_imgs : Numpy array (T x H x W (x C), where C is number of channels)
            Group of images to stitch together. 
        tile_dims : list, tuple, or array-like with 2 inetgers in form of (X, Y)
            Number of images in both X and Y direction. 

    Returns 
    ----------
        stitched_img : Numpy array (h x w (x C), where h = H x Y and w = W x X)
            Image stitched accordingly.  
    """ 
    h, w = tiled_imgs[0].shape[:2]

    if len(tiled_imgs[0].shape) == 3:
        stitched_img = np.zeros((tile_dims[1] * h, tile_dims[0] * w, 3))
    else:
        stitched_img = np.zeros((tile_dims[1] * h, tile_dims[0] * w))
        
    for i in range(len(tiled_imgs)):
        # compute coordinates in the result image 
        x = (i % tile_dims[0]) * w
        y = (i // tile_dims[0]) * h 
        stitched_img[y:y+h, x:x+w] = tiled_imgs[i]
    
    if np.max(stitched_img) <= 1.:
        stitched_img = stitched_img.astype(np.float32) 
    else:
        stitched_img = stitched_img.astype(np.int32)
        
    return stitched_img
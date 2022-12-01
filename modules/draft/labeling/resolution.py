import numpy as np 

from .label import label_image
from .image_processing import stitch_img 


def get_high_res_image(dz, address, level, limit, do_label=False, clustering_algo=None):
    """Recursively get the higher resolution image. 
    
    Parameters
    ----------
        dz : DeepZoom  
            The DeepZoom instance. 
        address : ndarray of shape (2,), dtype=float 
            Address where the patch resides at the initial level. 
        limit : int 
            Number of levels to zoom in. If 1, the function returns a 1 level higher resolution image. 
        do_label : bool, default=False 
            Decide whether to label images simultaneously or not. 
        clustering_algo : class instance, default=None 
            Clustering algorithm instance which contains predict() method and cluster_centers_ member. 
    
    Returns 
    ----------
        stitched_img : ndarray of shape (H, W, 3), dtype=float 
            High resolution image which has been stitched. 
        stitched_label : {ndarray of shape (H, W), None}, dtype={int, None}
            Label corresponding to stitched_img. 
    """
    if do_label is True and clustering_algo is None:
        raise ValueError("You must pass a clustering algorithm instance in order to label the image.")

    # if limit became 0 or level reaches to the limit of deepzoom, return the result 
    # Note : If limit is 0 initially, then we take single image at an initial level 
    if limit == 0 or level == dz.dzg.level_count - 1:
        img, _ = dz.get_tile(level = level, address=address)
        label = None 
        
        if do_label is True:
            label = label_image(img, clustering_algo)

        return (img, label)
    
    # update both level and limit 
    limit -= 1
    level += 1
    
    ul_img, ul_label = get_high_res_image(dz, (address * 2), level, limit, do_label=do_label, clustering_algo=clustering_algo)
    ur_img, ur_label = get_high_res_image(dz, (address * 2) + np.array([1, 0]), level, limit, do_label=do_label, clustering_algo=clustering_algo)
    bl_img, bl_label = get_high_res_image(dz, (address * 2) + np.array([0, 1]), level, limit, do_label=do_label, clustering_algo=clustering_algo)
    br_img, br_label = get_high_res_image(dz, (address * 2) + np.array([1, 1]), level, limit, do_label=do_label, clustering_algo=clustering_algo)
    
    stitched_img = stitch_img([ul_img, ur_img, bl_img, br_img], (2, 2))

    stitched_label = None 
    if do_label is True:
        stitched_label = stitch_img([ul_label, ur_label, bl_label, br_label], (2, 2))
    
    return (stitched_img, stitched_label)
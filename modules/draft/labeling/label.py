import numpy as np 
import pydensecrf.densecrf as dcrf 
import pydensecrf.utils as dutils 
from scipy.sparse import csgraph 
from skimage import img_as_ubyte, morphology 
from skimage.color import label2rgb 
from skimage.feature import multiscale_basic_features
from skimage.transform import resize 
from sklearn.neighbors import radius_neighbors_graph

from .image_processing import clip_img 


def _extract_features_multi(img, sigma_min, sigma_max, multi_channel=True, reshape=True):
    """Extract multiple meta features in image at pixel level. 
    
    Parameters
    ----------
        img : ndarray of shape (height, width, channels) 
            Array which represents an image. 
        sigma_min : float 
            Smallest value of sigma in Gaussian kernel. 
        sigma_max : float
            Largest value of sigma in Gaussian kernel.
        multi_channel : bool True 
            Determine if an image is multi-channel image or not.
        reshape : bool True 
            
    Returns 
    ----------
        X : Numpy array 
            Feature vector extracted by multiscale_basic_features. 
    """
    if type(img) is not np.ndarray:
        raise TypeError(f"Input image must be a numpy array format, instead of {type(img)}")

    h, w = img.shape[:2]
    # compute multiple features per pixel 
    X = multiscale_basic_features(
        img,
        intensity=True,
        edges=True,
        texture=True,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        multichannel=multi_channel 
    )
    
    if reshape: 
        X = X.reshape(h * w, -1)
        
    return X


def dcrf2d(img_label, img_rgb, gt_prob=0.7, n_cluster=None, zero_unsure=False, stdxy_g=1, 
            compat_g=10, compat_b=5, stdxy_b=10, stdrgb=5, niter=5, bg_label=-1):
    """Apply DenseCRF to an image given. 
    
    Parameters
    ----------
        img_label : Numpy array (H x W)
            Array which is a pixel level label, corresponding to an image. 
        img_rgb : Numpy array (H x W x 3)
            Array which represents 3 channel RGB image.  
        gt_prob : float ranged in [0, 1], default = 0.7
            Probability of ground truth labels.  
        n_cluster : int, default = None 
            Number of clusters. Only specify this if maximum number of classes is known. 
        
    
    Returns 
    ----------
        MAP : Numpy array (H x W)
            Refined label image as a result of DenseCRF.  
        overlay : Numpy array (H x W x 3)
            Result label (MAP) overlayed image. 
    """
    h, w = img_label.shape[:2]
    # if overall numer of clusters are not defined, then we assign unique labels within labels 
    if n_cluster is None:
        n_cluster = np.unique(img_label).shape[0]
    
    d = dcrf.DenseCRF2D(w, h, n_cluster)

    # get unary potentials (neg log probability)
    U = dutils.unary_from_labels(img_label, n_cluster, gt_prob=gt_prob, zero_unsure=zero_unsure)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(stdxy_g, stdxy_g), compat=compat_g, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(stdxy_b, stdxy_b), srgb=(stdrgb, stdrgb, stdrgb), rgbim=img_as_ubyte(img_rgb),
                           compat=5,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    # Run n inference steps.
    Q = d.inference(niter)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0).reshape(h, w)
    overlay = label2rgb(MAP, img_rgb, bg_label=bg_label)

    return (MAP, overlay)


def group_classes_by_radius(X, radius=0.05, metric="euclidean"):
    """Group the classes based on feature positions. 
    
    Parameters
    ----------
        X : Numpy array (classes x features)
            Feature vectors to group up. 
        radius : float 
            Threshold for grouping up the classes or not. 
        metric : str, refer to the list on [this page](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html)
            Distance metric. 

    Returns 
    ----------
        n_components : int 
            Number of connected components detected. 
        group : dict 
            key-value pair where key represents unique classes, and value reparesenting minimum class index in the same group (CC). 
    """
    if X.shape[0] > 1000:
        raise ValueError(f"Too much classes to handle ({X.shape[0]}). It must be less than 1000.")
    
    adj = radius_neighbors_graph(X, radius=radius, metric=metric)
    n_components, labels = csgraph.connected_components(adj, directed=False)
    group = dict(zip(np.arange(X.shape[0]), labels))
    return (n_components, group) 


def label_image(img, clustering_algo, cluster_radius=0.05, sigma_min=1, sigma_max=2, clip=True):
    """Generate label for a given image."""
    _, group_dict = group_classes_by_radius(clustering_algo.cluster_centers_, radius=cluster_radius)
    
    # if we need to, we clip the image values 
    if clip:
        img = clip_img(img)

    h, w = img.shape[:2]
    
    # retrieve feature from image 
    feature = _extract_features_multi(img, sigma_min, sigma_max, multi_channel=True, reshape=True) 
    km_label = clustering_algo.predict(feature).reshape(h, w)
    dcrf_label, _ = dcrf2d(km_label, img, n_cluster=20, stdxy_g=1, compat_g=10, 
                    stdxy_b=10, compat_b=5, stdrgb=10, niter=50)
    
    # group up the class labels 
    u, inverse = np.unique(dcrf_label, return_inverse=True) 
    grouped_label = np.array([group_dict[x] for x in u])[inverse].reshape(dcrf_label.shape) 
    
    return grouped_label.astype(int) 


def get_mask(label, n_classes=None): 
    """Returns both bool and integer masks of a label image""" 
    unique_labels = np.unique(label)
    if n_classes is None: 
        n_classes = len(unique_labels)

    bool_masks = np.zeros((n_classes, label.shape[0], label.shape[1]), dtype=bool)
    int_masks = np.zeros((n_classes, label.shape[0], label.shape[1]), dtype=np.int32)
    
    for i in unique_labels:
        mask = (label == i)
        bool_masks[i] = mask 
        int_masks[i] = mask.astype(np.int32)
    
    return (bool_masks, int_masks)


def _get_contours(mask, erode_width=1):
    """Returns edge mask. 
    """
    copied_mask = mask.copy()
    for i in range(erode_width):
        copied_mask = morphology.erosion(copied_mask)
    contours = np.logical_xor(mask, copied_mask)
    return contours 
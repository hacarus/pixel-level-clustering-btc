"""shannon entropy with CUDA."""
import cupy as cp
import numpy as np


PATTERN_PLACE_HOLDER = "<PLACE_HOLDER>"
in_params = """
    raw uint8 input,
    int16 width,
    int16 height,
    int16 kernel_size
"""
out_params = "float32 output"
preamble = """
    __device__ int get_x_idx(int i, int width) {
        return i % width;
    }
    
    __device__ int get_y_idx(int i, int width) {
        return i / width;
    }
    
    __device__ float calc_normalized_hist(
        int x_idx, int y_idx,
        int width, int height,
        int kernel_size,
        const CArray<unsigned char, 2, false, true> &input) {
        const int nbins=256;
        float bincount[nbins];
        float entropy=0;
        int sum = 0;
        
        for (int i=0; i < nbins; i++){
            bincount[i]=1e-16;   // to avoid error
        }
        
        // calc hist
        for (int dy = -kernel_size; dy <= kernel_size; dy++) {
            for (int dx = -kernel_size; dx <= kernel_size; dx++) {
                int x = x_idx + dx;
                int y = y_idx + dy;
                if (x < 0 || x >= width || y < 0 || y >= height) {
                    continue;
                }
                int idx[] = {y, x};
                bincount[input[idx]]++;
            }
        }
        
        __syncthreads();
        // summation
        for (int i=0; i < nbins; i++){
            sum += bincount[i];
        }
        
        __syncthreads();
        // calc entropy
        for (int i=0; i < nbins; i++){
            entropy -= (bincount[i] / sum) * (__log2f(bincount[i]) - __log2f(sum));
        }
        
        return entropy;
    }
"""
operation="""
    int x_idx = get_x_idx(i, width);
    int y_idx = get_y_idx(i, width);
    output = calc_normalized_hist(x_idx, y_idx, width, height, kernel_size, input);
"""

# generate kernel function on GPU.
get_shannon_entropy = cp.ElementwiseKernel(
    in_params=in_params,
    out_params=out_params,
    preamble=preamble,
    operation=operation,
    name='get_shannon_entropy')


def _elementwise_shannon_entropy(img: np.ndarray, kernel_size: int = 10, multichannel: bool = False):
    """Elementwise shannon entropy.

    Parameters
    ----------
        img: (N, M [, C]) ndarray,
            single or multichannnel image.
        kernel_size: int, default 10,
            window size to calculate pixelwise shannon enrtopy.
        multichannel: bool, default False,
            Given image has multichannel or not.
    
    Returns
    -------
        img_entropy: (N, M) ndarray,
            image of shannon entropy.
    """
    if multichannel and img.ndim == 2:
        raise ValueError("Set multichannel=Flase for two-dimensional image.")
    if not multichannel and img.ndim == 3:
        raise ValueError("Set multichannel=True for three-dimensional image.")
    if img.ndim < 2:
        raise ValueError("Wrong img shape. two or three dimensional image is only available.")

    img_entropy = cp.zeros_like(img, dtype=cp.float32)
    if not multichannel:
        get_shannon_entropy(img, img.shape[1], img.shape[0], kernel_size, img_entropy)
    if multichannel:
        img_entr_list = []
        for ch in range(img.shape[2]):
            _img = img[..., ch]
            _img_entropy = cp.zeros(_img.shape[:2], dtype=cp.float32)
            get_shannon_entropy(_img, _img.shape[1], _img.shape[0], kernel_size, _img_entropy)
            img_entr_list.append(_img_entropy)
        img_entropy = np.stack(img_entr_list, axis=2)
    return img_entropy


def elementwise_shannon_entropy(img: np.ndarray, kernel_size: int = 10, multichannel: bool = False):
    return _elementwise_shannon_entropy(cp.array(img), kernel_size, multichannel).get()

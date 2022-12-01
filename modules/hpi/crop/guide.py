"""guide.py."""
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from hpi.dataloader import DeepZoom, NDP


def _show_guide(dz: DeepZoom, level: int, title: Optional[str] = None):
    """Show address on WSI."""
    wsi_shape = np.array(dz.dzg.level_dimensions[level])
    shift = dz.tile_size - dz.overlap

    ndp = NDP(dz.ndpi, dz.ndpa)
    ndp_level = dz.dzg.level_count - level - 1
    thumnail = ndp.get_image(level=ndp_level)
    if dz.ndpa is not None:
        masks = ndp.get_mask(level=ndp_level)
        mask = np.array(list(masks.values())).sum(axis=0).astype(float)
        mask = mask / mask.max()
        mask[mask == 0] = np.nan
    else:
        mask = np.zeros(shape=thumnail.shape[:2], dtype=float)
        mask[:] = np.nan
    
    aspect = wsi_shape[1] / wsi_shape[0]
    if aspect > 1:
        fig, ax = plt.subplots(figsize=(10, 10 * aspect))
    else:
        fig, ax = plt.subplots(figsize=(6 / aspect, 6))

    ax.imshow(thumnail)
    ax.imshow(mask, alpha=0.5, cmap="gnuplot_r")
    ax.set_title(title)
    
    xticks = np.arange(0, wsi_shape[0] + 1, shift)
    yticks = np.arange(0, wsi_shape[1] + 1, shift)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(np.arange(xticks.shape[0]))
    ax.set_yticklabels(np.arange(yticks.shape[0]))

    ax.set_ylabel("address[1]", fontsize=16)
    ax.set_xlabel("address[0]", fontsize=16)
    ax.grid()
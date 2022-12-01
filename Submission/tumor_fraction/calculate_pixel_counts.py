from typing import Dict
from pathlib import Path
import numpy as np
import pandas as pd
from omegaconf import DictConfig
import hydra

from eda import git_root
from logger import setup_logger


script_path = Path(__file__)
log_dir = script_path.parent / "log"
logger = setup_logger(script_path.name, log_dir / (script_path.stem + ".log"))


def sort(x: Path) -> int:
    """Sort."""
    return int(x.stem.split("_")[-1])


def get_cluster_fraction(img: np.ndarray, norm: bool = False) -> Dict[int, int]:
    """Get fraction of each cluster.

    Parameters
    ----------
    img: np.ndarray
        labeled image.
    norm: bool
        do normalization or not.

    Returns
    -------
    Dict[cluster_id, pixel count]
    """
    mask = img > 0
    ulabels, counts = np.unique(img[mask], return_counts=True)
    if norm:
        counts = counts / counts.sum()
    return dict(zip(ulabels, counts))


@hydra.main(config_name="config", config_path="config",
            version_base=None)
def main(cfg: DictConfig):
    """Main process."""
    ROOT = git_root(absolute=True)
    input_dir_raw = ROOT / cfg.data.INPUT_DIR_RAW
    input_dir_cluster = ROOT / cfg.data.INPUT_DIR_CLUSTER
    output_dir = ROOT / cfg.data.OUTPUT_DIR
    output_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

    paths_cluster = sorted(
        list(input_dir_cluster.glob("*klabels*.npy")), key=sort
    )
    paths_mask = sorted(list(input_dir_raw.glob("*mask*.npy")), key=sort)

    # calculate pixel-counts list
    logger.info("calculate pixel-counts")
    pixel_counts = []
    # loop over tissue.
    for i, (path, path_annot) in enumerate(zip(paths_cluster, paths_mask)):
        img = np.load(path, allow_pickle=True)
        mask = np.load(path_annot, allow_pickle=True).item()
        mask_normal = mask["liver"]
        mask_abnormal = mask["tumor"]

        # extract
        img_normal = img * mask_normal
        img_abnormal = img * mask_abnormal

        for _img, status in zip([img_normal, img_abnormal],
                                ["non-cancerous", "cancerous"]):
            counts = get_cluster_fraction(_img, norm=False)
            counts["tissue"] = i
            counts["status"] = status
            pixel_counts.append(counts)

    # list -> dataframe, drop noise
    logger.info("drop noise clusters")
    df_pixel_counts = pd.DataFrame(pixel_counts)
    df_pixel_counts = df_pixel_counts.fillna(0)
    df_pixel_counts = df_pixel_counts.drop(cfg.cluster.ignore, axis=1)

    logger.info(f"save: {output_dir}/pixel_counts.csv")
    df_pixel_counts.to_csv(output_dir / "pixel_counts.csv")


if __name__ == "__main__":
    main()

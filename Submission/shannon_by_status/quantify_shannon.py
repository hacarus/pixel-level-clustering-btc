from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import cv2
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


def get_unique_labels(img_label: np.ndarray) -> NDArray[np.integer]:
    """Get unique labels from a labeled image."""
    mask = img_label > 0
    ulabels = np.unique(img_label[mask])
    return ulabels


def extract(img_label: np.ndarray, img_intensity: np.ndarray) -> pd.DataFrame:
    """Extract intensity from img_intensity."""
    img_label = cv2.resize(img_label, img_intensity.shape[:2][::-1], None,
                           None, None, interpolation=cv2.INTER_NEAREST)
    ulabels = get_unique_labels(img_label)
    dfs = []
    for ulabel in ulabels:
        df = pd.DataFrame(img_intensity[img_label == ulabel])
        df = df.assign(cluster=ulabel)
        dfs.append(df)
    return pd.concat(dfs)


@hydra.main(config_name="config", config_path="config",
            version_base=None)
def main(cfg: DictConfig):
    """Main process."""
    ROOT = git_root(absolute=True)
    input_dir_cluster = ROOT / cfg.models.data.INPUT_DIR_CLUSTER
    input_dir_shannon = ROOT / cfg.models.data.INPUT_DIR_SHANNON
    output_dir = ROOT / cfg.models.data.OUTPUT_DIR
    output_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

    paths_cluster = sorted(
        list(input_dir_cluster.glob("*klabels*.npy")), key=sort
    )
    paths_shannon = sorted(
        list(input_dir_shannon.glob("*shannon*.npy")), key=sort
    )

    # calculate pixel-counts list
    logger.info("quantify shannon entropy")
    for i, (path_cluster, path_shannon) in enumerate(zip(paths_cluster,
                                                         paths_shannon)):
        logger.debug(f"load: {path_cluster}")
        img_cluster = np.load(path_cluster, allow_pickle=True)
        logger.debug(f"load: {path_shannon}")
        img_shannon = np.load(path_shannon, allow_pickle=True)
        logger.debug(
            f"resize: {img_shannon.shape} -> {img_cluster.shape[::-1]}"
        )
        img_shannon = cv2.resize(img_shannon.astype(np.float32),
                                 img_cluster.shape[::-1], None,
                                 None, None, interpolation=cv2.INTER_CUBIC)

        logger.debug("extract")
        df_shannon = extract(img_cluster, img_shannon)
        logger.debug(f"save: {output_dir}/shannon_by_cluster_{i}.csv")
        df_shannon.to_csv(output_dir / f"shannon_by_cluster_{i}.csv")


if __name__ == "__main__":
    main()

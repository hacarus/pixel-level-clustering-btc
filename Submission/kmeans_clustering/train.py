from typing import Sequence, Dict, Tuple
from pathlib import Path
import copy
import pickle
import numpy as np
from numpy.typing import NDArray
from skimage.feature import multiscale_basic_features
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import cv2
from tqdm import tqdm
from omegaconf import DictConfig
import hydra


from eda import git_root
from hpi.tissue import mask_tissue
from logger import setup_logger


script_path = Path(__file__)
log_dir = script_path.parent / "log"
logger = setup_logger(script_path.name, log_dir / (script_path.stem + ".log"))


def load_model(path: Path):
    """Load clustering model."""
    if path.exists():
        with open(path, mode="rb") as f:
            cpipeline = pickle.load(f)
    else:
        raise FileNotFoundError
    return cpipeline


def extract_mbfs(paths_image: Sequence[Path], scale: float,
                 params_mbf: Dict) -> Tuple[NDArray[np.floating], ...]:
    """Extract MBF from each image."""
    Xs = []
    sizes = []
    logger.info("Extract MBF from all the images.")
    for path in tqdm(paths_image):
        # load image
        logger.debug(f"load {path}")
        img = np.load(path, allow_pickle=True)[..., :3]
        mask = mask_tissue(img, largest=True)

        # rescale images
        logger.debug("rescale")
        shape = np.int16(scale * np.array(img.shape[:2]))
        img = cv2.resize(img, shape[::-1], interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask.astype(np.uint8), shape[::-1],
                          interpolation=cv2.INTER_NEAREST) > 0
        logger.debug("extract MBF.")
        Xs.append(multiscale_basic_features(img, **params_mbf)[mask])
        area = Xs[-1].shape[0]
        sizes.append(np.ones(area) * area)
    return np.vstack(Xs), np.hstack(sizes)


@hydra.main(config_name="config", config_path="config",
            version_base=None)
def main(cfg: DictConfig):
    """Main process."""
    if not cfg.models.train:
        raise ValueError(
            f"invalid mode: cfg.models.train={cfg.modes.train}"
        )

    ROOT = git_root(absolute=True)
    np.random.seed(cfg.common.SEED)
    input_dir = ROOT / cfg.models.data.INPUT_DIR
    paths_image = sorted(list(input_dir.glob("cropped*")))

    for sigma_min, sigma_max in cfg.mbf_experiments.sigma_minmax_pairs:
        logger.info(f"(sigma_min, sigma_max)={sigma_min}, {sigma_max}")
        # make output directory
        output_dir = (
            ROOT / cfg.models.data.OUTPUT_DIR / f"{sigma_min}_{sigma_max}"
        )
        output_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

        # apply MBF
        _params_mbf = copy.copy(cfg.mbf)
        _params_mbf.update(dict(sigma_min=sigma_min, sigma_max=sigma_max))
        X, _ = extract_mbfs(paths_image, cfg.common.SCALE, _params_mbf)

        # train clustering.
        steps_clustering = [
            ("standardize", StandardScaler()),
            ("clustering", KMeans(
                n_clusters=cfg.models.clustering.N_CLUSTERS, n_init=1,
                random_state=cfg.common.SEED)),
        ]

        n_samples = cfg.models.clustering.N_SAMPLE
        logger.info(
            f"downsample pixels into {n_samples}."
        )
        inds = np.random.choice(range(X.shape[0]), size=n_samples)
        cpipeline = Pipeline(steps_clustering, verbose=False)

        # down sampling and fitting.
        logger.info("fit.")
        cpipeline.fit(X[inds])

        # save model
        path_model = output_dir / "pipe_kmeans.pkl"
        logger.info(f"save model to {path_model}.")
        with open(path_model, mode="wb") as f:
            pickle.dump(cpipeline, f)


if __name__ == "__main__":
    main()

from typing import Sequence, Dict
from pathlib import Path
import copy
import pickle
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from skimage.feature import multiscale_basic_features
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
                 params_mbf: Dict) -> NDArray[np.floating]:
    """Extract MBF from each image."""
    Xs = []
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
    X = np.vstack(Xs)
    return X


def predict_clusters(
    pipe_kmeans: Pipeline,
    img: np.ndarray,
    mask: np.ndarray,
    params_mbf: Dict[str, float],
) -> NDArray[np.integer]:
    """Predict clusters."""
    # initialize variable.
    klabels = np.zeros_like(mask, dtype=np.int32, order="C")

    # KMeans and DenseCRF
    X = multiscale_basic_features(img, **params_mbf)[mask]
    klabels[mask] = pipe_kmeans.predict(X) + 1   # 0 indicates background
    return klabels


def predict(cfg: DictConfig) -> None:
    """Predict."""
    ROOT = git_root(absolute=True)
    input_dir = ROOT / cfg.models.data.INPUT_DIR
    paths_image = sorted(list(input_dir.glob("cropped*")))

    for sigma_min, sigma_max in cfg.mbf_experiments.sigma_minmax_pairs:
        logger.info(f"(sigma_min, sigma_max)={sigma_min}, {sigma_max}")
        model_dir = (
            ROOT / cfg.models.data.MODEL_DIR / f"{sigma_min}_{sigma_max}"
        )
        output_dir = (
            ROOT / cfg.models.data.OUTPUT_DIR / f"{sigma_min}_{sigma_max}"
        )

        # make output directory
        output_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

        # path to clustering model.
        path_model = model_dir / "pipe_kmeans.pkl"
        cpipeline = load_model(path_model)

        # apply MBF
        _params_mbf = copy.copy(cfg.mbf)
        _params_mbf.update(dict(sigma_min=sigma_min, sigma_max=sigma_max))

        for i in range(len(paths_image)):
            # load image
            path = ROOT / cfg.models.data.INPUT_DIR / f"cropped_{i}.npy"
            img = np.load(path, allow_pickle=True)[..., :3]
            print(f"processing tissue {path.name}...")

            mask = mask_tissue(img, largest=True)
            shape_org = mask.shape

            # rescale images
            shape = np.int16(cfg.common.SCALE * np.array(img.shape[:2]))
            img = cv2.resize(img, shape[::-1], interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask.astype(np.uint8), shape[::-1],
                              interpolation=cv2.INTER_NEAREST) > 0

            klabels = predict_clusters(cpipeline, img, mask, _params_mbf)
            print(img.shape, mask.shape, klabels.shape)

            # resize to original shape.
            klabels = cv2.resize(klabels, shape_org[::-1],
                                 interpolation=cv2.INTER_NEAREST)

            # save
            np.save(output_dir / f"klabels_{i}.npy", klabels)

            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            ax.imshow(klabels, cmap="tab20")
            ax.set_title("kmeans")
            fig.savefig(output_dir / f"tissue_{i}.png")
            plt.close("all")


@hydra.main(config_name="config", config_path="config", version_base=None)
def main(cfg: DictConfig):
    """Main process."""
    if cfg.models.train:
        raise ValueError(
            f"invalid mode: cfg.models.train={cfg.modes.train}"
        )

    ROOT = git_root(absolute=True)
    np.random.seed(cfg.common.SEED)
    input_dir = ROOT / cfg.models.data.INPUT_DIR
    paths_image = sorted(list(input_dir.glob("cropped*")))

    for sigma_min, sigma_max in cfg.mbf_experiments.sigma_minmax_pairs:
        logger.info(f"(sigma_min, sigma_max)={sigma_min}, {sigma_max}")
        model_dir = (
            ROOT / cfg.models.data.MODEL_DIR / f"{sigma_min}_{sigma_max}"
        )
        output_dir = (
            ROOT / cfg.models.data.OUTPUT_DIR / f"{sigma_min}_{sigma_max}"
        )

        # make output directory
        output_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

        # path to clustering model.
        path_model = model_dir / "pipe_kmeans.pkl"
        logger.info(f"load model: {path_model}")
        cpipeline = load_model(path_model)

        # apply MBF
        _params_mbf = copy.copy(cfg.mbf)
        _params_mbf.update(dict(sigma_min=sigma_min, sigma_max=sigma_max))

        for i in range(len(paths_image)):
            # load image
            path = ROOT / cfg.models.data.INPUT_DIR / f"cropped_{i}.npy"
            logger.debug(f"load: {path}")
            img = np.load(path, allow_pickle=True)[..., :3]

            logger.debug("Get mask")
            mask = mask_tissue(img, largest=True)
            shape_org = mask.shape

            # rescale images
            logger.debug("rescale")
            shape = np.int16(cfg.common.SCALE * np.array(img.shape[:2]))
            img = cv2.resize(img, shape[::-1], interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask.astype(np.uint8), shape[::-1],
                              interpolation=cv2.INTER_NEAREST) > 0

            logger.debug("predict clusters")
            klabels = predict_clusters(cpipeline, img, mask, _params_mbf)

            # resize to original shape.
            logger.debug("rescale")
            klabels = cv2.resize(klabels, shape_org[::-1],
                                 interpolation=cv2.INTER_NEAREST)

            # save
            logger.debug("save predicted clusters.")
            np.save(output_dir / f"klabels_{i}.npy", klabels)

            logger.debug("save plot")
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            ax.imshow(klabels, cmap="tab20")
            ax.set_title("kmeans")
            fig.savefig(output_dir / f"tissue_{i}.png")
            plt.close("all")


if __name__ == "__main__":
    main()

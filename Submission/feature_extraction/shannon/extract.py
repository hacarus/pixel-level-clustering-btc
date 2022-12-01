from pathlib import Path
import numpy as np

from omegaconf import DictConfig
import hydra

from eda import git_root
from eda.feature_analysis._entropy import shannon_entropy_recursive
from logger import setup_logger


script_path = Path(__file__)
log_dir = script_path.parent / "log"
logger = setup_logger(script_path.name, log_dir / (script_path.stem + ".log"))


@hydra.main(config_name="config", config_path="config", version_base=None)
def main(cfg: DictConfig):
    """Main process."""
    ROOT = git_root(absolute=True)
    input_dir = ROOT / cfg.species.data.INPUT_DIR
    output_dir = (
        ROOT / cfg.species.data.OUTPUT_DIR / f"kernel_{cfg.shannon.kernel_size}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    npaths = len(list(input_dir.glob("cropped_*")))
    for i in range(npaths):
        input_path = input_dir / f"cropped_{i}.npy"

        logger.info(f"load: {input_path}")
        img = np.load(input_path, allow_pickle=True)[..., :3]
        img_se = shannon_entropy_recursive(img, **cfg.shannon)

        # save results
        output_path = output_dir / f"shannon_{i}.npy"
        logger.info(f"save: {output_path}")
        np.save(output_path, img_se)


if __name__ == "__main__":
    main()

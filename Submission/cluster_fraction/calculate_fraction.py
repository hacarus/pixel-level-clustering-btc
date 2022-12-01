from typing import Dict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig
import hydra

from eda import git_root
from hpi.uicc import extract_pM, extract_pN, extract_pT, classify_stage
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


def add_uicc_classification(df):
    """Add UICC classification to the dataframe."""
    df = (df.assign(T=df["pTNM"].apply(lambda x: extract_pT(x)))
          .assign(N=df["pTNM"].apply(lambda x: extract_pN(x)))
          .assign(M=df["pTNM"].apply(lambda x: extract_pM(x))))
    # stages
    stages = [classify_stage(*(df.iloc[i].loc["T":"M"]))
              for i in range(df.shape[0])]
    df = df.assign(stage=stages)
    return df


@hydra.main(config_name="config", config_path="config",
            version_base=None)
def main(cfg: DictConfig):
    """Main process."""
    ROOT = git_root(absolute=True)
    input_dir = ROOT / cfg.data.INPUT_DIR
    output_dir = ROOT / cfg.data.OUTPUT_DIR
    output_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
    path_meta = ROOT / cfg.data.META

    # get clinical information
    df_meta = pd.read_csv(path_meta, index_col=0)
    df_meta["sex"].replace("F", "Female", inplace=True)
    df_meta["sex"].replace("M", "Male", inplace=True)
    # each patient has two samples.
    # _df_meta = df_meta.drop_duplicates(["TMA_ID"])
    # _df_meta = add_uicc_classification(_df_meta)

    # NOTE: each patient has two samples.
    _df_meta_undrop = add_uicc_classification(df_meta)
    _df_meta_undrop = _df_meta_undrop.ffill()

    paths_human_cluster = sorted(list(input_dir.glob("*klabels*.npy")),
                                 key=sort)

    # calculate pixel-counts list
    logger.info("calculate pixel-counts")
    pixel_counts = []
    for path in paths_human_cluster:
        img = np.load(path)
        pixel_counts.append(get_cluster_fraction(img, norm=False))

    # list -> dataframe, drop noise
    logger.info("drop unknown label")
    df_pixel_counts = pd.DataFrame(pixel_counts)
    df_pixel_counts = df_pixel_counts.assign(stage=_df_meta_undrop["stage"])
    df_pixel_counts = df_pixel_counts.fillna(0)
    df_pixel_counts = df_pixel_counts.drop(cfg.cluster.ignore, axis=1)
    df_pixel_counts = (df_pixel_counts
                       .query('not stage == "Unknown"')
                       .drop('stage', axis=1))

    # pixel count to ratio
    logger.info("calculate fraction")
    all_area = df_pixel_counts.sum().sum()
    df_fraction = pd.DataFrame({
        'cluster': df_pixel_counts.columns,
        'area': df_pixel_counts.sum(),
        'fraction': df_pixel_counts.sum() / all_area
    })
    df_fraction = df_fraction.sort_index()

    logger.info(f"save: {output_dir}/fraction.csv")
    df_fraction.to_csv(output_dir / "fraction.csv")

    logger.info(f"plot: {output_dir}/fraction.png")

    df_fraction = df_fraction.sort_values("fraction")
    order = df_fraction["cluster"]
    sns.barplot(data=df_fraction, x="cluster", y="fraction", order=order,
                color="gray")
    fig = plt.gcf()
    fig.savefig(output_dir / "fraction.png")
    plt.close("all")


if __name__ == "__main__":
    main()

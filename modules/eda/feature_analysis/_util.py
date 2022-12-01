"""util.py."""


def _check_size_overlap(size: int, overlap: float):
    """Check size."""
    if size * overlap % 1 != 0:
        return True
    return False


def _check_overlap(overlap: float):
    """Check overlap."""
    if 0 < overlap < 1:
        return False
    return True


def _check_tilesize(tile_size: int, img_size: int):
    if img_size % tile_size > 0:
        return True
    return False

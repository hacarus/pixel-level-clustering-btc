"""types.py"""


def isint(val: str):
    """String represents int or not."""
    try:
        int(val)
        return True
    except (ValueError, TypeError):
        pass
    return False


def isfloat(val: str):
    """String represents float or not."""
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        pass
    return False


def cast(val: str):
    """Cast type of a string to a number."""
    if isint(val):
        return int(val)
    if isfloat(val):
        return float(val)
    return val

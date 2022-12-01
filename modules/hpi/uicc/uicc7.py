"""Functions to clean data."""
import re


def extract_pT(x: str) -> str:
    """Extract pT."""
    m = re.match(r"(.+|^)(T[\d]|Tx)", x)
    if m:
        return m.group(2)
    return "Unknown"


def extract_pN(x: str) -> str:
    """Extract pN."""
    m = re.match(r"(.+|^)(N[\d]|Nx)", x)
    if m:
        return m.group(2)
    return "Unknown"


def extract_pM(x: str) -> str:
    """Extract pM."""
    m = re.match(r"(.+|^)(M[\d]|Mx)", x)
    if m:
        return m.group(2)
    return "Unknown"


def classify_stage(T: str, N: str, M: str) -> str:
    """UICC 7th edition.

    Intrahepatic bile duct cancer.
    """
    if M == "M1":
        return "IV-B"
    elif N == "N1":
        return "IV-A"
    elif T == "T1":
        return "I"
    elif T == "T2":
        return "II"
    elif T == "T3":
        return "III"
    elif T == "T4":
        return "IV-A"
    return "Unknown"

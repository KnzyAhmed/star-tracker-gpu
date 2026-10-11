"""Load the bright-star catalog as unit vectors.

    cat = load_catalog()          # stars brighter than magnitude 5.5
    cat.r      (N, 3) unit vectors in the sky frame
    cat.mag    (N,)   visual magnitude (smaller = brighter)
    cat.hr     (N,)   catalog number (Harvard Revised), a unique star ID
    cat.name   (N,)   name/designation string, may be empty
"""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tracker.camera import radec_to_unit

CATALOG_CSV = Path(__file__).resolve().parent.parent / "data" / "bsc5.csv"


@dataclass
class Catalog:
    r: np.ndarray
    mag: np.ndarray
    hr: np.ndarray
    name: np.ndarray

    def __len__(self):
        return len(self.mag)


def load_catalog(mag_limit=5.5, path=CATALOG_CSV):
    hr, name, ra, dec, mag = [], [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            m = float(row["vmag"])
            if m >= mag_limit:
                continue
            hr.append(int(row["hr"]))
            name.append(row["name"])
            ra.append(float(row["ra_deg"]))
            dec.append(float(row["dec_deg"]))
            mag.append(m)
    return Catalog(
        r=radec_to_unit(np.array(ra), np.array(dec)),
        mag=np.array(mag),
        hr=np.array(hr),
        name=np.array(name, dtype=object),
    )

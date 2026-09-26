"""Download the Yale Bright Star Catalog (VizieR V/50) and save a clean CSV.

Output columns: hr, name, ra_deg, dec_deg, vmag   (J2000, degrees)
Run once from the repo root:  python -m tracker.fetch_catalog
"""
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

OUT = Path(__file__).resolve().parent.parent / "data" / "bsc5.csv"


def to_degrees(ra_col, dec_col):
    """VizieR may return sexagesimal strings ('06 45 08.9') or floats in degrees."""
    if np.issubdtype(np.asarray(ra_col).dtype, np.number):
        return np.asarray(ra_col, float), np.asarray(dec_col, float)
    c = SkyCoord(list(ra_col), list(dec_col), unit=(u.hourangle, u.deg))
    return c.ra.deg, c.dec.deg


def main():
    v = Vizier(columns=["HR", "Name", "RAJ2000", "DEJ2000", "Vmag"], row_limit=-1)
    t = v.get_catalogs("V/50/catalog")[0]
    print(f"downloaded {len(t)} rows")

    ok = ~(np.ma.getmaskarray(t["RAJ2000"]) | np.ma.getmaskarray(t["DEJ2000"])
           | np.ma.getmaskarray(t["Vmag"]))
    t = t[ok]
    ra, dec = to_degrees(t["RAJ2000"], t["DEJ2000"])

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        f.write("hr,name,ra_deg,dec_deg,vmag\n")
        for hr, name, r, d, m in zip(t["HR"], t["Name"], ra, dec, t["Vmag"]):
            name = "" if name is np.ma.masked else str(name).strip().replace(",", " ")
            f.write(f"{int(hr)},{name},{r:.6f},{d:.6f},{float(m):.2f}\n")

    vmag = np.asarray(t["Vmag"], float)
    print(f"kept {len(t)} rows with positions -> {OUT}")
    print(f"brighter than 5.5: {(vmag < 5.5).sum()}")

    hr = np.asarray(t["HR"], int)
    for label, hr_id, ra_exp, dec_exp in [("Sirius", 2491, 101.287, -16.716),
                                          ("Polaris", 424, 37.955, 89.264)]:
        i = np.flatnonzero(hr == hr_id)[0]
        print(f"{label}: ra={ra[i]:.3f} (expect {ra_exp}), dec={dec[i]:.3f} (expect {dec_exp})")


if __name__ == "__main__":
    main()

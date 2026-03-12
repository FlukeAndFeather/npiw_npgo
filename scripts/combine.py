"""
Combine per-year processed NetCDF chunks into a single file.

Usage:
    python combine.py \
        --processed-dir /scratch/user/npiw_processed \
        --output /scratch/user/npiw_2004_2020.nc \
        --years 2004 2020
"""

import argparse
import os

import xarray as xr

# ── Defaults ──────────────────────────────────────────────────────────────────
LON        = (147, 230)
LAT        = (30, 60)
START_YEAR = 2004
END_YEAR   = 2020
# ──────────────────────────────────────────────────────────────────────────────


def processed_filename(year, lon, lat):
    return f"npiw_{year}_{int(lon[0])}to{int(lon[1])}E_{int(lat[0])}to{int(lat[1])}.nc"


def main():
    parser = argparse.ArgumentParser(description="Combine per-year isopycnal NetCDF chunks.")
    parser.add_argument("--processed-dir", required=True,                                   help="Directory containing per-year processed files")
    parser.add_argument("--output",        required=True,                                   help="Path for the combined output NetCDF")
    parser.add_argument("--years",  nargs=2, type=int, default=[START_YEAR, END_YEAR],      metavar=("START", "END"))
    parser.add_argument("--lon",    nargs=2, type=float, default=LON,                       metavar=("LON_W", "LON_E"))
    parser.add_argument("--lat",    nargs=2, type=float, default=LAT,                       metavar=("LAT_S", "LAT_N"))
    args = parser.parse_args()

    start_year, end_year = args.years

    year_files, missing = [], []
    for year in range(start_year, end_year + 1):
        path = os.path.join(args.processed_dir, processed_filename(year, args.lon, args.lat))
        if os.path.exists(path):
            year_files.append(path)
        else:
            missing.append(year)

    if missing:
        print(f"WARNING: missing processed files for years: {missing}")
    if not year_files:
        raise RuntimeError("No processed files found — nothing to combine.")

    print(f"Combining {len(year_files)} year(s): {start_year}–{end_year}")
    for f in year_files:
        print(f"  {f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    combined = xr.open_mfdataset(year_files, combine="by_coords")
    encoding = {v: {"zlib": True, "complevel": 4} for v in combined.data_vars}
    combined.to_netcdf(args.output, encoding=encoding)
    print(f"\nCombined output: {args.output}")


if __name__ == "__main__":
    main()

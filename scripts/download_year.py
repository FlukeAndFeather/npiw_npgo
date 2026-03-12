"""
Download one year of GLORYS12 data from Copernicus Marine Service.

Usage:
    python download_year.py --year 2010 --raw-dir /scratch/user/glorys_raw

The antimeridian-crossing case (LON_E > 180) is handled by issuing two
requests and concatenating before saving.
"""

import argparse
import os

import copernicusmarine
import xarray as xr

# ── Defaults ──────────────────────────────────────────────────────────────────
LON       = (147, 230)   # degrees East; >180 crosses the antimeridian
LAT       = (30, 60)
DEPTH_MAX = 1000         # metres
DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
VARIABLES  = ["thetao", "so"]
# ──────────────────────────────────────────────────────────────────────────────


def raw_filename(year, lon, lat):
    return f"glorys_{year}_{int(lon[0])}to{int(lon[1])}E_{int(lat[0])}to{int(lat[1])}.nc"


def fetch(lon_min, lon_max, lat_min, lat_max, start, end, depth_max):
    return copernicusmarine.open_dataset(
        dataset_id=DATASET_ID,
        variables=VARIABLES,
        minimum_longitude=lon_min,
        maximum_longitude=lon_max,
        minimum_latitude=lat_min,
        maximum_latitude=lat_max,
        minimum_depth=0,
        maximum_depth=depth_max,
        start_datetime=start,
        end_datetime=end,
    )


def main():
    parser = argparse.ArgumentParser(description="Download one year of GLORYS12 data.")
    parser.add_argument("--year",      type=int,   required=True,  help="Year to download (e.g. 2010)")
    parser.add_argument("--raw-dir",   required=True,              help="Directory to write raw NetCDF files")
    parser.add_argument("--lon",       nargs=2, type=float, default=LON,       metavar=("LON_W", "LON_E"))
    parser.add_argument("--lat",       nargs=2, type=float, default=LAT,       metavar=("LAT_S", "LAT_N"))
    parser.add_argument("--depth-max", type=float,          default=DEPTH_MAX, metavar="METRES")
    args = parser.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)

    lon_w, lon_e = args.lon
    lat_s, lat_n = args.lat
    start = f"{args.year}-01-01"
    end   = f"{args.year}-12-31"

    out_path = os.path.join(args.raw_dir, raw_filename(args.year, args.lon, args.lat))

    if os.path.exists(out_path):
        print(f"Already exists, skipping: {out_path}")
        return

    print(f"Fetching {args.year} ({lon_w}–{lon_e}°E, {lat_s}–{lat_n}°N) ...")
    if lon_e <= 180:
        ds = fetch(lon_w, lon_e, lat_s, lat_n, start, end, args.depth_max)
    else:
        ds_west = fetch(lon_w,  180,          lat_s, lat_n, start, end, args.depth_max)
        ds_east = fetch(-180,   lon_e - 360,  lat_s, lat_n, start, end, args.depth_max)
        ds_east = ds_east.assign_coords(longitude=ds_east["longitude"] + 360)
        ds = xr.concat([ds_west, ds_east], dim="longitude")

    print(f"  Saving to {out_path} ...")
    ds.to_netcdf(
        out_path,
        encoding={v: {"zlib": True, "complevel": 4} for v in VARIABLES},
    )
    print(f"  Done: {out_path}")


if __name__ == "__main__":
    main()

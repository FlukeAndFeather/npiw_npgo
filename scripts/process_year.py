"""
Download one year of GLORYS12 data, compute density, and interpolate onto
multiple isopycnal surfaces. Raw data is never written to disk.

Usage:
    python process_year.py --year 2010 --processed-dir /scratch/user/npiw_processed

Output: one NetCDF per year with variables sal_iso and depth_iso on a
(time, sigma, latitude, longitude) grid.
"""

import argparse
import os

import copernicusmarine
import gsw
import numpy as np
import xarray as xr

# ── Defaults ──────────────────────────────────────────────────────────────────
LON        = (147, 230)
LAT        = (30, 60)
ISOPYCNALS = [26.6, 26.8, 27.0]
DEPTH_MAX  = 1000
DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
# ──────────────────────────────────────────────────────────────────────────────


def processed_filename(year, lon, lat):
    return f"npiw_{year}_{int(lon[0])}to{int(lon[1])}E_{int(lat[0])}to{int(lat[1])}.nc"


def fetch_glorys(lon_min, lon_max, lat_min, lat_max, start, end, depth_max):
    return copernicusmarine.open_dataset(
        dataset_id=DATASET_ID,
        variables=["thetao", "so"],
        minimum_longitude=lon_min,
        maximum_longitude=lon_max,
        minimum_latitude=lat_min,
        maximum_latitude=lat_max,
        minimum_depth=0,
        maximum_depth=depth_max,
        start_datetime=start,
        end_datetime=end,
    )


def interp_isopycnal_vectorized(sigma, val, target):
    """
    Vectorized isopycnal interpolation across all profiles simultaneously.

    sigma, val: (time, depth, lat, lon) numpy arrays
    Returns: (time, lat, lon) array
    """
    nt, nd, nlat, nlon = sigma.shape
    N = nt * nlat * nlon

    s = sigma.transpose(0, 2, 3, 1).reshape(N, nd)
    v = val.transpose(0, 2, 3, 1).reshape(N, nd)

    nan_mask = np.isnan(s) | np.isnan(v)
    s_filled = np.where(nan_mask, np.inf, s)

    n_below = (s_filled <= target).sum(axis=1)

    result = np.full(N, np.nan)
    valid = (n_below > 0) & (n_below < nd)

    if valid.any():
        idx = np.where(valid)[0]
        i0 = n_below[idx] - 1
        i1 = n_below[idx]

        s0, s1 = s[idx, i0], s[idx, i1]
        v0, v1 = v[idx, i0], v[idx, i1]

        ds = s1 - s0
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(ds != 0, (target - s0) / ds, 0.0)

        interp_val = v0 + frac * (v1 - v0)
        bracket_nan = np.isnan(s0) | np.isnan(s1) | np.isnan(v0) | np.isnan(v1)
        interp_val[bracket_nan] = np.nan
        result[idx] = interp_val

    return result.reshape(nt, nlat, nlon)


def main():
    parser = argparse.ArgumentParser(description="Download and process one year of GLORYS12 onto isopycnals.")
    parser.add_argument("--year",          type=int,   required=True,              help="Year to process (e.g. 2010)")
    parser.add_argument("--processed-dir", required=True,                          help="Directory to write processed NetCDF files")
    parser.add_argument("--lon",           nargs=2, type=float, default=LON,       metavar=("LON_W", "LON_E"))
    parser.add_argument("--lat",           nargs=2, type=float, default=LAT,       metavar=("LAT_S", "LAT_N"))
    parser.add_argument("--isopycnals",    nargs="+", type=float, default=ISOPYCNALS, help="σ₀ surfaces to interpolate onto")
    parser.add_argument("--depth-max",     type=float, default=DEPTH_MAX,          metavar="METRES")
    parser.add_argument("--start",         default=None,                           help="Override start date (YYYY-MM-DD); defaults to Jan 1 of --year")
    parser.add_argument("--end",           default=None,                           help="Override end date (YYYY-MM-DD); defaults to Dec 31 of --year")
    args = parser.parse_args()

    os.makedirs(args.processed_dir, exist_ok=True)

    out_path = os.path.join(args.processed_dir, processed_filename(args.year, args.lon, args.lat))
    if os.path.exists(out_path):
        print(f"Already exists, skipping: {out_path}")
        return

    lon_w, lon_e = args.lon
    lat_s, lat_n = args.lat
    start = args.start or f"{args.year}-01-01"
    end   = args.end   or f"{args.year}-12-31"

    # ── Fetch (streamed, never written to disk) ────────────────────────────────
    print(f"Fetching {args.year} ...")
    if lon_e <= 180:
        ds = fetch_glorys(lon_w, lon_e, lat_s, lat_n, start, end, args.depth_max)
    else:
        ds_west = fetch_glorys(lon_w,  180,         lat_s, lat_n, start, end, args.depth_max)
        ds_east = fetch_glorys(-180,   lon_e - 360, lat_s, lat_n, start, end, args.depth_max)
        ds_east = ds_east.assign_coords(longitude=ds_east["longitude"] + 360)
        ds = xr.concat([ds_west, ds_east], dim="longitude")

    # ── Density ────────────────────────────────────────────────────────────────
    print("  Computing density ...")
    SA = gsw.SA_from_SP(ds["so"], ds["depth"], ds["longitude"], ds["latitude"])
    CT = gsw.CT_from_pt(SA, ds["thetao"])
    ds["sigma_theta"] = (("time", "depth", "latitude", "longitude"), gsw.sigma0(SA, CT).data)

    depth_1d = ds["depth"].values

    # sal_chunks[iso] and depth_chunks[iso] accumulate monthly results
    sal_chunks   = {iso: [] for iso in args.isopycnals}
    depth_chunks = {iso: [] for iso in args.isopycnals}
    times = []

    # ── Interpolate monthly chunks ─────────────────────────────────────────────
    print("  Interpolating ...")
    for month in range(1, 13):
        time_sel = ds.time.dt.month == month
        if not time_sel.any():
            continue

        ds_m = ds[["so", "sigma_theta"]].sel(time=time_sel).compute()
        sigma_np = ds_m["sigma_theta"].values
        so_np    = ds_m["so"].values
        depth_4d = np.broadcast_to(
            depth_1d[np.newaxis, :, np.newaxis, np.newaxis],
            sigma_np.shape,
        ).copy()

        for iso in args.isopycnals:
            sal_chunks[iso].append(interp_isopycnal_vectorized(sigma_np, so_np,     iso))
            depth_chunks[iso].append(interp_isopycnal_vectorized(sigma_np, depth_4d, iso))

        times.append(ds_m.time.values)
        print(f"    month {month}/12 done")

    time_arr = np.concatenate(times, axis=0)

    # Stack isopycnals into a sigma dimension: (time, sigma, lat, lon)
    sal_arr   = np.stack([np.concatenate(sal_chunks[iso],   axis=0) for iso in args.isopycnals], axis=1)
    depth_arr = np.stack([np.concatenate(depth_chunks[iso], axis=0) for iso in args.isopycnals], axis=1)

    coords = {
        "time":      time_arr,
        "sigma":     np.array(args.isopycnals),
        "latitude":  ds.latitude.values,
        "longitude": ds.longitude.values,
    }
    dims = ("time", "sigma", "latitude", "longitude")
    out = xr.Dataset(
        {
            "sal_iso":   xr.DataArray(sal_arr,   dims=dims, coords=coords),
            "depth_iso": xr.DataArray(depth_arr, dims=dims, coords=coords),
        }
    )
    out["sal_iso"].attrs   = {"units": "psu", "long_name": "Salinity on σ₀ isopycnal"}
    out["depth_iso"].attrs = {"units": "m",   "long_name": "Depth of σ₀ isopycnal"}
    out["sigma"].attrs     = {"units": "kg/m³", "long_name": "Potential density anomaly σ₀"}

    out.to_netcdf(
        out_path,
        encoding={
            "sal_iso":   {"zlib": True, "complevel": 4},
            "depth_iso": {"zlib": True, "complevel": 4},
        },
    )
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()

import os
import numpy as np
import copernicusmarine
import gsw
import xarray as xr

# ── Configuration ─────────────────────────────────────────────────────────────
# Longitude as (west_bound, east_bound) in degrees East.
# To cross the antimeridian, set east_bound > 180.
# e.g. LON = (147, 230) spans 147°E → 130°W
LON       = (147, 230)
LAT       = (30, 60)
START     = "2004-01-01"
END       = "2020-12-31"
ISOPYCNAL = 26.8
DEPTH_MAX = 1000  # metres
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

fname = f"npiw_iso{ISOPYCNAL}_{START}_{END}_{LON[0]}to{LON[1]}E_{LAT[0]}to{LAT[1]}.nc"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, fname)


def interp_isopycnal_vectorized(sigma, val, target):
    """
    Vectorized isopycnal interpolation across all profiles simultaneously.

    sigma, val: numpy arrays with depth on axis=1, shape (time, depth, lat, lon)
    Returns: (time, lat, lon) array

    Replaces the per-profile Python loop in apply_ufunc/vectorize=True.
    Assumes sigma is monotonically increasing with depth (standard ocean convention).
    NaN values in sigma are treated as missing and excluded per profile.
    """
    nt, nd, nlat, nlon = sigma.shape
    N = nt * nlat * nlon

    # Reshape to (N, depth) for vectorized operations
    s = sigma.transpose(0, 2, 3, 1).reshape(N, nd)
    v = val.transpose(0, 2, 3, 1).reshape(N, nd)

    # Replace NaN sigma with inf so they don't count as "below target"
    nan_mask = np.isnan(s) | np.isnan(v)
    s_filled = np.where(nan_mask, np.inf, s)

    # Count levels with sigma <= target (lower bracket = n_below - 1)
    n_below = (s_filled <= target).sum(axis=1)  # (N,)

    result = np.full(N, np.nan)
    valid = (n_below > 0) & (n_below < nd)

    if valid.any():
        idx = np.where(valid)[0]
        i0 = n_below[idx] - 1
        i1 = n_below[idx]

        s0, s1 = s[idx, i0], s[idx, i1]
        v0, v1 = v[idx, i0], v[idx, i1]

        ds = s1 - s0
        with np.errstate(invalid='ignore', divide='ignore'):
            frac = np.where(ds != 0, (target - s0) / ds, 0.0)

        interp_val = v0 + frac * (v1 - v0)
        # Invalidate if NaN in the bracket values
        bracket_nan = np.isnan(s0) | np.isnan(s1) | np.isnan(v0) | np.isnan(v1)
        interp_val[bracket_nan] = np.nan

        result[idx] = interp_val

    return result.reshape(nt, nlat, nlon)


def fetch_glorys(lon_min, lon_max, start, end):
    return copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["thetao", "so"],
        minimum_longitude=lon_min,
        maximum_longitude=lon_max,
        minimum_latitude=LAT[0],
        maximum_latitude=LAT[1],
        minimum_depth=0,
        maximum_depth=DEPTH_MAX,
        start_datetime=start,
        end_datetime=end,
    )


def process_year(year):
    start = f"{year}-01-01"
    end   = f"{year}-12-31"
    lon_w, lon_e = LON

    print(f"  Fetching {year}...")
    if lon_e <= 180:
        ds = fetch_glorys(lon_w, lon_e, start, end)
    else:
        ds_west = fetch_glorys(lon_w, 180, start, end)
        ds_east = fetch_glorys(-180, lon_e - 360, start, end)
        ds_east = ds_east.assign_coords(longitude=ds_east["longitude"] + 360)
        ds = xr.concat([ds_west, ds_east], dim="longitude")

    print(f"  Computing density {year}...")
    SA = gsw.SA_from_SP(ds["so"], ds["depth"], ds["longitude"], ds["latitude"])
    CT = gsw.CT_from_pt(SA, ds["thetao"])
    ds["sigma_theta"] = (("time", "depth", "latitude", "longitude"), gsw.sigma0(SA, CT).data)

    # Process one month at a time to stay within memory (~600MB/month)
    # Compute so + sigma_theta together per month to avoid downloading twice
    print(f"  Interpolating {year} (monthly chunks)...")
    depth_1d = ds["depth"].values

    sal_chunks, depth_chunks, times = [], [], []
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

        sal_chunks.append(interp_isopycnal_vectorized(sigma_np, so_np,     ISOPYCNAL))
        depth_chunks.append(interp_isopycnal_vectorized(sigma_np, depth_4d, ISOPYCNAL))
        times.append(ds_m.time.values)
        print(f"    month {month}/12 done")

    sal_arr   = np.concatenate(sal_chunks,   axis=0)
    depth_arr = np.concatenate(depth_chunks, axis=0)
    time_arr  = np.concatenate(times,        axis=0)

    coords = {"time": time_arr, "latitude": ds.latitude.values, "longitude": ds.longitude.values}
    out = xr.Dataset(
        {
            "sal_iso":   xr.DataArray(sal_arr,   dims=("time","latitude","longitude"), coords=coords),
            "depth_iso": xr.DataArray(depth_arr, dims=("time","latitude","longitude"), coords=coords),
        }
    )
    out["sal_iso"].attrs   = {"units": "psu", "long_name": f"Salinity at σ₀={ISOPYCNAL}"}
    out["depth_iso"].attrs = {"units": "m",   "long_name": f"Depth of σ₀={ISOPYCNAL} isopycnal"}
    return out


# ── Process year by year, write to separate files, then combine ───────────────
start_year = int(START[:4])
end_year   = int(END[:4])

CHUNKS_DIR = os.path.join(OUTPUT_DIR, "chunks")
os.makedirs(CHUNKS_DIR, exist_ok=True)

year_files = []
for year in range(start_year, end_year + 1):
    chunk_path = os.path.join(CHUNKS_DIR, f"npiw_iso{ISOPYCNAL}_{year}_{LON[0]}to{LON[1]}E_{LAT[0]}to{LAT[1]}.nc")
    year_files.append(chunk_path)

    if os.path.exists(chunk_path):
        print(f"Processing {year}... (already exists, skipping)")
        continue

    print(f"Processing {year}...")
    out = process_year(year)
    out.to_netcdf(
        chunk_path,
        encoding={
            "sal_iso":   {"zlib": True, "complevel": 4},
            "depth_iso": {"zlib": True, "complevel": 4},
        },
    )
    print(f"  Saved {year}.")

print("Combining years...")
combined = xr.open_mfdataset(year_files, combine="by_coords")
combined.to_netcdf(
    OUTPUT_PATH,
    encoding={
        "sal_iso":   {"zlib": True, "complevel": 4},
        "depth_iso": {"zlib": True, "complevel": 4},
    },
)
print(f"\nComplete: {OUTPUT_PATH}")

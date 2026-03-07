import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import copernicusmarine
import gsw
import xarray as xr

# ── Configuration ─────────────────────────────────────────────────────────────
# Longitude as (west_bound, east_bound) in degrees East.
# To cross the antimeridian, set west_bound > 180 or east_bound > 180.
# e.g. LON = (147, 230) spans 147°E → 130°W
LON        = (147, 230)
LAT        = (30, 60)
START      = "2022-01-01"
END        = "2022-12-31"
ISOPYCNAL  = 26.8
DURATION   = 100     # seconds
DEPTH_MAX  = 1000  # metres
SAL_CONTOURS   = [33.7, 33.9, 34.1]   # psu
DEPTH_CONTOURS = [200,  400,  600]    # m
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures", "animations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

fname = f"npiw_sal_depth_{START}_{END}_{LON[0]}to{LON[1]}E_{LAT[0]}to{LAT[1]}.mp4"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, fname)


def interp_to_isopycnal(sigma, val, target):
    mask = ~np.isnan(sigma) & ~np.isnan(val)
    if mask.sum() < 2:
        return np.nan
    s, v = sigma[mask], val[mask]
    if target < s.min() or target > s.max():
        return np.nan
    return float(np.interp(target, s, v))


def apply_isopycnal(ds, var, target):
    """Interpolate var to isopycnal across all time steps."""
    return xr.apply_ufunc(
        interp_to_isopycnal,
        ds["sigma_theta"].chunk({"depth": -1}),
        ds[var].chunk({"depth": -1}),
        kwargs={"target": target},
        input_core_dims=[["depth"], ["depth"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )


# ── Fetch data ────────────────────────────────────────────────────────────────
# Convert LON from 0-360 to -180/180 for CMEMS, splitting at antimeridian if needed
def fetch_glorys(lon_min, lon_max):
    return copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["thetao", "so"],
        minimum_longitude=lon_min,
        maximum_longitude=lon_max,
        minimum_latitude=LAT[0],
        maximum_latitude=LAT[1],
        minimum_depth=0,
        maximum_depth=DEPTH_MAX,
        start_datetime=START,
        end_datetime=END,
    )

print("Fetching GLORYS data...")
lon_w, lon_e = LON  # in 0-360

if lon_e <= 180:
    # No antimeridian crossing
    ds = fetch_glorys(lon_w, lon_e)
else:
    # Crosses antimeridian: split into two fetches
    # Western chunk: lon_w → 180 (stored as-is in CMEMS)
    # Eastern chunk: -180 → (lon_e - 360), then shift to 0-360
    ds_west = fetch_glorys(lon_w, 180)
    ds_east = fetch_glorys(-180, lon_e - 360)
    ds_east = ds_east.assign_coords(longitude=ds_east["longitude"] + 360)
    ds = xr.concat([ds_west, ds_east], dim="longitude")

# ── Compute sigma_theta ───────────────────────────────────────────────────────
print("Computing density...")
SA = gsw.SA_from_SP(ds["so"], ds["depth"], ds["longitude"], ds["latitude"])
CT = gsw.CT_from_pt(SA, ds["thetao"])
sigma_theta = gsw.sigma0(SA, CT)
ds["sigma_theta"] = (("time", "depth", "latitude", "longitude"), sigma_theta.data)

# ── Interpolate to isopycnal for all time steps ───────────────────────────────
print("Interpolating to isopycnal...")
print("Salinity...")
sal_iso   = apply_isopycnal(ds, "so",    ISOPYCNAL).compute()
print("Depth...")
depth_iso = apply_isopycnal(ds, "depth", ISOPYCNAL).compute()

# ── Colormap norms (fixed across all frames) ─────────────────────────────────
sal_norm   = mcolors.TwoSlopeNorm(vmin=32,  vcenter=34.3, vmax=36)
depth_norm = mcolors.TwoSlopeNorm(vmin=0,   vcenter=500,  vmax=1000)

lons = sal_iso["longitude"].values
lats = sal_iso["latitude"].values
times = sal_iso["time"].values

# ── Build figure ──────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

mesh1 = ax1.pcolormesh(lons, lats, sal_iso.isel(time=0),
                       cmap="PiYG", norm=sal_norm)
cb1 = plt.colorbar(mesh1, ax=ax1, label="Salinity (psu)")

mesh2 = ax2.pcolormesh(lons, lats, depth_iso.isel(time=0),
                       cmap="RdYlBu", norm=depth_norm)
cb2 = plt.colorbar(mesh2, ax=ax2, label="Depth (m)")

def lon_label(v):
    """Convert 0-360 longitude value to a readable label."""
    v = v % 360
    if v == 0 or v == 180:
        return f"{int(v)}°"
    return f"{int(v)}°E" if v < 180 else f"{int(360 - v)}°W"

tick_lons = np.arange(
    np.ceil(lons.min() / 30) * 30,
    np.floor(lons.max() / 30) * 30 + 1,
    30
)

for ax in (ax1, ax2):
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xticks(tick_lons)
    ax.set_xticklabels([lon_label(v) for v in tick_lons])

date_str0 = str(times[0])[:10]
title1 = ax1.set_title(f"Salinity on σ₀={ISOPYCNAL} isopycnal, {date_str0}")
title2 = ax2.set_title(f"Depth of σ₀={ISOPYCNAL} isopycnal, {date_str0}")
plt.tight_layout()

contours1 = [None]
contours2 = [None]


def update(i):
    date_str = str(times[i])[:10]
    mesh1.set_array(sal_iso.isel(time=i).values.ravel())
    mesh2.set_array(depth_iso.isel(time=i).values.ravel())
    title1.set_text(f"Salinity on σ₀={ISOPYCNAL} isopycnal, {date_str}")
    title2.set_text(f"Depth of σ₀={ISOPYCNAL} isopycnal, {date_str}")

    # Remove previous contours
    if contours1[0] is not None:
        contours1[0].remove()
    if contours2[0] is not None:
        contours2[0].remove()

    contours1[0] = ax1.contour(lons, lats, sal_iso.isel(time=i),
                                levels=SAL_CONTOURS, colors="k", linewidths=0.7)
    contours2[0] = ax2.contour(lons, lats, depth_iso.isel(time=i),
                                levels=DEPTH_CONTOURS, colors="k", linewidths=0.7)
    ax1.clabel(contours1[0], fmt="%.1f", fontsize=7)
    ax2.clabel(contours2[0], fmt="%d m", fontsize=7)


n_frames = len(times)
save_fps = n_frames / DURATION  # fps needed to hit target duration

print(f"Rendering {n_frames} frames...")
anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)

anim.save(OUTPUT_PATH, fps=save_fps, writer="ffmpeg", dpi=150)
print(f"Saved: {OUTPUT_PATH}")

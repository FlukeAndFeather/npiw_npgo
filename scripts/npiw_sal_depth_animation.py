import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copernicusmarine
import gsw
import xarray as xr

# ── Configuration ─────────────────────────────────────────────────────────────
LON        = (-180, -130)
LAT        = (30, 60)
START      = "2022-01-01"
END        = "2022-06-30"
ISOPYCNAL  = 26.8
DURATION   = 5     # seconds
DEPTH_MAX  = 1000  # metres
SAL_CONTOURS   = [33.7, 33.9, 34.1]   # psu
DEPTH_CONTOURS = [200,  400,  600]    # m
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures", "animations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

fname = f"npiw_sal_depth_{START}_{END}_{LON[0]}to{LON[1]}_{LAT[0]}to{LAT[1]}.mp4"
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
print("Fetching GLORYS data...")
ds = copernicusmarine.open_dataset(
    dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
    variables=["thetao", "so"],
    minimum_longitude=LON[0],
    maximum_longitude=LON[1],
    minimum_latitude=LAT[0],
    maximum_latitude=LAT[1],
    minimum_depth=0,
    maximum_depth=DEPTH_MAX,
    start_datetime=START,
    end_datetime=END,
)

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

# ── Fixed colormap limits ─────────────────────────────────────────────────────
sal_vmin,   sal_vmax   = float(np.nanmin(sal_iso)),   float(np.nanmax(sal_iso))
depth_vmin, depth_vmax = float(np.nanmin(depth_iso)), float(np.nanmax(depth_iso))

lons = sal_iso["longitude"].values
lats = sal_iso["latitude"].values
times = sal_iso["time"].values

# ── Build figure ──────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

mesh1 = ax1.pcolormesh(lons, lats, sal_iso.isel(time=0),
                       cmap="RdYlBu_r", vmin=sal_vmin, vmax=sal_vmax)
cb1 = plt.colorbar(mesh1, ax=ax1, label="Salinity (psu)")

mesh2 = ax2.pcolormesh(lons, lats, depth_iso.isel(time=0),
                       cmap="viridis_r", vmin=depth_vmin, vmax=depth_vmax)
cb2 = plt.colorbar(mesh2, ax=ax2, label="Depth (m)")
cb2.ax.invert_yaxis()

for ax in (ax1, ax2):
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

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

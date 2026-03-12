import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import xarray as xr

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "processed",
    "npiw_iso26.8_2022-01-01_2022-12-31_147to230E_30to60.nc"
)
DURATION       = 100   # seconds
SAL_CONTOURS   = [33.7, 33.9, 34.1]   # psu
DEPTH_CONTOURS = [200, 400, 600]       # m
# ──────────────────────────────────────────────────────────────────────────────

if not os.path.exists(INPUT_PATH):
    sys.exit(f"Input file not found: {INPUT_PATH}\nRun npiw_process.py first.")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures", "animations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ds = xr.open_dataset(INPUT_PATH)
sal_iso   = ds["sal_iso"]
depth_iso = ds["depth_iso"]
isopycnal = ds.attrs.get("isopycnal", "?")

lons  = sal_iso["longitude"].values
lats  = sal_iso["latitude"].values
times = sal_iso["time"].values

base = os.path.splitext(os.path.basename(INPUT_PATH))[0]
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{base}.mp4")

# ── Colormap norms ────────────────────────────────────────────────────────────
sal_norm   = mcolors.TwoSlopeNorm(vmin=32,  vcenter=34.3, vmax=36)
depth_norm = mcolors.TwoSlopeNorm(vmin=0,   vcenter=500,  vmax=1000)


def lon_label(v):
    v = v % 360
    if v == 0 or v == 180:
        return f"{int(v)}°"
    return f"{int(v)}°E" if v < 180 else f"{int(360 - v)}°W"


tick_lons = np.arange(
    np.ceil(lons.min() / 30) * 30,
    np.floor(lons.max() / 30) * 30 + 1,
    30
)

# ── Build figure ──────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

mesh1 = ax1.pcolormesh(lons, lats, sal_iso.isel(time=0), cmap="PiYG", norm=sal_norm)
cb1 = plt.colorbar(mesh1, ax=ax1, label="Salinity (psu)")

mesh2 = ax2.pcolormesh(lons, lats, depth_iso.isel(time=0), cmap="RdYlBu", norm=depth_norm)
cb2 = plt.colorbar(mesh2, ax=ax2, label="Depth (m)")

for ax in (ax1, ax2):
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xticks(tick_lons)
    ax.set_xticklabels([lon_label(v) for v in tick_lons])

date_str0 = str(times[0])[:10]
title1 = ax1.set_title(f"Salinity on σ₀={isopycnal} isopycnal, {date_str0}")
title2 = ax2.set_title(f"Depth of σ₀={isopycnal} isopycnal, {date_str0}")
plt.tight_layout()

contours1 = [None]
contours2 = [None]


def update(i):
    date_str = str(times[i])[:10]
    mesh1.set_array(sal_iso.isel(time=i).values.ravel())
    mesh2.set_array(depth_iso.isel(time=i).values.ravel())
    title1.set_text(f"Salinity on σ₀={isopycnal} isopycnal, {date_str}")
    title2.set_text(f"Depth of σ₀={isopycnal} isopycnal, {date_str}")

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
save_fps = n_frames / DURATION

print(f"Rendering {n_frames} frames...")
anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)

anim.save(OUTPUT_PATH, fps=save_fps, writer="ffmpeg", dpi=150)
print(f"Saved: {OUTPUT_PATH}")

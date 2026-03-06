## Goal

Visualize the penetration of NPIW east and south across the North Pacific.

## Output

An animation covering a _configurable_ space/time domain. E.g., Jan-Jun of 2022 in lon [-180, -130] and lat [30, 60]. Each panel of the animation should mirror the prototype figure created at the end of 01_explore_glorys.ipynb. I.e., salinity at isopycnal above, depth of isopycnal below. Also _configure_ the total duration and frame rate of the animation.

## Data and processing

Use the GLORYS model outputs via Copernicus. Calculate density using gsw functions. Interpolate density using scipy. 01_explore_glorys.ipynb contains code chunks for each of these tasks.

## Organization

I want one script that I can run from the command line that will produce an .mp4 output. Make the output look like the following tree structure. The .mp4 output should go in a subfolder of figures called "animations".

npiw_npgo/
├── notebooks/       # exploration, scratch
├── scripts/         # refined, runnable pipeline
├── outputs/         # figures, processed data
│   └── figures/
├── environment.yml
└── .gitignore

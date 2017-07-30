### Display 3DFSC and coloring original map by angular resolution
### Written by Tom Goddard
### Modified by Yong Zi Tan

# Open lineplot.py
open lineplot.py

# Open both volumes and hide the 3DFSC volume.
set bg_color white
open #0 T40-3DFSC.mrc
volume #0 originIndex 71,71,71
volume #0 voxelSize 1.31
open #1 130K-T40.mrc
volume #1 originIndex 72,72,72
volume #1 voxelSize 1.31
volume #0 hide
focus
colorkey 0.2,0.14 0.8,0.1 tickMarks True tickThickness 3 "Poorer" red "Better" blue ; 2dlabels create Label text "Relative XY-plane resolution (AU)" color black xpos 0.2 ypos 0.16

# Execute lineplot.py
fscplot #0


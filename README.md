# Anisotropy
These are programs that deal with anisotropy (both resolution and sampling) written by PR Baldwin


 Author: Philip Baldwin, March 2017


  This program is  ThreeDFSC_ReleaseFeb2017.py
 A conical resolution program  written by P. R. Baldwin in December 2016

Usage: ThreeDFSC_ReleaseFeb2017.py
                HalfMap1.hdf    HalfMap2.hdf   OutputStringLabel   AngstromsPerPixel            
DeltaTheta is hard coded to 20 degrees


Best is if this program comes with the anaconda package (2 or 3)
 which corresponds to Python2.7 or 3.5
This requires the existence of numba, but not numbapro

--------------------------------  Imports
Here are the imports 

from sys import argv
import csv
import time
import os
import sys
import numpy as np
from math import *
from numba import *
from numba import autojit
import copy
import h5py

import matplotlib
import matplotlib.pyplot as plt

-----------------------------  End Imports


The functions need to be kept separated so that the precompiler can
 notice the @autojit decorations and  precompile the code,
 
First convert half maps to hdf  using EMAN 
    e2proc3d.py T10_3_r1_map2Sh2.mrc T10_3_r1_map2Sh2.hdf
    a good value for  dthetaInDegrees is 20, and is hard coded




#--------------------------------------------------------------------------------------
#    Example Run


ThreeDFSC_ReleaseFeb2017.py HA-T10_half1_class001_unfil.hdf HA-T10_half2_class001_unfil.hdf T10               1.09
                              HalfMap1                        HalfMap2                      LabelForOutputs   A/Pixel

Creates a directory for output named
             ResultsT_10
   

ResEMT10Out.hdf              The main 3D FSC output: it is a volume of the 3D directional FSC
ResEMT10OutglobalFSC.csv     The usual global FSC: There are 3 columns: normalized frequency, freq in inv A, FSC
PlotsT10.csv                 Total Five Columns: The directional FSC taken along x,y,z. 4) cos of the phase residual, 5) global FSC

FTPlotT10.jpg                Plot of Fourier Intensity averaged over shells in Fourier space
PlotsT10.jpg                 Same 5 Columns as Plots...jpg


                            Additional diagnostics
Radialn1T10Out.hdf  
Radialn2T10Out.hdf  
RadialResEMT10Out.hdf     
#RadialcHT10Out.hdf  
#Radialn12T10Out.hdf  
#RadialFTT10Out.hdf   

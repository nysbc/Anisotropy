This is CreateSampVolPart1ReleaseMar2017.py

Author: Philip Baldwin, March 2017
CreateSampVolPart1ReleaseMar2017.py


usage:
CreateSampVolPart1ReleaseMar2017.py ...
        StarFileDir StarFile  ResFileDir RMax 
           OutputStringLabel NumberToUse CSym CorD

Please use absolute paths for StarFileDir and ResFileDir

This program takes a cryosparc csv file or a Relion Star file
 from the StarFileDir. This gives a set of Euler Angles,
    or equivalently projection directions. 
       The directions are symmetrized by CSym (the axial 
   symmetry) and CorD (whether it is C or D). 
   A Sampling Volume is created of size N^3 (where N =2RMax+1).
   It is written to the same directory [ResFileDir] as that for the conical FSC outputs,
  using OutputStringLabel to label the Sampling Volume Name.
  It is 'SampVolBT'+OutputStringLabel +'Mid'+str(Mid)+'.hdf'

The StarFileDir may be  different from the ResFileDir.
Typically the ResFileDir is constructed from the same OutputStringLabel,
    but we allow for some redundancy here.
   

Comments: The order of this program is  RMax^3 * NVecs. 
This can be slow, if RMax is large. I would recommend not picking RMax above 50 for now.
SampFuncFromProjN needs to be rewritten to be of order 
    RMax^2 * NVecs. 
We know the projection directions and know approximately very easily
the lattice sites in the projection slab. We simply must do 
a nint to find the correct sites from a sufficiently fine sampling and find the unique sites.



 
StarFileDir   = argv[1] ; # os.chdir(StarFileDir);
StarFile      = argv[2] ; # this can be either a Relion .star file
                         # or a cryosparc .csv file 
ResFileDir    = argv[3]; # This is the directory where the half maps are, 
                         # and where the output will be placed.

RMax               = int(argv[4]);
OutputStringLabel  =     argv[5] ;  # This will allow us to access all the files that were created from the FSCcode 
                                    # ThreeDFSC_ReleaseMar2017.py; It should be the same label as used in the FSC code
                                    # The Results Directory from the FSC code will look like 
                                    #  'Results'+OutputStringLabel

NumberToUse        = int(argv[6]);
CSym               = int(argv[7]);# This is the rotational symmetry around z
CorD               = int(argv[8]);# This is 1 or 2 for C or D



 StarFile might be 'HA-T30_data.star' ,  or 'cryosparc_exp000039_000.csv'


Best is if this program comes with the anaconda package (2 or 3)

which corresponds to Python2.7 or 3.5

This requires the existence of numba, but not numbapro



#%%    Imports
import sys
from sys import argv
import time
import pandas as pd;
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import shutil
import csv
import h5py
import numpy as np;
from numba import autojit


#%%

#Functions
def ReadStarOrCryoSparcFile(StarFile):
    #fH = open('HA-T30_data.star','r')
    #fH = open('cryosparc_exp000039_000.csv','r')
  return StarFileData


 
def AnglesToVecsCD(AngleTilts,AngleRots,CSym,DSym):#DSym=1 or 2
    return NVecs

def AnglesToVecsCDv2(AngleTilts,AngleRots,CSym,DSym):#DSym=1 or 2
    return NVecs


def SampFuncFromProjN(Mid, NVecs):# Mid is the related to the image size N = 2*Mid+1
    return [SampVol,LattSit]
    SampVol is the N^3 volume
    LattSit is the N^3 by 3 list of the lattice sites.


Oututs Created are:

NVecsProteaMar072017.npy         - This is a numpy storage for the NVecs (projection directions)
SampVolBTProteaMar072017Mid31.hdf - This is the Sampling Volume for the set of NVecs 




#!/usr/bin/env python
# -*- coding: UTF-8 -*-
### Require Anaconda3

### ============================
### 3D FSC Software Wrapper
### Written by Philip Baldwin
### Edited by Yong Zi Tan and Dmitry Lyumkis
### Downloaded from https://github.com/nysbc/Anisotropy
### 
### See Paper:
### Addressing preferred specimen orientation in single-particle cryo-EM through tilting
### 10.1038/nmeth.4347
###
### Credits:
### 1) UCSF Chimera, especially Tom Goddard
### 2) mrcfile 1.0.0 by Colin Palmer (https://github.com/ccpem/mrcfile)
###
### Version 2.3 (23 July 2017)
### 
### Revisions 
### 1.1 - Added mpl.use('Agg') to allow matplotlib to be used without X-server
###     - Added Sum of Standard Deviation
### 1.2 - Added FSCCutoff Option
### 2.0 - Incorporation of AutoJIT version of 3D FSC for 10x faster processing
### 2.1 - 3D FSC takes MRC files
### 2.2 - Fixed bugs with 3DFSC missing a line in volume, and plotting title error
### 2.3 - Fixed various bugs, new thresholding algorithm, added progress bar, improved Chimera plotting, more error checking
version = "2.3"
### ============================

#pythonlib
import matplotlib
matplotlib.use('Agg')
from optparse import OptionParser
import os
import sys
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import mrcfile
import time

#sys.path.append(os.path.join(os.getcwd(),'../'))

#from ThreeDFSC.programs import ThreeDFSC_ReleaseAug2017
#from ThreeDFSC.programs import ThreeDFSC_Analysis_V3
from programs import ThreeDFSC_ReleaseAug2017
from programs import ThreeDFSC_Analysis_V3
start_time = time.time()

# Check Anaconda version

def masking(inmrc,mask,masked_outmrc):
	inputmrc = (mrcfile.open(inmrc)).data
	mask = (mrcfile.open(mask)).data
	outarray = np.multiply(inputmrc,mask)
	
	mrc_write = mrcfile.new(masked_outmrc,overwrite=True)
	mrc_write.set_data(outarray.astype('<f4'))
	mrc_write.close()
	return os.path.abspath(str(masked_outmrc))

def execute(options):
    print("Execute")
    # Part 00
    
    # Check required inputs
    if (None in (options.halfmap1, options.halfmap2, options.fullmap, options.apix)):
            print ("\n\033[1;31;40mError: A required input is missing.\033[0;37;40m\n")
            parser.print_help()
            sys.exit()
    
    # Convert file paths to absolutes
    halfmap1_pre = os.path.abspath(str(options.halfmap1))
    halfmap2_pre = os.path.abspath(str(options.halfmap2))
    fullmap = os.path.abspath(str(options.fullmap))
    
    # Masking
    if (bool(options.mask) == False):
            halfmap1 = halfmap1_pre
            halfmap2 = halfmap2_pre
    else:
            mask = os.path.abspath(str(options.mask))
            halfmap1 = masking(options.halfmap1,mask,options.halfmap1[:-4] + "_masked.mrc")
            halfmap2 = masking(options.halfmap2,mask,options.halfmap2[:-4] + "_masked.mrc")
            print ("\nMasking performed: " + options.halfmap1[:-4] + "_masked.mrc and " + options.halfmap2[:-4] + "_masked.mrc generated.")
    # Check half maps
    if halfmap1 == halfmap2:
            print ("\n\033[1;31;40mError: Both your half maps point to the same file.\033[0;37;40m\n")
            sys.exit()
    
    # Part 01
    
    if (options.Skip3DFSCGeneration == "False"):
            print ("\n\033[1;34;40mStep 01: Generating 3DFSC Volume \033[0;37;40m")
            ThreeDFSC_ReleaseAug2017.main(halfmap1,halfmap2,options.ThreeDFSC,options.apix,options.dthetaInDegrees)
            directory = "Results_" + options.ThreeDFSC
            if not os.path.exists(directory):
                os.makedirs(directory)
            os.system("cp Results_" + options.ThreeDFSC + "/ResEM" + options.ThreeDFSC + "Out.mrc Results_" + options.ThreeDFSC + "/" + options.ThreeDFSC + ".mrc")

            print ("3DFSC Results_" + options.ThreeDFSC + "/" + options.ThreeDFSC + ".mrc generated.")
    elif (options.Skip3DFSCGeneration == "True"):
            print ("\n\033[1;34;40mStep 01: Skipped\033[0;37;40m\nUsing pre-existing 3DFSC volume and output files.")
            if os.path.isfile("Results_" + options.ThreeDFSC + "/ResEM" + options.ThreeDFSC + "OutglobalFSC.csv") == False:
                    print ("\033[1;31;40mResults_" + options.ThreeDFSC + "/ResEM" + options.ThreeDFSC + "OutglobalFSC.csv missing! Please re-run entire 3DFSC program to generate the files needed for analysis.\033[0;37;40m\n")
                    sys.exit()
            elif os.path.isfile("Results_" + options.ThreeDFSC + "/ResEM" + options.ThreeDFSC + "Out.mrc") == False:
                    print ("\033[1;31;40mResults_" + options.ThreeDFSC + "/ResEM" + options.ThreeDFSC + "Out.mrc missing! Please re-run entire 3DFSC program to generate the files needed for analysis.\033[0;37;40m\n")
                    sys.exit()
            else:
                    os.system("cp Results_" + options.ThreeDFSC + "/ResEM" + options.ThreeDFSC + "Out.mrc Results_" + options.ThreeDFSC + "/" + options.ThreeDFSC + ".mrc")
    else:
            print ("\033[1;31;40mPlease key in either True or False for --Skip3DFSCGeneration option.\033[0;37;40m\n")
            sys.exit()
    
    # Part 02
    print ("\n\033[1;34;40mStep 02: Generating Analysis Files \033[0;37;40m")
    ThreeDFSC_Analysis_V3.main(halfmap1,halfmap2,fullmap,options.apix,options.ThreeDFSC,options.dthetaInDegrees,options.histogram,options.FSCCutoff,options.ThresholdForSphericity,options.HighPassFilter)
    print ("\nDone")
    print ("Results are in the folder Results_" + str(options.ThreeDFSC))
    print ("--- %s seconds ---" % (time.time() - start_time))
    print ("Please email pbaldwin@nysbc.org, ytan@nysbc.org and dlyumkis@salk.edu if there are any problems/suggestions. Thank you.\n")
    return

if __name__ == '__main__':
    parser = OptionParser(usage="usage: %prog [options]", version="%prog " + version)
    parser.add_option("--halfmap1", dest="halfmap1", action="store", type="string", help="First half map of 3D reconstruction. MRC format. Can be masked or unmasked. \033[1;34;40mRequired. \033[0;37;40m", metavar="HALFMAP1.MRC")
    parser.add_option("--halfmap2", dest="halfmap2", action="store", type="string", help="Second half map of 3D reconstruction. MRC format. Can be masked or unmasked. \033[1;34;40mRequired. \033[0;37;40m", metavar="HALFMAP2.MRC")
    parser.add_option("--fullmap", dest="fullmap", action="store", type="string", help="Full map of 3D reconstruction. MRC format. Can be masked or unmasked, can be sharpened or unsharpened. \033[1;34;40mRequired. \033[0;37;40m", metavar="FULLMAP.MRC")
    parser.add_option("--apix", dest="apix", action="store", type="float", default=1, help="Angstrom per pixel of 3D map. \033[1;34;40mRequired. \033[0;37;40m", metavar="FLOAT")
    parser.add_option("--mask", dest="mask", action="store", type="string", help="If given, it would be used to mask the half maps during 3DFSC generation and analysis.", metavar="MASK.MRC")
    parser.add_option("--ThreeDFSC", dest="ThreeDFSC", action="store", type="string", default="3DFSCOutput", help="Name of output 3DFSC map. No file extension required - it will automatically be given a .mrc extension. No paths please.", metavar="FILENAME")
    parser.add_option("--dthetaInDegrees", dest="dthetaInDegrees", action="store", type="float", default=20, help="Angle of cone to be used for 3D FSC sampling in degrees. Default is 20 degrees.", metavar="FLOAT")
    parser.add_option("--histogram", dest="histogram", action="store", type="string", default="histogram", help="Name of output histogram graph. No file extension required - it will automatically be given a .pdf extension. No paths please.", metavar="FILENAME")
    parser.add_option("--FSCCutoff", dest="FSCCutoff", action="store", type="float", default=0.143, help="FSC cutoff criterion. 0.143 is default.", metavar="FLOAT")
    parser.add_option("--ThresholdForSphericity", dest="ThresholdForSphericity", action="store", type="float", default=0.5, help="Threshold value for 3DFSC volume for calculating sphericity. 0.5 is default.", metavar="FLOAT")
    parser.add_option("--HighPassFilter", dest="HighPassFilter", action="store", type="float", default=150.0, help="High pass filter for thresholding in Angstrom. Prevents small dips in directional FSCs at low spatial frequency due to noise from messing up the thresholding step. Decrease if you see a huge wedge missing from your thresholded 3DFSC volume. 150 Angstroms is default.", metavar="FLOAT")
    parser.add_option("--Skip3DFSCGeneration", dest="Skip3DFSCGeneration", action="store", type="string", default="False", help="Allows for skipping of 3DFSC generation to directly run the analysis on a previously generated set of results.", metavar="True or False")

    (options, args) = parser.parse_args()
    execute(options)

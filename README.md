Test
This is the most recent version of 3DFSC, by Philip Baldwin, Yong Zi Tan, and Dmitry Lyumkis.

Installation: "conda env create -f environment.yml"

3DFSC is compatible and tested with Python 3.5+

Command line execute of the script is possible with the command "./run_fsc" from the top-most directory (tested in both bash and csh).

3DFSC is also accessible programmaticaly, for example:

import ThreeDFSC.ThreeDFSC_Start as run_fsc

run_rsc.execite(options)


...where "options" is an opt-parser object containing the parameters of the program. Thanks,

--CJN

# Anisotropy
These are programs that deal with anisotropy (both resolution and sampling) 


 Author: Philip Baldwin, March 2017
 Please contact me at pbaldwin@nysbc.org


# ThreeDFSC_ReleaseMar2017.py

 A conical resolution program 

Usage: ThreeDFSC_ReleaseMar2017.py

                HalfMap1.hdf    HalfMap2.hdf   OutputStringLabel   AngstromsPerPixel  
                
DeltaTheta (the conical patch parameter) is hard coded to 20 degrees


OutputStringLabel is a string that is used to create the output directory
which is 'Results'+OutputStringLabel

Within that directory is the main output which is ResEM+OutputStringLabel+'Out.hdf' 
  which is the 3DFSC volume


Best is if this program comes with the anaconda package (2 or 3)

 which corresponds to Python2.7 or 3.5
 
This requires the existence of numba, but not numbapro

For more, see README_3DFSC.md




# CreateSampVolPart1ReleaseMar2017.py


Usage:
CreateSampVolPart1ReleaseMar2017.py ...

         StarFileDir StarFile  ResFileDir RMax 
           OutputStringLabel NumberToUse CSym CorD


This program takes a cryosparc csv file or a Relion Star file
 
 from the StarFileDir. This gives a set of Euler Angles,
 
  or equivalently projection directions. 
  
   The directions are symmetrized by CSym (the axial 
    
    symmetry) and CorD (whether it is C or D). 
   
    A Sampling Volume is created of size N^3 (where N =2RMax+1).
  
    It is written to the same directory [ResFileDir] as that for the conical FSC outputs,
  
      using OutputStringLabel to label the Sampling Volume Name.

 It is 'SampVolBT'+OutputStringLabel +'RMax'+str(RMax)+'.hdf'

The StarFileDir may be  different from the ResFileDir.

 Typically the ResFileDir is constructed from the same OutputStringLabel.


For more, see README_Sampling.md


   


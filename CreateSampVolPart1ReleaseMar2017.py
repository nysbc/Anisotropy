#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 01:12:08 2016

@author: pbaldwin
"""
#%%
import sys
from sys import argv
import time
import pandas as pd;
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import shutil
#from time import *
import csv
import h5py

#%%

def ReadStarOrCryoSparcFile(StarFile):
    #fH = open('HA-T30_data.star','r')
    #fH = open('cryosparc_exp000039_000.csv','r')

    fH= open(StarFile)
    All=fH.readlines();
    fH.close();
    
    StarFileSplit=StarFile.split('.')
    if StarFileSplit[1]=='star':
        Found=0;
        Count=-1;
        CountNumberOfColumns=0;
        ColNames=[];
        while 1:
            Count+=1;
            AllNow=All[Count]
            if len(AllNow)<4: continue
            AllFirst4= AllNow[:4]
            if AllFirst4 != '_rln':
                if Found==1: break
                continue
            Found=1;
            CountNumberOfColumns +=1;
            ColumnNameTemp=AllNow[4:-1]
            ColumnNameParts = ColumnNameTemp.split('#');
            ColumnNumber= int(ColumnNameParts[1]);
            ColumnName =''.join(ColumnNameParts[0].split())
            ColNames.append(ColumnName)
        
        #ColNames=['Voltage','DefocusU','DefocusV','DefocusAngle','SphericalAberration', 
        #  'DetectorPixelSize', 'CtfFigureOfMerit', 'Magnification', 'AmplitudeContrast', 'ImageName',
        #  'CoordinateX','CoordinateY', 'NormCorrection','MicrographName', 'GroupName',
        #  'GroupNumber', 'OriginX',  'OriginY', 'AngleRot',  'AngleTilt',
        #  'AnglePsi', 'ClassNumber', 'LogLikeliContribution', 'RandomSubset', 'OriginalParticleName', 
        # 'NrOfSignificantSamples', 'NrOfFrames', 'MaxValueProbDistribution']
        sColNames= len(ColNames)#=28
    
        #ColNames=['Voltage','DefocusU','DefocusV','DefocusAngle','SphericalAberration', 
        #  'DetectorPixelSize','Magnification', 'AmplitudeContrast', 'ImageName',
        # 'MicrographName', 'GroupName', 'GroupNumber', 'AngleRot',  'AngleTilt','AnglePsi', 
        #  'OriginX',  'OriginY', 'ClassNumber', 'NormCorrection',
        #  'RandomSubset',   'LogLikeliContribution',   'MaxValueProbDistribution','NrOfSignificantSamples']
        #  sColNames= len(ColNames)#=23
        StarFileData     = pd.read_csv(StarFile  , header=None,skiprows=Count,names=ColNames,delim_whitespace=True);
        return StarFileData
    if StarFileSplit[1]=='csv':# Its from cryosparc
        Found=0;
        Count=-1;
        CountNumberOfColumns=0;
        ColNames=[];
        while 1:
            Count+=1;
            AllNow=All[Count]
            AllFirst = AllNow[:-1]
            if AllFirst=='_header':
                break
        ColLine= All[Count+1]
        ColLine= ColLine[:-1]
        ColNames=ColLine.split(',')
        sColNames=len(ColNames);
        for jColName in range(sColNames):
            ColNameNowParts= ColNames[jColName].split('.');
            if ColNameNowParts[0]=='ctf_params':
                ColNames[jColName]= 'ctf_'+ColNameNowParts[1];
            if ColNameNowParts[0]=='alignments':
                if len(ColNameNowParts)==3:
                    ColNames[jColName]='align_'+ColNameNowParts[2];
                if len(ColNameNowParts)==4:
                    ColNames[jColName]='align_'+ColNameNowParts[2]+'_'+ColNameNowParts[3];
        StarFileData     = pd.read_csv(StarFile  , header=None,skiprows=Count+4,names=ColNames);
        return StarFileData




#%%    
from numba import autojit
@autojit    
def AnglesToVecsCD(AngleTilts,AngleRots,CSym,DSym):#DSym=1 or 2
    sAngleTilts=len(AngleTilts);
    Skip= DSym*CSym;
    NVecs = np.zeros([sAngleTilts*Skip,3])
    for jAngle in range(sAngleTilts):
        AngleRotNow0  = AngleRots[jAngle];
        AngleTiltNow = AngleTilts[jAngle];
        for jAngSym in range(CSym):
            jAngAdd = jAngSym*360.0/CSym;
            AngleRotNow  = AngleRotNow0+jAngAdd;
            for jDSym in range(DSym):
                if jDSym==1: AngleTiltNow = 180.0- AngleTiltNow;
                c1= np.cos(AngleRotNow*np.pi/180.0)
                s1= np.sin(AngleRotNow*np.pi/180.0)
                cy= np.cos(AngleTiltNow*np.pi/180.0)
                sy= np.sin(AngleTiltNow*np.pi/180.0)
                nx= c1*sy;
                ny= s1*sy;
                nz= cy;
                if np.isnan(nx): continue
                NVecs[jAngle*Skip+jAngSym*DSym+jDSym,:]=[nx,ny,nz]       
    return NVecs
    

#%%

from numba import autojit
@autojit#([float[:],float[:],int,int])   
def AnglesToVecsCDv2(AngleTilts,AngleRots,CSym,DSym):#DSym=1 or 2
    sAngleTilts=len(AngleTilts);
    Skip= DSym*CSym;
    NVecs = np.zeros([sAngleTilts*Skip,3])
    for jAngleTot in range(sAngleTilts*Skip):
        jAngle=jAngleTot//Skip;
        # get AngleRot
        AngleRotNow0  = AngleRots[jAngle];
        jAngSym = (jAngleTot%Skip)//DSym
        jAngAdd = jAngSym*360.0/CSym;
        AngleRotNow  = AngleRotNow0+jAngAdd;
        # get AngleTilt
        jDSym   = (jAngleTot%DSym)
        AngleTiltNow = AngleTilts[jAngle];
        if jDSym==1: AngleTiltNow = 180.0- AngleTiltNow;
        # create nhat
        c1= np.cos(AngleRotNow*np.pi/180.0)
        s1= np.sin(AngleRotNow*np.pi/180.0)
        cy= np.cos(AngleTiltNow*np.pi/180.0)
        sy= np.sin(AngleTiltNow*np.pi/180.0)
        nx= c1*sy;
        ny= s1*sy;
        nz= cy;
        if np.isnan(nx): continue
        NVecs[jAngleTot,:]=[nx,ny,nz]       
    return NVecs
    

#%%

import numpy as np;
from numba import autojit

@autojit
def SampFuncFromProjN(Mid, NVecs):
    N=2*Mid+1;    
    sNVecs = len(NVecs);
    print(sNVecs)
    SampVol = np.zeros(N*N*N, dtype=int);
    LattSit = np.zeros([N*N*N,3], dtype=int);
    #MagnSit = np.zeros([N*N*N],dtype=float);
  
    Count=-1;
    for kx in range(-Mid,Mid+1):
        k2x=kx*kx;
        for ky in range(-Mid,Mid+1):
            k2y=k2x+ky*ky;
            for kz in range(-Mid,Mid+1):
                k2=k2y+kz*kz;
                Count+=1;
                LattSit[Count,:] = [kx,ky,kz];
                #MagnSit[Count]   = np.sqrt(k2);
               
    #SampVol = np.zeros([N,N,N]);

    print('Total sites to do is %g which is N^3. \n'%(N*N*N))
    #print('Total sites to do is %g which is N^3. '+'The number of vectors is %g\n'%(N*N*N, sNVecs))
    for jCount in range(N*N*N):
        [kx,ky,kz]= LattSit[jCount,:]
        if ((jCount%2000)==0): print(jCount);
        for iN in range(sNVecs):
            [nx,ny,nz]= NVecs[iN];
            dotProd  = (kx*nx+ky*ny+kz*nz);
            if abs(dotProd) <= 0.5:
                SampVol[jCount] +=1;
#                        SampVol[Mid+ix][Mid+iy][Mid+iz] += 1;
#                        SampVol[Mid-ix][Mid-iy][Mid-iz] += 1;
    #SampVol =SampVol.reshape(N,N,N)
    return [SampVol,LattSit]

#%%     Section 0,   Read in information and set up file names

# CreateConicalSampVolDec2016Part1.py  StarFileDir  StarFile 
#  ResFileDirectory       OutputStringLabel    NumberToUse  CSym   DSym
# Creates SampVolVec in the Results Directory

if 0:
    #StarFileDir =  '/gpfs/appion/eeng/forPhil/spotiton'
    #StarFileDir =  '/home/pbaldwin/MarinaStuff/forPhil/spotiton'
    #StarFileDir =  '/data1/pbaldwin/ptcls150K'
    StarFileDir =  '/data1/pbaldwin/Desktop/FromYZ/FSC/92_ForPhilEulerAngles'
    StarFileDir =  '/data1/pbaldwin/Desktop/FromYZ/FSC/HighFRET/zprWvRKoDFgYDLZME'
    StarFileDir =  '/data1/pbaldwin/Desktop/FromYZ/FSC/GDH/16oct28e'
    StarFileDir =  '/data1/pbaldwin/Desktop/FromYZ/FSC/GDH/17jan12d'
    StarFileDir =  '/data1/pbaldwin/eeng/16oct19jRelion/job030'
    StarFile    =  'run_data.star'
    ResFileDir =StarFileDir;
    
StarFileDir = argv[1]
os.chdir(StarFileDir);
StarFile= argv[2]; 

ResFileDir = argv[3]  ;     # This is where the half maps are, and where the output will be placed 
if 0:
    OutputStringLabel= 'FRET'; 
    OutputStringLabel= 'GDH_16oct28e'; 
    OutputStringLabel= 'GDH_17jan12d'; 
    OutputStringLabel= 'ProteaMar072017'; 



RMax   = int(argv[4]);# 71
OutputStringLabel= argv[5]; # This will access all the files from the FSC side 

NumberToUse = int(argv[6]);
CSym = int(argv[7]);# This is the rotational symmetry around z
CorD = int(argv[8]);# This is 1 or 2



ResultsDir      = 'Results'+OutputStringLabel+'/';
NVecsOut        = 'NVecs'+OutputStringLabel;

ResEMOutHDF_FN  =  ResultsDir+'ResEM'+OutputStringLabel+'Out.hdf'

dthetaInDegrees =20;
dthetaInRadians = dthetaInDegrees*np.pi/180.0;
Thresh = np.cos(dthetaInRadians)   ;
fractionOfTheSphereAveraged = (1-Thresh)/2;
# Now, deltaTheta takes up a cone which has area
#    2 pi * (1-cos(deltaTheta))


# Other outputs that we do not use in our program

FTOut =  ResultsDir+'FTPlot'+OutputStringLabel;
PlotsOut= ResultsDir+'Plots'+OutputStringLabel;
fNResRoot=ResEMOutHDF_FN[0:-4]
fN_csv_out= fNResRoot+'.csv';

ResNumOutMRC= fNResRoot+'Num.mrc';# The numerator which is the cross product
ResDenOutMRC= fNResRoot+'Den.mrc';# The fn for denominator, which is normalization

resultAveOut = fNResRoot+'globalFSC.csv';


# sys.exit()



#%%          Section 1.   Read in Star File
#    Set threshhold for ang averaging,  Read in star file, 

startTime = time.time()


if 0:
    StarFile    =  'shiny_K10class7_9_2_6_10.star'
    StarFile    =  'HA-T40_data.star'
    StarFile    =  'cryosparc_exp000039_000.csv'
    StarFile    =  'run_data.star'

# Should have DPRFile= 'DPR.mrc'; FSCFile= 'ResEMR.mrc';  FTFile= 'FT.mrc'

#PtclClassNames = (StarFileData.ImageName);
#PtclClassNamesList = list(set(PtclClassNames));
#StarFileAttribs=StarFileAttribs[:29]


StarFileData   =  ReadStarOrCryoSparcFile(StarFile) # 13028 rows x 28 columns

NumParticles     = len(StarFileData);

       
deltaTime =time.time()-startTime;

print("Read in Star File created in %f seconds for size nx=%g " % (deltaTime, NumParticles))

#%%     Section 2.    Convert angles to Projection Directions



startTime = time.time()


if 0:
    StarFileData.to_csv('StarFile' +StarFile[:-5] + '.csv')
    
try:
    AngleRots  = np.array(StarFileData.AngleRot.tolist())
    AngleTilts = np.array(StarFileData.AngleTilt.tolist())
    AnglePsis  = np.array(StarFileData.AnglePsi.tolist())
except:
    AngleRots  = np.array(StarFileData.align_r_0.tolist())*180/np.pi
    AngleTilts = np.array(StarFileData.align_r_1.tolist())*180/np.pi
    AnglePsis  = np.array(StarFileData.align_r_2.tolist())*180/np.pi


NVecs0= AnglesToVecsCD(AngleTilts,AngleRots,CSym,CorD)# CorD: 1 means C, 2 means D
sNVecs0=len(NVecs0)


NVecs= NVecs0
sNVecs = len(NVecs)

  
AA= np.random.choice(sNVecs0,NumberToUse)
NVecs= NVecs0[AA,:]
sNVecs = len(NVecs)

AngleTiltsFinal= np.arccos(NVecs0[:,2])*180/np.pi;#   Change Back to NVecs
AngleRotsFinal = np.arctan2(NVecs0[:,0],NVecs0[:,1])*180/np.pi;

       
deltaTime =time.time()-startTime;

print("AnglesToVecsCD created in %f seconds for size sNVecs0=%g " % (deltaTime,sNVecs0))

print(ResFileDir, StarFileDir, NVecsOut)
#sys.exit()

os.chdir(ResFileDir)
np.save(NVecsOut,NVecs0)

#%%     Section 1,    Apply    SampFuncFromProjN

Mid =RMax;
N=2*Mid+1;
startTime = time.time()
 
#NVecs[0]= [1,0,0]

[SampVol,LattSit] = SampFuncFromProjN(Mid, NVecs);

print(np.sum(SampVol))

#  SampVol is 580970312 should be divisible by number of Projections
#  sNVecs is 129220 = 9230*14

deltaTime =time.time()-startTime;

print("SampVol created in %f seconds for size N=%g, number of NVecs=%g "
      % (deltaTime,2*Mid+1,sNVecs))


SampVolB =np.zeros([N,N,N],dtype=int);
for jCount in range(N*N*N):
    [jx,jy,jz]=LattSit[jCount,:];
    SampVolB[jx+Mid,jy+Mid,jz+Mid]= SampVol[jCount];


os.chdir(ResFileDir)
h5f_write = h5py.File('SampVolBT'+OutputStringLabel +'Mid'+str(Mid)+'.hdf','w')
h5f_write.create_dataset('MDF/images/0/image',data=SampVolB.T)
# <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
h5f_write.close()


os.getcwd()


sys.exit()


#%%  Assessment of Angles

if 0:
    
    TitleStringAngsJIT = 'HistogramOfTilts';
    f= plt.figure(TitleStringAngsJIT)
    n,bins, patches = plt.hist(AngleTilts,60)# AngleRots goes from -60 to 60
    f.suptitle(TitleStringAngsJIT)

    f.savefig(OutputStringLabel+'AngleTilts.jpg');
    
    TitleStringAngsJIT = 'HistogramOfRots';
    f= plt.figure(TitleStringAngsJIT)
    n,bins, patches = plt.hist(AngleRots,60)# AngleRots goes from -60 to 60
    f.suptitle(TitleStringAngsJIT)
    f.savefig(OutputStringLabel+'AngleRots.jpg');
    
    TitleStringAngsJIT = 'HistogramOfPsis';
    f= plt.figure(TitleStringAngsJIT)
    n,bins, patches = plt.hist(AnglePsis,60)# AngleRots goes from -60 to 60
    f.suptitle(TitleStringAngsJIT)
    f.savefig(OutputStringLabel+'AnglePsis.jpg');
    
    TitleStringAngsJIT = 'HistogramOfTilts';
    f= plt.figure(TitleStringAngsJIT)
    n,bins, patches = plt.hist(AngleTiltsFinal,60)# AngleRots goes from -60 to 60
    f.suptitle(TitleStringAngsJIT)
    f.savefig(OutputStringLabel+'TiltsFinal.jpg');

    
    TitleStringAngsJIT = 'HistogramOfRots';
    f= plt.figure(TitleStringAngsJIT)
    n,bins, patches = plt.hist(AngleRotsFinal,60)# AngleRots goes from -60 to 60
    f.suptitle(TitleStringAngsJIT)
    
    f.savefig(OutputStringLabel+'RotsFinal.jpg');
    

#%%


    h5f_HalfMap1 =h5py.File('../run_half1_class001_unfil.hdf','r')
    dataSetNow=h5f_HalfMap1['MDF/images/0/image']
    f=np.array(dataSetNow)
    #
    h5f_HalfMap1 =h5py.File('../run_half2_class001_unfil.hdf','r')
    dataSetNow=h5f_HalfMap2['MDF/images/0/image']
    g=np.array(dataSetNow)
    #
    h5f_HalfMap1.close()
    h5f_HalfMap2.close()

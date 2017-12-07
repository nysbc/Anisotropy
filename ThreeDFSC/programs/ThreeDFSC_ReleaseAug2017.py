#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Anaconda 3 required
# This program is  ThreeDFSC_ReleaseFeb2017.py
# A conical resolution program written by P. R. Baldwin in November 2016
# Downloaded from https://github.com/nysbc/Anisotropy
# ThreeDFSC_ReleaseJul2017.py
#                  HalfMap1.mrc      HalfMap2.mrc     OutputStringLabel     A/pixel DeltaTheta                      
#
# Creates    ResEMOutresultAve+OutputStringLabel.csv (which is the usual FSC)
#            ResEMOut+OutputStringLabel.hdf which is the 3D FSC file ResEMR (the real part of the cccs) 
#            Plots+OutputStringLabel.jpg which is the slices along x, y, z 
#
# This requires the existence of numba, but not numbapro
#
# The functions need to be kept separated so that the precompiler can
# notice the @autojit decorations and  precompile the code,
#
# Uses mrcfile 1.0.0 by Colin Palmer (https://github.com/ccpem/mrcfile)
# 
# For Phil: Line 547 for jSurf in range(Num2Surf+1): ### Fixed the problem!

from sys import argv
import csv
import time
import os
import sys
import numpy as np
from math import *
import math
from numba import *
from numba import autojit, prange, cuda
import numba
import cuda_kernels
import copy
import h5py

import matplotlib
import matplotlib.pyplot as plt

import mrcfile

## Progress bar, adapted from https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a

def print_progress(iteration, total, prefix='', suffix='', decimals=1):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration    - Required    : current iteration (Int)
        total        - Required    : total iterations (Int)
        prefix        - Optional    : prefix string (Str)
        suffix        - Optional    : suffix string (Str)
        decimals    - Optional    : positive number of decimals in percent complete (Int)
        bar_length    - Optional    : character length of bar (Int)
    """
    
    rows, columns = os.popen('stty size', 'r').read().split()
    bar_length = int(float(columns)/2)
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total))) ## adjusted base on window size
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\x1b[2K\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


## Added to mute the print statements

def blockPrint():
    #sys.stdout = open(os.devnull, 'w')
    pass

def enablePrint():
    #sys.stdout = sys.__stdout__
    pass

#%%      Section -1 Function Definitions

@numba.autojit
def ExtractAxes(f):
    
    [nx,ny,nz]    = f.shape;

    nx2=nx//2;
    ny2=ny//2;
    nz2=nz//2;

    xf=np.zeros(nx2+1)
    yf=np.zeros(ny2+1)
    zf=np.zeros(nz2+1)
    
    for ix in prange(nx2):
        xf[ix]=f[ix+nx2,ny2,nz2];
        
    for iy in prange(ny2):
        yf[iy]=f[nx2,iy+ny2,nz2];
        
    for iz in prange(nz2):
        zf[iz]=f[nx2,ny2,iz+nz2];

    return [xf,yf,zf]

#%%     Section -1 Function Definitions

@numba.autojit
def AddAxes(f,jDir, Val):
    
    [nx,ny,nz]    = f.shape;

    fOut = f.copy()
    nx2=nx//2;
    ny2=ny//2;
    nz2=nz//2;

    # np.max(f);# f[nx2-1,ny2-1,nz2-1];

    if jDir==0:
        fOut[:      ,ny2-1,nz2-1]=Val;
        
    if jDir==1:
        fOut[nx2-1, :    ,nz2-1]=Val;
        
    if jDir==2:
        fOut[nx2-1,ny2-1, :      ]=Val;
        
    return fOut

#%%     Section -1 Function Definitions

@numba.autojit
def ZeroPad(nx,ny,nz,fT,gT):
    fp=np.zeros([nx+2,ny,nz]);
    gp=np.zeros([nx+2,ny,nz]);

    for ix in prange(nx):
        for iy in range(ny):
            for iz in range(nz):
                fp[ix,iy,iz]=fT[ix,iy,iz]
                gp[ix,iy,iz]=gT[ix,iy,iz]
    return [fp,gp]

#Functions read in 0.078624 seconds for size nx=256 
#%%     Section -1 Function Definitions

@numba.autojit
def FFTArray2Real(nx,ny,nz,F):

    d1=np.zeros([nx+4,ny,nz]);# 36,32,32

    for ix in prange(0,nx+3,2):
        ixover2= ix//2;
        for iy in range(ny):
            for iz in range(nz):
                FNow=F[ixover2,iy,iz];
                d1[ix  ][iy][iz]=FNow.real;
                d1[ix+1][iy][iz]=FNow.imag;
    return d1


#%%     Section -1 Function Definitions %     Create FT outputs

@numba.autojit
def CreateFTLikeOutputs(inc,nx,ny,nz,ToBeAveraged,nx2,ny2,nz2,dx2,dy2,dz2):# created from CreateFSCOutputs

    ret = np.zeros(inc+1)
    lr = np.zeros(inc+1)
    #Count=0;
    for iz in prange(nz):
        kz=iz;
        if (iz>nz2): kz=iz-nz;# This is the actual value kz, can be neg
        argz= float(kz*kz)*dz2;
        for iy in range(ny):
            ky=iy;          # This is the actual value ky, can be neg
            if (iy>ny2): ky=iy-ny;
            argy= argz+float(ky*ky)*dy2;
            for ix in range(0,nx,2):
                if ( (ix==0) & (kz<0)& (ky<0)):     continue;
                argx = 0.5*np.sqrt(argy + float(ix*ix)*0.25*dx2);
                r=int(round(inc*2*argx));
                if (r <= inc):
                    #ii = ix + (iy    + iz * ny)* lsd2;
                    ret[r] += ToBeAveraged[ix  ,iy,iz];
                    lr[r]  += 2.0;# Number of Values

    return [ret,lr]

#%%     Section -1 Function Definitions %     Create FSC outputs for Pawel program

@numba.autojit
def CreateFSCOutputs(inc,nx,ny,nz,d1,d2,nx2,ny2,nz2,dx2,dy2,dz2):

    ret = np.zeros(inc+1)
    n1    = np.zeros(inc+1)
    n2    = np.zeros(inc+1)
    lr = np.zeros(inc+1)
    #Count=0;
    for iz in prange(nz):
        kz=iz;
        if (iz>nz2): kz=iz-nz;# This is the actual value kz, can be neg
        argz= float(kz*kz)*dz2;
        for iy in range(ny):
            ky=iy;          # This is the actual value ky, can be neg
            if (iy>ny2): ky=iy-ny;
            argy= argz+float(ky*ky)*dy2;
            for ix in range(0,nx,2):
                if ( (ix==0) & (kz<0)& (ky<0)):     continue;
                argx = 0.5*np.sqrt(argy + float(ix*ix)*0.25*dx2);
                r=int(round(inc*2*argx));
                if (r <= inc):
                    #ii = ix + (iy    + iz * ny)* lsd2;
                    ret[r] += d1[ix     ,iy,iz] * d2[ix,iy,iz] ;
                    ret[r] += d1[ix+1,iy,iz] * d2[ix+1,iy,iz] ;
                    n1[r]  += d1[ix     ,iy,iz] * d1[ix,iy,iz];
                    n1[r]  += d1[ix+1,iy,iz] * d1[ix+1,iy,iz];
                    n2[r]  += d2[ix     ,iy,iz] * d2[ix,iy,iz];
                    n2[r]  += d2[ix+1,iy,iz] * d2[ix+1,iy,iz];
                    lr[r]  += 2.0;# Number of Values

    return [ret,n1,n2,lr]


#%%     Section -1 Function Definitions 
#            Find values of product at individual points, organized on shells in FS

@numba.autojit
def createFSCarrays(nx,ny,nz,lsd2,lr,inc,dx2,dy2,dz2,d1,d2,nx2,ny2,nz2):

    lrMaxOver2= int(lr[-1]//2);
            
    kXofR    = np.zeros([inc+1,lrMaxOver2],dtype=int)
    kYofR    = np.zeros([inc+1,lrMaxOver2],dtype=int)
    kZofR    = np.zeros([inc+1,lrMaxOver2],dtype=int)
    retofRR = np.zeros([inc+1,lrMaxOver2])
    retofRI = np.zeros([inc+1,lrMaxOver2])
    n1ofR    = np.zeros([inc+1,lrMaxOver2])
    n2ofR    = np.zeros([inc+1,lrMaxOver2])
    
    NumAtEachR = np.zeros(inc+1,dtype=int);
#                
    rmax=0;
    for iz in prange(nz):
        kz=iz;
        if (iz>nz2):
            kz=iz-nz;# This is the actual value kz, can be neg
        argz= float(kz*kz)*dz2;
        for iy in range(ny):
            ky=iy;# This is the actual value ky, can be neg
            if (iy>ny2):
                ky=iy-ny;
            argy= argz+float(ky*ky)*dy2;
            for ix in range(0,nx,2):
                if ( (ix==0) & (ky<0)): continue
                if ( (ix==0) & (ky==0)& (kz<0)): continue
                kx=float(ix)/2.0;
                argx = 0.5*np.sqrt(argy + kx*kx*dx2);
                r=int(round(inc*2*argx))
                if(r <= inc):
                    NumAtEachR[r]+=1;
                    LastInd = NumAtEachR[r]-1;
                    #ii = ix + (iy    + iz * ny)* lsd2;
                    kXofR[r,LastInd] =int(round(kx));
                    kYofR[r,LastInd] =ky;
                    kZofR[r,LastInd] =kz;
                    retrRNow  = d1[ix  ,iy,iz] * d2[ix    ,iy,iz];
                    retrRNow += d1[ix+1,iy,iz] * d2[ix+1,iy,iz];
                    retrINow  = d1[ix  ,iy,iz] * d2[ix+1,iy,iz];
                    retrINow -= d1[ix+1,iy,iz] * d2[ix    ,iy,iz];
                    n1rNow      = d1[ix  ,iy,iz] * d1[ix    ,iy,iz];
                    n1rNow     += d1[ix+1,iy,iz] * d1[ix+1,iy,iz];
                    n2rNow      = d2[ix  ,iy,iz] * d2[ix    ,iy,iz];
                    n2rNow     += d2[ix+1,iy,iz] * d2[ix+1,iy,iz];
                    #retofRR[r]=retNowR;  retofRI[r]=retNowI;  
                    #n1ofR[r]=n1Now;   n2ofR[r]=n2Now;
                    retofRR[r, LastInd]=retrRNow;
                    retofRI[r, LastInd]=retrINow;
                    n1ofR[r, LastInd]=n1rNow;    
                    n2ofR[r, LastInd]=n2rNow;

                    if r>rmax: rmax=r;
    print(rmax)
    return [kXofR, kYofR, kZofR,retofRR,retofRI,n1ofR,n2ofR, NumAtEachR]

#                 retrR    = d1.get_value_at(ix  ,iy,iz) * d2.get_value_at(ix    ,iy,iz) ;
#                 retrR += d1.get_value_at(ix+1,iy,iz) * d2.get_value_at(ix+1,iy,iz) ;
#                 retrI    = d1.get_value_at(ix  ,iy,iz) * d2.get_value_at(ix+1,iy,iz) ;
#                 retrI -= d1.get_value_at(ix+1,iy,iz) * d2.get_value_at(ix    ,iy,iz) ;
#                 n1r    = d1.get_value_at(ix  ,iy,iz) * d1.get_value_at(ix    ,iy,iz);
#                 n1r   += d1.get_value_at(ix+1,iy,iz) * d1.get_value_at(ix+1,iy,iz);
#                 n2r    = d2.get_value_at(ix  ,iy,iz) * d2.get_value_at(ix    ,iy,iz);


#                 n2r   += d2.get_value_at(ix+1,iy,iz) * d2.get_value_at(ix+1,iy,iz);
#

#%%     Section -1 Function Definitions 
#            Find values of product at individual points, organized on shells in FS
@numba.autojit
def createFTarrays(nx,ny,nz,lsd2,lr,inc,dx2,dy2,dz2,dcH,dFPower,nx2,ny2,nz2):

    lrMaxOver2= int(lr[-1]//2);
            
    kXofR    = np.zeros([inc+1,lrMaxOver2],dtype=int)
    kYofR    = np.zeros([inc+1,lrMaxOver2],dtype=int)
    kZofR    = np.zeros([inc+1,lrMaxOver2],dtype=int)
    retcH    = np.zeros([inc+1,lrMaxOver2])
    retFT    = np.zeros([inc+1,lrMaxOver2])
    n12ofR    = np.zeros([inc+1,lrMaxOver2])
    
    NumAtEachR = np.zeros(inc+1,dtype=int);
#                
    rmax=0;
    for iz in prange(nz):
        kz=iz;
        if (iz>nz2):
            kz=iz-nz;# This is the actual value kz, can be neg
        argz= float(kz*kz)*dz2;
        for iy in range(ny):
            ky=iy;# This is the actual value ky, can be neg
            if (iy>ny2):
                ky=iy-ny;
            argy= argz+float(ky*ky)*dy2;
            for ix in range(0,nx,2):
                if ( (ix==0) & (ky<0)): continue
                if ( (ix==0) & (ky==0)& (kz<0)): continue
                kx=float(ix)/2.0;
                argx = 0.5*np.sqrt(argy + kx*kx*dx2);
                r=int(round(inc*2*argx))
                if(r <= inc):
                    NumAtEachR[r]+=1;
                    LastInd = NumAtEachR[r]-1;
                    #ii = ix + (iy    + iz * ny)* lsd2;
                    kXofR[r,LastInd] =int(round(kx));
                    kYofR[r,LastInd] =ky;
                    kZofR[r,LastInd] =kz;
                    retcHNow  = dcH[ix    ,iy,iz];
                    retFTNow  = dFPower[ix    ,iy,iz];
                    n12rNow      = 1;
                    #retofRR[r]=retNowR;  retofRI[r]=retNowI;  
                    #n1ofR[r]=n1Now;   n2ofR[r]=n2Now;
                    retcH[r, LastInd]=retcHNow;
                    retFT[r, LastInd]=retFTNow;
                    n12ofR[r, LastInd]=n12rNow;      

                    if r>rmax: rmax=r;
    print(rmax)
    return [kXofR, kYofR, kZofR,retcH,retFT,n12ofR]

#                 retrR    = d1.get_value_at(ix  ,iy,iz) * d2.get_value_at(ix    ,iy,iz) ;
#                 retrR += d1.get_value_at(ix+1,iy,iz) * d2.get_value_at(ix+1,iy,iz) ;
#                 retrI    = d1.get_value_at(ix  ,iy,iz) * d2.get_value_at(ix+1,iy,iz) ;
#                 retrI -= d1.get_value_at(ix+1,iy,iz) * d2.get_value_at(ix    ,iy,iz) ;
#                 n1r    = d1.get_value_at(ix  ,iy,iz) * d1.get_value_at(ix    ,iy,iz);
#                 n1r   += d1.get_value_at(ix+1,iy,iz) * d1.get_value_at(ix+1,iy,iz);
#                 n2r    = d2.get_value_at(ix  ,iy,iz) * d2.get_value_at(ix    ,iy,iz);


#                 n2r   += d2.get_value_at(ix+1,iy,iz) * d2.get_value_at(ix+1,iy,iz);
#



#%%     Section -1 Function Definitions %    For a given shell, this function returns whether a pair are close or not

def AveragesOnShellsInnerLogicKernelCuda(kXNow,kYNow,kZNow,\
                                         kXofR_global_mem,\
                                         kYofR_global_mem,\
                                         kZofR_global_mem,\
                                         retofRR_global_mem,\
                                         retofRI_global_mem,\
                                         n1ofR_global_mem,\
                                         n2ofR_global_mem,\
                                         NumOnSurf,\
                                         Thresh,\
                                         Start,\
                                         End,\
                                         r):

    stream = cuda.stream()
    # Copy data to gpu device
    #kXNow_global_mem = cuda.to_device(kXNow,stream=stream)
    #kYNow_global_mem = cuda.to_device(kYNow,stream=stream)
    #kZNow_global_mem = cuda.to_device(kZNow,stream=stream)

    # Allocate memory on gpu for temporary and output data
    NumAtROutPre_global_mem = cuda.device_array((NumOnSurf,End-Start))
    Prod11_global_mem = cuda.device_array(NumOnSurf,stream=stream,dtype=np.float32)

    # Set threads per block and blocks per grid
    threadsperblock = (64,1,1)
    blockspergrid_x = int(math.ceil(kXofR_global_mem[r][:NumOnSurf].shape[0] / threadsperblock[0]))
    blockspergrid = (blockspergrid_x,1,1)
    start_cuda = time.time()

    # Kernel 1, calculate Prod11
    cuda_kernels.cuda_calcProd11[blockspergrid, threadsperblock,stream](\
                                                                        kXofR_global_mem,\
                                                                        kYofR_global_mem,\
                                                                        kZofR_global_mem,\
                                                                        Prod11_global_mem,\
                                                                        NumOnSurf,\
                                                                        r)

    stream.synchronize()
    print("Time to complete Prod11 calculation is ",time.time() - start_cuda)
    Inner2_start = time.time()
    # Kernel 2, calculate Inner2
    cuda_kernels.cuda_calcInner2[blockspergrid, threadsperblock,stream](\
                                                                        kXofR_global_mem,\
                                                                        kYofR_global_mem,\
                                                                        kZofR_global_mem,\
                                                                        Prod11_global_mem,\
                                                                        NumAtROutPre_global_mem,\
                                                                        End,\
                                                                        Start,\
                                                                        Thresh,\
                                                                        NumOnSurf,\
                                                                        r)
    stream.synchronize()
    print("Time to complete Inner2 calculation is ",time.time() - Inner2_start)
    print("\nCUDA inner calculations completed in "+str(time.time() - start_cuda)+".")

    return NumAtROutPre_global_mem
#    NumAtROutPre = NumAtROutPre_global_mem.copy_to_host(stream=stream)
#    stream.synchronize()
#    end_cuda = time.time()
#    print("\nCUDA calculation completed in ",end_cuda - start_cuda,".")
#
#    return NumAtROutPre

@numba.autojit    
def AveragesOnShellsInnerLogicKernelnonCuda(kXNow,kYNow,kZNow,NumOnSurf,Thresh,Start, End):
#     NumAtROutPre = np.zeros(int(NumOnSurf*(NumOnSurf+1)/2),dtype=int)
#     NumAtROutPre = np.zeros([NumOnSurf,NumOnSurf],dtype=int)
#     NumAtROutPre = np.identity(NumOnSurf,dtype=int)
#     NumAtROutPre = np.zeros((NumOnSurf,End-Start), dtype=np.int)
#     print('Hello')

#    NumAtROutPre has dimensions NumOnSurf by End-Start
#    Each element NumAtROutPre[m,n]    denotes whether m is close to Start+n
#

    NumAtROutPre = np.zeros((NumOnSurf,End-Start), dtype=np.int)

    Thresh2=Thresh*Thresh
    #Count=0;
    for jSurf1 in prange(NumOnSurf):
        #retNow1RL =retofRR[r] 
        kX1=kXNow[jSurf1]; 
        kY1=kYNow[jSurf1]; 
        kZ1=kZNow[jSurf1];#      Single Values
        Prod11    = kX1*kX1+kY1*kY1+kZ1*kZ1;
        if Prod11==0: continue
        for jSurf2 in range(End-Start):# labels kX, etc
            #if jSurf1==jSurf2: continue
            #Count+=1;
            kX2=kXNow[jSurf2+Start];
            kY2=kYNow[jSurf2+Start];
            kZ2=kZNow[jSurf2+Start];#    Single Values
            Prod12    = kX1*kX2+kY1*kY2+kZ1*kZ2;
            Prod22    = kX2*kX2+kY2*kY2+kZ2*kZ2;
            if Prod22==0:continue
            Inner2    = Prod12*Prod12/(Prod11*Prod22);
            if Inner2>Thresh2:# Then angle is sufficiently small
                #NumAtROutPre[NumOnSurf*jSurf1 -(jSurf1+1)*jSurf1//2 +jSurf2 ]     = 1;
                NumAtROutPre[jSurf1,jSurf2]      = 1;
                #NumAtROutPre[jSurf2,jSurf1 ]    = 1;
                #N(X-1) - (X)(X-1)/2 + Y
                #print(r,jSurf1,jSurf2,retNow1L,retofR[r][jSurf2],retofROut[r][jSurf2],);
    return NumAtROutPre


    #print("sum, sum of NumAtROutPre = %g " %(np.sum(np.sum(NumAtROutPre,axis=0))));
    


def sumRowsCuda(\
                NumAtROutPre_global_mem,Start,End):

    sum_array_global_mem = cuda.device_array((End-Start))
    threadsperblock = (64,1,1)
    blockspergrid_x = int(math.ceil(NumAtROutPre_global_mem.shape[0]/threadsperblock[0]))
    blockspergrid = (blockspergrid_x,1,1)
    stream = cuda.stream()
    sum_rows_start = time.time()
    cuda_kernels.sum_rows[threadsperblock,blockspergrid,stream](\
        NumAtROutPre_global_mem,\
        sum_array_global_mem,Start,End)
    stream.synchronize()
    print("Time to complete sum_rows is ",time.time() - sum_rows_start)
    copy_qq_start = time.time()
    qq = sum_array_global_mem.copy_to_host(stream=stream)
    print("Time to copy qq to host is ",time.time() - copy_qq_start)
    return qq

def AveragesOnShellsInnerLogicCCuda(\
                                    retNowR_global_mem,\
                                    retNowI_global_mem,\
                                    n1ofR_global_mem,\
                                    n2ofR_global_mem,\
                                    NumAtROutPre_global_mem,\
                                    End,\
                                    Start,\
                                    NumOnSurf,\
                                    r):
    setup_start = time.time()
    threadsperblock = (64,1,1)
    blockspergrid_x = int(math.ceil(retNowR_global_mem[r][:NumOnSurf].shape[0]/threadsperblock[0]))
    blockspergrid = (blockspergrid_x,1,1)
    # set up stream
    stream = cuda.stream()
    print("Time to set up kernel is ",time.time() - setup_start)
    device_array_start = time.time()
    #reduced_global_mem = cuda.device_array((retNowR_global_mem[r][:NumOnSurf].shape[0],(End-Start)))
    reduced_global_mem = cuda.device_array((4,(End-Start)))
    print("Shape of reduced_global_mem is ",reduced_global_mem.shape)
    print("Time to create cuda.device_array is ",time.time() - device_array_start)
    filter_time = time.time()
    cuda_kernels.filter_and_sum[threadsperblock,blockspergrid,stream](\
        retNowR_global_mem,\
        retNowI_global_mem,\
        n1ofR_global_mem,\
        n2ofR_global_mem,\
        NumAtROutPre_global_mem,\
        reduced_global_mem,\
        End,\
        Start,\
        NumOnSurf,\
        r)
    stream.synchronize()
    print("Time to complete filter_and_sum is ",time.time() - filter_time)
    reduced_start = time.time()
    reduced = reduced_global_mem.copy_to_host(stream=stream)
    print("Time to transfer reduced to host is ",time.time() - reduced_start)

    return reduced




#%%     Section -1 Function Definitions 
@numba.autojit
def AveragesOnShellsInnerLogicC(retNowR,retNowI,n1Now, n2Now,Start, End ,NumAtROutPre):

#    print("retNowR dimensions: ",np.shape(retNowR))
#    print("retNowI dimensions: ",np.shape(retNowI))
#    print("n1Now dimensions: ",np.shape(n1Now))
#    print("n2Now dimensions: ",np.shape(n2Now))
#    print("Start dimensions: ",np.shape(Start))
#    print("End dimensions: ",np.shape(End))
#    print("NumAtROutPre dimensions: ",np.shape(NumAtROutPre))
    NumNow= End -Start;
    retofROutRPre = np.zeros(NumNow)
    retofROutIPre = np.zeros(NumNow)
    n1ofROutPre      = np.zeros(NumNow)
    n2ofROutPre      = np.zeros(NumNow)
    #NumAtROutPre  = np.zeros((NumOnSurf,End-Start), dtype=np.int)
    for jSurf1 in range(NumNow):

        MultVec=NumAtROutPre[:,jSurf1];# NumAtROutPre has shape 15871 by 7936
        GoodInds = np.where(MultVec)[0] # MultVec has shape 15871

        retofROutRPre[jSurf1] = np.sum(retNowR[GoodInds])
        retofROutIPre[jSurf1] = np.sum(retNowI[GoodInds])
        n1ofROutPre[jSurf1]      = np.sum(     n1Now[GoodInds])
        n2ofROutPre[jSurf1]      = np.sum(     n2Now[GoodInds])
        
            # Infer jSurf2 from NeighborList
    return [retofROutRPre, retofROutIPre, n1ofROutPre,n2ofROutPre]


#%%     Section -1 Function Definitions 
def AveragesOnShellsUsingLogicB(inc,retofRR,retofRI,n1ofR,n2ofR, kXofR,kYofR,kZofR, \
                                    NumAtEachR,Thresh, RMax):
    print('This loop will go to '+str(RMax)+'\n' )    
    NumAtEachRMax=NumAtEachR[-1];
    retofROutR = np.zeros([inc+1,NumAtEachRMax]); #retofRR.copy();# Real part of output
    retofROutI = np.zeros([inc+1,NumAtEachRMax]); #retofRI.copy();# Imag part of output
    n1ofROut   = np.zeros([inc+1,NumAtEachRMax]); #n1ofR.copy();
    n2ofROut   = np.zeros([inc+1,NumAtEachRMax]); #n2ofR.copy();
    NumAtROut  = np.zeros([inc+1,NumAtEachRMax]); #

    print("shape of retofROutR is ",np.shape(retofROutR))
    NumAtEachRMaxCuda= 15871;
    
    retofROutR[0,0] = retofRR[0,0];
    retofROutI[0,0] = retofRI[0,0];
    n1ofROut[0,0]    = n1ofR[0,0];
    n2ofROut[0,0]    = n2ofR[0,0];
    
    #blockdim=(8,8);
    #griddim=(8,8);

    # Load all data into GPU memory
    # Need to convert data to contiguous arrays
    allocation_start = time.time()
    kXofR_global_mem = cuda.to_device(np.ascontiguousarray(kXofR,dtype=np.float32))
    kYofR_global_mem = cuda.to_device(np.ascontiguousarray(kYofR,dtype=np.float32))
    kZofR_global_mem = cuda.to_device(np.ascontiguousarray(kZofR,dtype=np.float32))
    retofRR_global_mem = cuda.to_device(np.ascontiguousarray(retofRR,dtype=np.float32))
    retofRI_global_mem = cuda.to_device(np.ascontiguousarray(retofRI,dtype=np.float32))
    n1ofR_global_mem = cuda.to_device(np.ascontiguousarray(n1ofR,dtype=np.float32))
    n2ofR_global_mem = cuda.to_device(np.ascontiguousarray(n2ofR,dtype=np.float32))
    print("Time to allocation memory is ",time.time() - allocation_start)
    enablePrint()
    for r in range(1,RMax+1):#range(1,inc+1):
        #if r!=2: continue
        if ((r-1)%5)==0: print(r)
        NumOnSurf = int(NumAtEachR[r]);
        #LastInd = NumAtEachR[r]-1 ;
        kXNow    = kXofR[r][:NumOnSurf]; 
        kYNow    = kYofR[r][:NumOnSurf]; 
        kZNow    = kZofR[r][:NumOnSurf];#     Vectors
        retNowR = retofRR[r][:NumOnSurf];  
        retNowI = retofRI[r][:NumOnSurf]; 
        n1Now    = n1ofR[r][:NumOnSurf]; 
        n2Now    = n2ofR[r][:NumOnSurf];#   for given 


        ## Progress bar
        print_progress(r,RMax)
        ##
        
        NumLoops=1+int(NumOnSurf*NumOnSurf/NumAtEachRMaxCuda/NumAtEachRMaxCuda);# kicks in at r=50
        NumLoops = 2
        Stride=int(NumOnSurf/NumLoops);
        startTime = time.time()
        blockPrint()
        for jLoop in range(NumLoops):

            Start=jLoop*Stride;
            End= Start+Stride;
            if jLoop==(NumLoops-1):
                End = NumOnSurf;
            #print("jLoop,Start,End = %g,  %g  %g " %(jLoop,Start,End) )
            NumAtROutPre = np.zeros((NumOnSurf,End-Start), dtype=np.int)
            #print("NumAtROutPre.shape %g %g" %(NumAtROutPre.shape))
            InnerLogicCuda_start = time.time()
            NumAtROutPre_global_mem = AveragesOnShellsInnerLogicKernelCuda(kXNow,kYNow,kZNow,\
                                                                           kXofR_global_mem,\
                                                                           kYofR_global_mem,\
                                                                           kZofR_global_mem,\
                                                                           retofRR_global_mem,\
                                                                           retofRI_global_mem,\
                                                                           n1ofR_global_mem,\
                                                                           n2ofR_global_mem,\
                                                                           NumOnSurf,\
                                                                           Thresh,\
                                                                           Start,\
                                                                           End,\
                                                                           r);
            print("Time to complete InnerLogicKernelCuda is ",time.time()-InnerLogicCuda_start)

            #InnerLogicnonCuda_start = time.time()
            #NumAtROutPre = AveragesOnShellsInnerLogicKernelnonCuda(kXNow,kYNow,kZNow, NumOnSurf, Thresh,Start, End);
            #print("Time to complete InnerLogicKernelnonCuda is ",time.time()-InnerLogicnonCuda_start)

            deltaTimeN =time.time()-startTime;
            #print("sum, sum of NumAtROutPre = %g " %(np.sum(np.sum(NumAtROutPre,axis=0))));
            #print("how many zeros of NumAtROutPre = %g " %(len(np.where(NumAtROutPre==0)[0] ) ) );
            startTime = time.time()

            LogicCCuda_start = startTime
            #cuda.synchronize()

            reduced = AveragesOnShellsInnerLogicCCuda(\
                                                      retofRR_global_mem,\
                                                      retofRI_global_mem,\
                                                      n1ofR_global_mem,\
                                                      n2ofR_global_mem,\
                                                      NumAtROutPre_global_mem,\
                                                      End,\
                                                      Start,\
                                                      NumOnSurf,\
                                                      r)
            #stream = cuda.stream() 
            #num_copy_to_host_start = time.time()
            #NumAtROutPre = NumAtROutPre_global_mem.copy_to_host(stream=stream)
            cuda.synchronize()
            #print("Time to transferNumAtROutPre to host is ",time.time() - num_copy_to_host_start)
            print("\nTime to complete AveragesOnShellsInnerLogicCCuda is ",time.time()-LogicCCuda_start)

            #print(np.min(n2ofROutPre),np.max(n2ofROutPre),)

            retofROutR[r][Start:End] = reduced[0]
            retofROutI[r][Start:End] = reduced[1]
            n1ofROut[r][Start:End]   = reduced[2]
            n2ofROut[r][Start:End]   = reduced[3]
            
            # NEED ANOTHER KERNEL HERE TO DO MATRIX-ROW SUM


            #sum_start = time.time()
            #qq_cpu= np.sum(NumAtROutPre,axis=0);# length End-Start, 7936
            #print("CPU time to generate matrix sum is ",time.time()-sum_start)
            qq = sumRowsCuda(NumAtROutPre_global_mem,Start,End)
            #assert (qq_cpu == qq).all()

            NumAtROut[r][Start:End]      = qq;
            #qqq =np.where(qq==0)[0];
            #print("how many zeros of MultVec , %g " %(len(qqq) )  );
        deltaTime =time.time()-startTime;
        #if ((r-1)%5)==0: 
        #    print("NumAtROutPre created in %f seconds, retofROutRPre  in %f seconds for size r=%g " \
        #        % (deltaTimeN,deltaTime,r))

    #print(retofROutRPre)
    return [retofROutR, retofROutI, n1ofROut,n2ofROut,NumAtROut]

    
    


#%%     Section -1     Function Definitions 
#aa= np.array([ [1, 2, 3],[4,5,6]])
#Out[401]: array([[1, 2, 3],
#                [4, 5, 6]])
#bb= aa[1,0:2]
# array([4, 5])

# Keep autojit off!!!!!!
@numba.autojit
def NormalizeShells(nx,ny,nz,kXofR,kYofR,kZofR,inc,retofROutR, retofROutI, n1ofROut,n2ofROut,NumAtEachR, RMax):

    ResultR     = retofROutR.copy();
    ResultI     = retofROutI.copy();
    
    nx2 = int(nx/2);
    ny2 = int(ny/2);
    nz2 = int(nz/2);


    nxOut=nx-1; 
    nyOut=ny-1; 
    nzOut=nz-1; 
    
    if nx%2: nxOut+=1;# if nx was odd, nxOut=nx
    if ny%2: nyOut+=1;# if ny was odd, nyOut=ny
    if nz%2: nzOut+=1;# if nz was odd, nzOut=nz

    ResEMR = np.zeros([nxOut,nyOut,nzOut]);
    ResEMI = np.zeros([nxOut,nyOut,nzOut]);
    ResNum = np.zeros([nxOut,nyOut,nzOut]);
    ResDen = np.zeros([nxOut,nyOut,nzOut]);
    
    ResEMR[nx2-1,ny2-1,nz2-1] = 1.0;
    ResNum[nx2-1,ny2-1,nz2-1] =retofROutR[0][0]
    ResDen[nx2-1,ny2-1,nz2-1] = np.sqrt(n1ofROut[0][0] * n2ofROut[0][0])

    ShapeRR       = ResultR.shape;
    ShapeRI       = ResultI.shape;
    ShapeRetRR = retofROutR.shape;
    ShapeRetRI = retofROutI.shape;
    ShapeN1R   = n1ofROut.shape;
    ShapeN2R   = n2ofROut.shape;
    
    print(ShapeRI,ShapeRR,ShapeRetRR,ShapeRetRI,ShapeN1R,ShapeN2R,inc)
    #return [ResEMR,ResEMI,ResNum,ResDen]
    #ResultR[0][0]=1;
    #ResEMR.set_value_at(nx2-1,ny2-1,nz2-1,1);
    # Values in real space are going to 
    for r in prange(1,min(inc+1,RMax)):
        LastInd= NumAtEachR[r]-1;
        if (r%5 ==1): print(r, LastInd)
        # retofROutR[r][:LastInd]  = np.sum(retofROutRPre,axis=0);
        retNowR = retofROutR[r][:LastInd];
        retNowI = retofROutI[r][:LastInd] ;#Vectors for
        n1Now = n1ofROut[r][:LastInd] ; 
        n2Now = n2ofROut[r][:LastInd] ;#given radius
        Num2Surf = LastInd;
        for jSurf in range(Num2Surf+1): ### Fixed the problem!
            retNowRL = retNowR[jSurf] ;
            retNowIL = retNowI[jSurf] ;
            n1NowL = n1Now[jSurf];
            n2NowL = n2Now[jSurf];#      Single Values
            if n1NowL*n2NowL ==0:
                continue;
            ResultR[r][jSurf] =float(retNowRL/ (np.sqrt(n1NowL * n2NowL)));
            ResultI[r][jSurf] =float(retNowIL/ (np.sqrt(n1NowL * n2NowL)));
            retNowRL = retNowR[jSurf] ;
            retNowIL = retNowI[jSurf] ;
            n1NowL = n1Now[jSurf];
            n2NowL = n2Now[jSurf];#      Single Values
            ResultR[r][jSurf] =float(retNowRL/ (np.sqrt(n1NowL * n2NowL)));
            ResultI[r][jSurf] =float(retNowIL/ (np.sqrt(n1NowL * n2NowL)));
            kX =int(round(kXofR[r][jSurf]));
            kY =kYofR[r][jSurf];
            kZ =kZofR[r][jSurf];
            if (kX==nx2)|(kY==nx2)|(kZ==nx2): 
                continue;
            if kX>0:
                ResEMR[kX+nx2-1,kY+ny2-1,kZ+nz2-1] =  ResultR[r][jSurf];
                ResEMR[nx2-1-kX,ny2-1-kY,nz2-1-kZ] =  ResultR[r][jSurf]
                ResEMI[kX+nx2-1,kY+ny2-1,kZ+nz2-1] =  ResultI[r][jSurf]
                ResEMI[nx2-1-kX,ny2-1-kY,nz2-1-kZ] = -ResultI[r][jSurf]
                ResNum[kX+nx2-1,kY+ny2-1,kZ+nz2-1] =  retofROutR[r][jSurf]
                ResNum[nx2-1-kX,ny2-1-kY,nz2-1-kZ] =  retofROutR[r][jSurf]
                ResDen[kX+nx2-1,kY+ny2-1,kZ+nz2-1] =  np.sqrt(n1ofROut[r][jSurf] * n2ofROut[r][jSurf])    
                ResDen[nx2-1-kX,ny2-1-kY,nz2-1-kZ] =  np.sqrt(n1ofROut[r][jSurf] * n2ofROut[r][jSurf])
            else:#kx=0
                ResEMR[nx2-1, kY+ny2-1, kZ+nz2-1] = ResultR[r][jSurf]
                ResEMR[nx2-1,-kY+ny2-1,-kZ+nz2-1] = ResultR[r][jSurf]
                ResEMI[nx2-1, kY+ny2-1, kZ+nz2-1] = ResultI[r][jSurf]
                ResEMI[nx2-1,-kY+ny2-1,-kZ+nz2-1] = ResultI[r][jSurf]
                ResNum[nx2-1, kY+ny2-1, kZ+nz2-1] = retofROutR[r][jSurf]
                ResNum[nx2-1,-kY+ny2-1,-kZ+nz2-1] = retofROutR[r][jSurf]
                ResDen[nx2-1, kY+ny2-1, kZ+nz2-1] = np.sqrt(n1ofROut[r][jSurf] * n2ofROut[r][jSurf])
                ResDen[nx2-1,-kY+ny2-1,-kZ+nz2-1] = np.sqrt(n1ofROut[r][jSurf] * n2ofROut[r][jSurf])

    return [ResEMR,ResEMI,ResNum,ResDen,ResultR,ResultI]




#%%           Section 0      Set Up Variables




#import FSCLibrary

#  Section 0; Define Names

#print(argv[1],argv[2],argv[3])
#
#fNHalfMap1 = argv[1];
#fNHalfMap2 = argv[2];
#
#ResEMOutMRC = argv[3];
#dthetaInDegrees = float(argv[4]);
#
#

# cd /home/pbaldwin/Desktop/FromYZ/FSC/Synthetic  HA/
# cd /home/pbaldwin/Desktop/FromYZ/FSC/L17d_E/
#
#

#
# 
# ThreeDFSC_ScriptAFinalDec2016.py
#                  HalfMap1.hdf      HalfMap2.hdf     OutputStringLabel                         
# DeltaTheta is hard coded to 20 degrees
#
# Creates    ResEMOutresultAve+OutputStringLabel.csv (which is the usual FSC)
#            ResEMOut+OutputStringLabel.hdf which is the 3D FSC file ResEMR (the real part of the cccs) 
#            Plots+OutputStringLabel.jpg which is the slices along x, y, z 

def main(fNHalfMap1,fNHalfMap2,OutputStringLabel,APixels,dthetaInDegrees):
    blockPrint()
    #fNHalfMap1='run_half1_class001_unfil.mrc'
    #fNHalfMap2='run_half2_class001_unfil.mrc'
    #OutputStringLabel='ApoFullMRC'
    #APixels =1.06049


    #fNHalfMap1=argv[1];
    #fNHalfMap2=argv[2];

    #OutputStringLabel= argv[3]; 

    #APixels = float(argv[4]);

    ResultsDir= 'Results_'+OutputStringLabel+'/';
    isResultsDir=  os.path.isdir(ResultsDir)
    if (1-isResultsDir):
        os.mkdir(ResultsDir)

    ResEMOutHDF_FN= ResultsDir+'ResEM'+OutputStringLabel+'Out.mrc'

    #dthetaInDegrees = float(argv[5]);
    dthetaInRadians = dthetaInDegrees*np.pi/180.0;
    Thresh = np.cos(dthetaInRadians)   ;
    fractionOfTheSphereAveraged = (1-Thresh)/2;
    # Now, deltaTheta takes up a cone which has area
    #     2 pi * (1-cos(deltaTheta))


    FTOut =     ResultsDir+'FTPlot'+OutputStringLabel;
    PlotsOut= ResultsDir+'Plots'+OutputStringLabel;
    fNResRoot=ResEMOutHDF_FN[0:-4]
    fN_csv_out= fNResRoot+'.csv';

    ResNumOutMRC= fNResRoot+'Num.mrc';# The numerator which is the cross product
    ResDenOutMRC= fNResRoot+'Den.mrc';# The fn for denominator, which is normalization

    resultAveOut = fNResRoot+'globalFSC.csv';



    #.94;# acos(Thresh)= 20 degrees
    # the fraction of the sphere is int_0_d 

    # &&&&&&&&          Section 1; Usual FSC


    #calc_fourier_shell_correlation(EMData * with, float w)

    #Fourier Ring/Shell Correlation
    #Purpose: Calculate CCF in Fourier space as a function of spatial frequency
    #          between a pair of 2-3D images.
    #Method: Calculate FFT (if needed), calculate FSC.
    #Input:     f - real or complex 2-3D image
    #         g - real or complex 2-3D image
    #         w - float ring width
    #Output: 2D 3xk real image.
    #         k - length of FSC curve, depends on dimensions of the image and ring width
    #     1 column - FSC,
    #     2 column - normalized frequency [0,0.5]
    #     3 column - currently n /error of the FSC = 1/sqrt(n),
    #                      where n is the number of Fourier coefficients within given shell

    ###############                             BEGIN PAWEL'S CODE


    # EMData *f = this;
    # EMData *g = with;

    #  ------------------------------------------



    os.getcwd()


    #%%        Section 1    Read in Data, Take Transpose

    if 0:
        h5f_HalfMap1 =h5py.File(fNHalfMap1,'r')
        dataSetNow=h5f_HalfMap1['MDF/images/0/image']
        f=np.array(dataSetNow)
        #
        h5f_HalfMap2 =h5py.File(fNHalfMap2,'r')
        dataSetNow=h5f_HalfMap2['MDF/images/0/image']
        g=np.array(dataSetNow)
        #
        h5f_HalfMap1.close()
        h5f_HalfMap2.close()
        #h5f_HalfMap1.visit(print)
        
    if 1:
        h5f_HalfMap1= mrcfile.open(fNHalfMap1)
        f= h5f_HalfMap1.data

        h5f_HalfMap2= mrcfile.open(fNHalfMap2)
        g= h5f_HalfMap2.data

        h5f_HalfMap1.close()
        h5f_HalfMap2.close()


        
    startTime = time.time()

    fT=f.T;# Now it is like EMAN
    gT=g.T;#


    [nx,ny,nz] =fT.shape

    deltaTime =time.time()-startTime;
    print("Maps read in %f seconds for size nx=%g " % (deltaTime,nx))


    #%%         Section  Add Axes For Checking
    #      
    if 0:
        fTPlus=AddAxes(fT,2,10)
        
        h5f_write = h5py.File('fTPlus.hdf','w')
        h5f_write.create_dataset('MDF/images/0/image',data=fTPlus)
        # <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
        h5f_write.close()



    #%%         Section 2    Take Fourier Transform; find Cos phase residual
    #                     CosHangle


    startTime = time.time()


    [fp,gp]=ZeroPad(nx,ny,nz,fT,gT)
        
    deltaTime =time.time()-startTime;
    print("NormPad created in %f seconds for size nx=%g " % (deltaTime,nx))


    startTime = time.time()

    F =np.fft.fftn(fp);#F= temp.transpose();
    G =np.fft.fftn(gp);#G= temp.transpose();


    H=F*np.conj(G);
    H.shape
    HAngle= np.angle(H);
    CosHangle=np.cos(HAngle);


    deltaTime =time.time()-startTime;
    print("FFTs performed in %f seconds for size nx=%g " % (deltaTime,nx))

    #FFTs performed in 4.559401 seconds for size nx=256 

    #%%          Section 3 Create Real Arrays as in original EMAN program

    startTime = time.time()

    d1        = FFTArray2Real(nx,ny,nz,F);
    d2        = FFTArray2Real(nx,ny,nz,G);
    dcH        = FFTArray2Real(nx,ny,nz,CosHangle);
    dFPower = FFTArray2Real(nx,ny,nz,F*np.conj(F));
    #dPR = FFTArray2Real(nx,ny,nz,HAngle);

    deltaTime =time.time()-startTime;
    print("FFTArray2Real performed in %f seconds for size nx=%g " % (deltaTime,nx))

    # FFTArray2Real performed in 22.700670 seconds for size nx=256 , wo autojit
    # FFTArray2Real performed in 0.9       seconds for size nx=256 , w autojit
    # 

    # d1[15][16][17]  is d1.get_value_at(15,16,17)
    # But d1[15][16][17], for EMData objects is complex...
    # f[16,17,18] = f[16][17][18]
    # But this is EMAN's 18,17,16


    #%%         Section 4a Create FSC Outputs; n1 and n2 are the normalizations of
    #          f and g. cH means the cosine of the phase residual.

    nx2 = nx/2;
    ny2 = ny/2;
    nz2 = nz/2;

    lsd2=nx+2;

    dx2 = 1.0/float(nx2)/float(nx2);
    dy2 = 1.0/float(ny2)/float(ny2);
    dz2 = 1.0/float(nz2)/float(nz2);
    # int inc = Util::round(float(std::max(std::max(nx2,ny2),nz2))/w);
    w=1;

    inc = max(nx2,ny2,nz2)/w;
    inc = int(inc)




    startTime = time.time()

    CreateFTLikeOutputs_start = time.time()
    [retcHGlobal,lr]     = CreateFTLikeOutputs(inc,nx,ny,nz,dcH,nx2,ny2,nz2,dx2,dy2,dz2)
    print("Time to complete CreateFTLikeOutputs is ",time.time() - CreateFTLikeOutputs_start)

    CreateFSCOutputs_start = time.time()
    [ret,n1,n2,lr] = CreateFSCOutputs(inc,nx,ny,nz,d1,d2,nx2,ny2,nz2,dx2,dy2,dz2);
    print("Time to complete CreateFSCOutputs_start is ",time.time() - CreateFSCOutputs_start)

    CreateFTLikeOutputs_start = time.time()
    [FPower,lr]       = CreateFTLikeOutputs(inc,nx,ny,nz,dFPower,nx2,ny2,nz2,dx2,dy2,dz2)
    print("Time to complete CreateFTLikeOutputs_start is ",time.time() - CreateFTLikeOutputs_start)


    deltaTime =time.time()-startTime;
    print("CreateFSCOutputs performed in %f seconds for size nx=%g " % (deltaTime,nx))

    # CreateFSCOutputs performed in     0.393    seconds for size nx=256 ; using autojit
    # CreateFSCOutputs performed in 45.944    seconds for size nx=256 ; without autojit
    # list(ret)

    #%%         Section 4b     Write out FSCs. Define RMax based on this


    # for nx =32, linc is 17


    linc = 0;
    for i in range(inc+1):
        if (lr[i]>0):
            linc +=1;

    result    = [0 for i in range(3*linc)];

    ii = -1;
    for i in range(inc+1):
        if (lr[i]>0): 
            ii +=1;
            result[ii]          = float(i)/float(2*inc);
            result[ii+linc]      = float(ret[i] / (np.sqrt(n1[i] * n2[i])));
            result[ii+2*linc] = lr[i]  ;# Number of Values


    NormalizedFreq = result[0:(inc+1)];
    resultAve= result[(inc+1):(2*(inc+1))];# This takes values inc+1 values from inc+1 to 2*inc+1

    with open(resultAveOut, "w") as fL1:
        AveWriter = csv.writer(fL1)
        for j in range(inc+1):
            valFreqNormalized = NormalizedFreq[j];
            valFreq              = valFreqNormalized/APixels;
            valFSCshell= resultAve[j];
            AveWriter.writerow([valFreqNormalized,valFreq,valFSCshell])

    #k0     = np.array(range(Nxf))/APixels/2.0/Nxf;


    aa= np.abs(np.array(resultAve))<.13
    bb=np.where(aa)[0];
    try:
        RMax=bb[0]+4
    except:
        RMax=inc
        
    
    RMax= min(RMax,inc)
    print('simple FSC written out to '+resultAveOut)
    print('RMax = %d'%(RMax))


    # RMax=40;#      Change Me      PRB


    #with open(resultAveOut, "w") as fL1:
    #     AveWriter = csv.writer(fL1)
    #     for val in resultAve:
    #         AveWriter.writerow([val])

    #  Variables

    # result contains normalized freq,    FSC , and number of data points
    # n1 is the normalization factor for the first    half map over the sphere
    # n2 is the normalization factor for the second half map over the sphere
    # ret is the inner product over the whole sphere; becomes inner product after norm


    # &&&&&&&&&&&&&&&&&&&&&&&7          END PAWEL'S CODE and section 1
    #%%

    pltName = 'log rot ave FT';
    ff =plt.figure(pltName);
    #f.suptitle('Color as Function of Length')
    Nxf= np.int(nx2)+1
    k0    = np.array(range(Nxf))/APixels/2.0/Nxf;
    k= np.arange(Nxf)
    fig, ax = plt.subplots()
    ax.plot(k0,np.log(FPower), 'b', label='FPower')
    ax.set_xlabel('Spatial Frequency (1/A) ')
    ax.set_ylabel('log rot ave FT Power')

    legend = ax.legend(loc='upper right', shadow=True)

    fig.savefig(FTOut +'.jpg');



    pltName = 'DPR and global FSC';
    ff =plt.figure(pltName);
    #f.suptitle('Color as Function of Length')
    Nxf= np.int(nx2)+1
    k=np.array(range(Nxf))/1.07/2.0;
    k= np.array(range(Nxf))
    fig, ax = plt.subplots()
    ax.plot(k0, 2*retcHGlobal/lr, 'b', label='ave cos phase')
    ax.plot(k0, resultAve, 'g', label='FSC')
    legend = ax.legend(loc='upper right', shadow=True)
    ax.set_xlabel('Spatial Frequency (1/A) ')
    ax.set_ylabel('various FSCs')



    #%%           Section 5. Create generalized FSC  and FT arrays

    startTime = time.time()

    # The Number at each R is LastInd_OfR+1

    [kXofR,kYofR,kZofR,retofRR,retofRI,n1ofR,n2ofR,NumAtEachR] = \
        createFSCarrays(nx,ny,nz,lsd2,lr,inc,dx2,dy2,dz2,d1,d2,nx2,ny2,nz2)

    [kXofR,kYofR,kZofR,retcH,retFT,n12ofR] = \
        createFTarrays(nx,ny,nz,lsd2,lr,inc,dx2,dy2,dz2,dcH,dFPower,nx2,ny2,nz2)
        
    deltaTime =time.time()-startTime;

    print("FSC arrays created in %f seconds for size nx=%g " % (deltaTime,nx))

    NumAtEachRMax= NumAtEachR[-1]

    kXofR=kXofR[:,:NumAtEachRMax]; kYofR=kYofR[:,:NumAtEachRMax]; kZofR=kZofR[:,:NumAtEachRMax];
    retofRR = retofRR[:,:NumAtEachRMax]; retofRI =retofRI[:,:NumAtEachRMax];
    n1ofR    =    n1ofR[:,:NumAtEachRMax]; n2ofR     =    n2ofR[:,:NumAtEachRMax];

    NumAtEachRMax      =     NumAtEachR[RMax];
    NumAtEachRMaxCuda = 15871;# NumAtEachR[50];#15871

    MaxLoopsIllNeed = NumAtEachRMax*NumAtEachRMax/NumAtEachRMaxCuda/NumAtEachRMaxCuda;

    # kXofR,kYofR, kZofR
    # retofRR,retofRI
    # n1ofR,n2ofR
    # NumAtEachR is a one d arry indicating     the unique number of sites at each radius


    # Some Tests  j=2;list(kXofR[j][0:NumAtEachR[j]])
    #j=2;list(retofRI[j][0:NumAtEachR[j]])
    #j=2;list(retofRR[j][0:NumAtEachR[j]])
    #j=2;list(n1ofR[j][0:NumAtEachR[j]])
    #j=2;list(n2ofR[j][0:NumAtEachR[j]])

    # FSC arrays created in 1.399719 seconds for size nx=128 using autojit (vs 5 seconds)
    # FSC arrays created in 2         seconds for size nx=256 using autjit


    #If one normally indexes a square  array as (X,Y)
    # then the upper right part would have index
    #       N(X-1) - (X)(X-1)/2 + Y
    # The greatest Element would be when X=Y=N
    #     N(N-1)/2 +N = N(N+1)/2
    # For N=1; gives 1. For N=2, 

    #vv=NumAtEachRMax;
    #hh= int(vv*(vv+1)/2)
    #hh=5.1972* pow(10,9)


    #%%       Section 6. Average on Shells

    if 0:
        RMax=30;
        RMax=81;
        RMax=47;
        RMax=60;

    startTime = time.time()

    [retofROutR, retofROutI, n1ofROut,n2ofROut,NumAtROut] = \
                AveragesOnShellsUsingLogicB(inc,retofRR,retofRI,n1ofR,n2ofR, kXofR,kYofR,kZofR,     \
                                                        NumAtEachR,Thresh, RMax);

    if 0:
        [retofRcH, retofRFT, n12ofROut,n21ofROut,NumAtROut] = \
                AveragesOnShellsUsingLogicB(inc,retcH,retFT,n12ofR,n12ofR, kXofR,kYofR,kZofR,  \
                                                        NumAtEachR,Thresh, RMax);


    # j=2; list(retofROutR[j][:NumAtEachR[j]]) perfect
    # list(retofROutI[2][:NumAtEachR[2]]) perfect
    #  list(n1ofROut[2][:NumAtEachR[2]])  perfect
    #  list(n2ofROut[2][:NumAtEachR[2]]) perfect
    #  list(NumAtROut[2][:NumAtEachR[2]]) perfect

        
    deltaTime =time.time()-startTime;

    print("AveragesOnShells created in %f seconds for size nx=%g " % (deltaTime,nx))

    # no autojit AveragesOnShells created in 134.065687 seconds for size nx=64 
    #  autojit AveragesOnShells created in 58.065687 seconds for size nx=64 
    # autojit AveragesOnShells created in 0.232876 seconds for size nx=32 
    # autojit on inner loop (or both loops) AveragesOnShells created in 2.050189 seconds for size nx=64 
    # autojit AveragesOnShells created in 26.903263 seconds for size nx=128 
    # cudajit Average On shells created in 14.11 seconds for size nx=256
    # cudajit AveragesOnShells created in 0.058333 seconds for size nx=32 
    # cudajit AveragesOnShells created in 2.212252 seconds for size nx=256
    # AveragesOnShells created in 1602.677335 seconds for size nx=128  without matrix multiply
    # AveragesOnShells created in 149  seconds for size nx=128    with matrix multiply
    #
    #  NumAtROutPre.shape 1311 1311
    #sum, sum of NumAtROutPre = 104763 
    #r =15, jLoop = 0 
    #NumAtROutPre created in 2.635284 seconds, retofROutRPre  in 0.044771 seconds for size r=15 

    #NumAtROutPre created in 87.902457 seconds for size r=128 
    #retofROutRPre created in 19.214520 seconds for size r=128 
    #AveragesOnShells created in 107.825461 seconds for size nx=256 

    if 0:

        h5f_write = h5py.File(ResultsDir+'Radial'+'ResEM'+OutputStringLabel+'Out.hdf','w')
        h5f_write.create_dataset('MDF/images/0/image',data=retofROutR)
        # <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
        h5f_write.close()

        h5f_write = h5py.File(ResultsDir+'Radial'+'cH'+OutputStringLabel+'Out.hdf','w')
        h5f_write.create_dataset('MDF/images/0/image',data=retofRcH)
        # <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
        h5f_write.close()

        h5f_write = h5py.File(ResultsDir+'Radial'+'FT'+OutputStringLabel+'Out.hdf','w')
        h5f_write.create_dataset('MDF/images/0/image',data=retofRFT)
        # <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
        h5f_write.close()

        h5f_write = h5py.File(ResultsDir+'Radial'+'n12'+OutputStringLabel+'Out.hdf','w')
        h5f_write.create_dataset('MDF/images/0/image',data=n12ofROut)
        # <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
        h5f_write.close()
        
        h5f_write = h5py.File(ResultsDir+'Radial'+'n1'+OutputStringLabel+'Out.hdf','w')
        h5f_write.create_dataset('MDF/images/0/image',data=n1ofROut)
        # <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
        h5f_write.close()
        
        h5f_write = h5py.File(ResultsDir+'Radial'+'n2'+OutputStringLabel+'Out.hdf','w')
        h5f_write.create_dataset('MDF/images/0/image',data=n2ofROut)
        # <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
        h5f_write.close()


        

    if 0:
        fig = plt.figure()
        ax=fig.gca()
        #ThisHist = ax.hist(GuessedA*20)
        j=47;
        ax.scatter(retFT[j,:NumAtEachR[j]]/n12ofROut[j,:NumAtEachR[j]],
                retcH[j,:NumAtEachR[j]]/n12ofROut[j,:NumAtEachR[j]]);
        ax.scatter(retFT[j,:NumAtEachR[j]],       retcH[j,:NumAtEachR[j]]);
        ax.set_xlabel('FT power');
        ax.set_ylabel('Phase Residual');
        ax.set_title('Phase Residual vs FT Power');
        #fig.savefig('PointWeightsTimesValues1D.jpg')


    #%%      Section 7. We have the unaveraged quantities


    startTime = time.time()
    [ResEMR,ResEMI,ResNum,ResDen,ResultR,ResultI] = \
                NormalizeShells(nx,ny,nz,kXofR,kYofR,kZofR,inc,retofROutR, retofROutI, n1ofROut,n2ofROut, NumAtEachR, RMax);
    deltaTime =time.time()-startTime;

    print("NormalizeShells created in %f seconds for size nx=%g, RMax=%g " % (deltaTime,nx, RMax))

    # NormalizeShells created in 3.175528 seconds for size nx=64; wo autojit
    # NormalizeShells created in 3.175528 seconds for size nx=64    
    # NormalizeShells created in 3.602324 seconds for size nx=32 
    # NormalizeShells created in 0.401217 seconds for size nx=32 
    # NormalizeShells created in 3.374886 seconds for size nx=256 ; wo autojit
    # NormalizeShells created in infinity  seconds for size nx=256 ; w autojit
    #list(ResEMR[65,65,:]) for GS IR protein
    #list(ResEMR[63,63,:]) for HA Sh2 protein


    #%%
    #     Section 8.         Write Out FSC volumes to file
    #        
                
    #csvfile=open(OutP1csvFN,'w')
    #P1writer= csv.writer(csvfile,delimiter=' ',quotechar='|');
    #P1writer.writerow(PolynomialL1);
    #
    #
    #if 0:
    #     ResEMRB=np.zeros([334,334,334])
    #     ResEMRB[:,:,:] = ResEMR[:334,:334,:334]
    #     mrc.write(ResEMR,'Proteasome_FSC.mrc')
    #     ResEMR.write_mrc('Proteasome_FSC.mrc');
    #     ResNum.write_mrc('Proteasome_Num.mrc');
    #     ResDen.write_mrc('Proteasome_Den.mrc');
    #     h5f_write = h5py.File('IR_FSC.hdf','w')
    #     h5f_write.create_dataset('MDF/images/0/image',data=ResEMR)
    #     # <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
    #     h5f_write.close()
    #     #
    #     h5f_write = h5py.File('Proteasome_FSC.hdf','w')
    #     h5f_write.create_dataset('MDF/images/0/image',data=ResEMR)
    #     # <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
    #     h5f_write.close()
    #     #
    #     h5f_write = h5py.File('Proteasome_HalfMap1.hdf','w')
    #     h5f_write.create_dataset('MDF/images/0/image',data=f)
    #     # <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
    #     h5f_write.close()
    #

    if 0:
        h5f_write = h5py.File('ResEMOutHDF_FN.hdf','w')
        h5f_write.create_dataset('MDF/images/0/image',data=ResEMR.T)
        # <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
        h5f_write.close()


    # print('ResEMR'); print(type(ResEMR))

    if 1:
        ResEMRT= ResEMR.T;
        mrc_write=mrcfile.new(ResEMOutHDF_FN,overwrite=True)
        mrc_write.set_data(ResEMRT.astype('<f4'))
        mrc_write.voxel_size = (float(APixels),float(APixels),float(APixels))
        mrc_write.update_header_from_data()
        mrc_write.close()


    #%%

    if 0:# Legacy code that includes axes for plotting
        ResEMRPlus=AddAxes(ResEMR,2,10)
        h5f_write = h5py.File('ResEMROutWithAxes.hdf','w')
        h5f_write.create_dataset('MDF/images/0/image',data=ResEMRPlus.T)
        # <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
        h5f_write.close()




    #%%
    #     Section 9;          Write Out         PolynomialL1, PolynomialL2, N2, resultAve to file


    if 0:
        startTime = time.time()
        [PolynomialL1,PolynomialL2] = ReturnL1L2Moments(inc,LastInd_OfR,kXofR,kYofR,kZofR,n1ofR,n2ofR, ResultR, ResultI);
        deltaTime =time.time()-startTime;
        print("ReturnL1L2Moments created in %f seconds for size nx=%g" % (deltaTime,nx))


    # ReturnL1L2Moments created in 0.13 seconds for size nx=32; autojit 1.81 seconds

    #%%      Section 10 Plot 5 Axes
    [xf,yf,zf]= ExtractAxes(ResEMR);
    Nxf=len(xf)
    DPRf= 2*retcHGlobal[:-1]/lr[:-1];
    Globalf =  resultAve[:-1];

    #pltName='ProteosomePlots';
    #pltName='HAPlots';
    #pltName='IRPlots';
    pltName=PlotsOut;
    ff =plt.figure(pltName);
    #f.suptitle('Color as Function of Length')

    k=np.array(range(Nxf))/1.07/2.0;
    k0    = np.array(range(Nxf))/APixels/2.0/Nxf;

    k= np.array(range(Nxf))
    fig, ax = plt.subplots()
    ax.plot(k0, xf, 'b', label='x dir')
    ax.plot(k0, yf, 'g', label='y dir')
    ax.plot(k0, zf, 'r', label='z dir')
    ax.plot(k0, DPRf, 'k', label='ave cos phase')
    ax.plot(k0, Globalf, 'y', label='global FSC')

    ax.set_xlabel('Spatial Frequency (1/A) ')
    ax.set_ylabel('FSCs')

    # Now add the legend with some customizations.
    legend = ax.legend(loc='upper right', shadow=True)


    fig.savefig(pltName +'.jpg');

    #ResEMR[:,ny2,nz2]

    xyzf = np.reshape(np.concatenate( (xf,yf,zf,DPRf,Globalf)),(5,Nxf)    )
    xyzf = xyzf.T;
    np.savetxt(PlotsOut+'.csv',xyzf)

    #%%
    #sys.exit()

    ## Flush out plots
    plt.clf()
    plt.cla()
    plt.close()    

    #%%

    #fOut = AddAxes(f,0);
    #
    #
    #h5f_write = h5py.File('Proteasome_withz.hdf','w')
    #h5f_write.create_dataset('MDF/images/0/image',data=fOut)
    ## <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
    #h5f_write.close()
    #
    #
    #
    enablePrint()

if __name__ == "__main__":
    main(argv[1],argv[2],argv[3],float(argv[4]),float(argv[5]))
    

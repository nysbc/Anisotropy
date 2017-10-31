import math
import numpy as np
import numba
from numba import autojit, prange

@numba.autojit
def AveragesOnShells(kXNow,kYNow,kZNow, NumOnSurf, Thresh,Start, End):
#        NumAtROutPre = np.zeros(int(NumOnSurf*(NumOnSurf+1)/2),dtype=int)
#        NumAtROutPre = np.zeros([NumOnSurf,NumOnSurf],dtype=int)
#        NumAtROutPre = np.identity(NumOnSurf,dtype=int)
#        NumAtROutPre = np.zeros((NumOnSurf,End-Start), dtype=np.int)
#        print('Hello')

#       NumAtROutPre has dimensions NumOnSurf by End-Start
#       Each element NumAtROutPre[m,n]  denotes whether m is close to Start+n

# CJN
# NumOnSurf is equal to the length of kXNow, kYNow, and kZNow
# Thresh does not change
# (End - Start) is something like the size of the shell being analyzed

       #print("End - Start is ",End," - ",Start)
        NumAtROutPre = np.zeros((NumOnSurf,End-Start), dtype=np.int)

        Thresh2=Thresh*Thresh
        #print("Thresh is",Thresh)
        #print("NumOnSurf is",NumOnSurf)

        for jSurf1 in prange(NumOnSurf):
                #retNow1RL =retofRR[r] 

                kX1=kXNow[jSurf1];
                kY1=kYNow[jSurf1];
                kZ1=kZNow[jSurf1];#       Single Values
                Prod11  = kX1*kX1+kY1*kY1+kZ1*kZ1;
                #print("Prod11 is",Prod11)
                if Prod11==0:
                    continue
                    print("Prod11 == 0")
                for jSurf2 in prange(End-Start):# labels kX, etc
                        #if jSurf1==jSurf2: continue
                        #Count+=1;
                        kX2=kXNow[jSurf2+Start];
                        kY2=kYNow[jSurf2+Start];
                        kZ2=kZNow[jSurf2+Start];#       Single Values
                        Prod12  = kX1*kX2+kY1*kY2+kZ1*kZ2;
                        Prod22  = kX2*kX2+kY2*kY2+kZ2*kZ2;
                        #print("Prod12 is",Prod12)
                        #print("Prod22 is",Prod22)
                        if Prod22==0:continue
                        Inner2  = Prod12*Prod12/(Prod11*Prod22);
                        NumAtROutPre[jSurf1,jSurf2] = Inner2;
                        if Inner2>Thresh2:# Then angle is sufficiently small
                                #NumAtROutPre[NumOnSurf*jSurf1 -(jSurf1+1)*jSurf1//2 +jSurf2 ]   = 1;
                                NumAtROutPre[jSurf1,jSurf2]       = 1;
                                #NumAtROutPre[jSurf2,jSurf1 ]   = 1;
                                #N(X-1) - (X)(X-1)/2 + Y
                                #print(r,jSurf1,jSurf2,retNow1L,retofR[r][jSurf2],retofROut[r][jSurf2],);
        #print(Count, NumOnSurf*(NumOnSurf+1)//2)
        return NumAtROutPre


if __name__ == "__main__":
# Host code
    NumOnSurf = 3
    End = 3
    Start = 0
    Thresh = 0.93

    #A = np.full((NumOnSurf, 3), 1.3, np.float)
    A = np.array([[1,2,3],[4,5,6],[7,8,9]])

    threadsperblock = (8,8)
    blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
    blockspergrid = (blockspergrid_x)


# AveragesOnShellsInnerLogicKernelnonCuda(kXNow,kYNow,kZNow,NumOnSurf,Thresh,Start, End):

    C = AveragesOnShells(A[:,0],A[:,1],A[:,2],NumOnSurf,Thresh,Start,End)

    print("C is ",C)

    print("shape of C is",np.shape(C))

from __future__ import division
from numba import cuda
import math
import numba
import numpy as np
import time

@cuda.jit
def filter_and_sum(retNowR,retNowI,n1Now,n2Now,NumAtROutPre,reduced,End,Start):

    x = cuda.grid(1)
    if (x >= reduced.shape[1]):
      return
#    print("reduced shape is ",reduced.shape)
#    print("x is ",x)
    retofROutRPre = 0
    retofROutIPre = 0
    n1ofROutPre = 0
    n2ofROutPre = 0
#    print("End-Start is",End-Start)
#    print("x is ",x)
#    print("NumAtROutPre shape is ",NumAtROutPre.shape)
#    print("NumatROutPre[:,",x,"] is ",NumAtROutPre[:,x])
#    print("NumatROutPre[:,",x,"].shape is ",NumAtROutPre[:,x].shape)
#    print("********************************************************")
#    print("retNowR[",x,"] is ",retNowR[x])
#    print("retNowR[",x,"].shape is ",retNowR[x].shape)
#    print("retNowR.shape is ",retNowR.shape)
    for i in range((End-Start)):
#        print("i is ",i)
        MultVec = NumAtROutPre[:,x]
#        print("MultVec: ",MultVec)
#        print("NumAtROutPre[:,"+str(x)+"] is :",NumAtROutPre[:,x])
#        print("Shape of retNowR[x,i] is: ",np.shape(retNowR[i]))
        retofROutRPre += retNowR[i]*MultVec[i]
#        print("retNowR[x,"+str(i)+"]: ",retNowR[x,i])
#        print("MultVec["+str(i)+"]: ",MultVec[i])
#        print("retNowR[x,"+str(i)+"]*MultVec["+str(i)+"] = ",retNowR[x,i]*MultVec[i])
        retofROutIPre += retNowI[i]*MultVec[i]
#        print("retofROutRPre is: ",retofROutRPre)
        n1ofROutPre += n1Now[i]*MultVec[i]
        n2ofROutPre += n2Now[i]*MultVec[i]


    reduced[0,x] = retofROutRPre
    reduced[1,x] = retofROutIPre
    reduced[2,x] = n1ofROutPre
    reduced[3,x] = n2ofROutPre
#    print("reduced "+str(x)+"is ",reduced,x)



# CUDA kernel
@cuda.jit()
def cuda_calcProd11(kXNow,kYNow,kZNow, Prod11):
    """Calculate partial product Prod11
    """
    x = cuda.grid(1)
    if x >= kXNow.shape[0]:
        return

    Prod11[x] = kXNow[x]*kXNow[x] + kYNow[x]*kYNow[x] + kZNow[x]*kZNow[x]
    #Prod11 = kX1*KX1 + kY1*kY1 + kZ1*kZ1

@cuda.jit()
def cuda_calcInner2(kXNow,kYNow,kZNow,Prod11,C,End,Start,Thresh):
    """Calculate Prod12
    """

    x = cuda.grid(1)
    if x >= Prod11.shape[0]:
        return

    Thresh2 = Thresh*Thresh

    for i in range(End-Start):

        if Prod11[x]==0:
            C[x,i] = 0

        else:

            #Prod12 = kX1*kX2 + kY1*kY2 + kZ1*kZ2
            Prod12 = kXNow[x]*kXNow[(i+Start)] + kYNow[x]*kYNow[(i+Start)] + kZNow[x]*kZNow[(i+Start)]
            
            #Prod22 = kX2*kX2 + kY2*kY2 + kZ2*kZ2
            Prod22 = kXNow[(i+Start)]*kXNow[(i+Start)] + kYNow[(i+Start)]*kYNow[(i+Start)] + kZNow[(i+Start)]*kZNow[(i+Start)]

            if Prod22==0:
                C[x,i] = 0
            else:

                Inner2 = Prod12*Prod12/(Prod11[x]*Prod22)

                if Inner2>Thresh2:
                    C[x,i] = 1
                else:
                    C[x,i] = 0

if __name__ == "__main__":
# Host code
    NumOnSurf = 15000
    End = 15000
    Start = 0
    Thresh = 0.93

    A = np.random.randint(0,9,size=(NumOnSurf,3))
# Copy the arrays to the device
    A_global_mem = cuda.to_device(A)

# Allocate memory on the device for the result
    C_global_mem = cuda.device_array((NumOnSurf,End-Start))
    Prod11_global_mem = cuda.device_array((NumOnSurf))

# Configure the blocks
    threadsperblock = (64,1,1)
    blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
    blockspergrid = (blockspergrid_x,1,1)
    print("blockspergrid_x = ",blockspergrid_x)
    print("blockspergrid = ",blockspergrid)

# Start the kernel 
    start_gpu = time.time()
    cuda_calcProd11[blockspergrid, threadsperblock](A_global_mem,Prod11_global_mem)

    cuda_calcInner2[blockspergrid, threadsperblock](A,Prod11_global_mem,C_global_mem,End,Start,Thresh)

# Copy the result back to the host
    stream = cuda.stream()
    start_copy_to_host = time.time()
    C = C_global_mem.copy_to_host(stream=0)
    end_copy_to_host = time.time()
    end_gpu = time.time()
    print(C)
    print("shape of A is ",np.shape(A))
    print("shape of C is ",np.shape(C))
    print("time to execute GPU is ",end_gpu - start_gpu)

    print("time to copy to host is ",end_copy_to_host - start_copy_to_host)

from __future__ import division
from numba import cuda
import math
import numba
import numpy as np
import time





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
    threadsperblock = (32,1,1)
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

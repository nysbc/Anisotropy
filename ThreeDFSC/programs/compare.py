import cudatest
import jittest
import math
import numpy as np
from numba import cuda
import time


def test():

    start = time.time()

# Host code
    NumOnSurf = 1000
    End = 200
    Start = 0
    Thresh = 0.93

    
#   A = np.full((NumOnSurf, 3),1.12 , np.int)


    A = np.random.randint(0,9,size=(NumOnSurf,3))
#A = [[1, 2, 3],[4, 5, 6],[7, 8, 9]]
#A = np.random.rand(NumOnSurf,(End-Start))

# Copy the arrays to the device
    A_global_mem = cuda.to_device(A)

# Allocate memory on the device for the result
    C_global_mem = cuda.device_array((NumOnSurf,End-Start))
#C_global_mem = np.zeros((NumOnSurf,End-Start),dtype=np.int)


# Configure the blocks
    threadsperblock = (32,32)
    blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
    blockspergrid = (blockspergrid_x,blockspergrid_x)

    #blockspergrid = 512

    print("blockspergrid_x = ",blockspergrid_x)
    print("blockspergrid = ",blockspergrid)

    start_cuda = time.time()
# Start the kernel 
    cudatest.innerProductCuda[blockspergrid, threadsperblock](A_global_mem, C_global_mem,End,Start,Thresh)

# Copy the result back to the host
    C = C_global_mem.copy_to_host()
#C = cudatest.innerProductCuda[blockspergrid, threadsperblock](A_global_mem, C_global_mem,End,Start,Thresh)
    end_cuda = time.time()

    print("AveragesOnShells CUDA: ")
    print(C)
    print("shape of C is ",np.shape(C))

    start_jit = time.time()
    C2 = jittest.AveragesOnShells(A[:,0],A[:,1],A[:,2], NumOnSurf, Thresh,Start, End)
    end_jit = time.time()
    print("AveragesOnShells jit: ")
    print(C2)

    print(type(C))
    print(type(C2))

    print(sum(sum(C==C2)))
    print(NumOnSurf*(End-Start))

    #for i in enumerate(C):
    #    for j in enumerate(i):
    #        if C[i,j] != C2[i,j]:
    #            print(C[i,j] != C2[i,j])

    end = time.time()
    print("CUDA version completed in %.3f seconds."%(end_cuda - start_cuda))
    print("AUTOJIT version completed in %.3f seconds."%(end_jit - start_jit))
    print("Completed in %.3f seconds."%(end-start))

    return C,C2

if __name__ == "__main__":
    test()

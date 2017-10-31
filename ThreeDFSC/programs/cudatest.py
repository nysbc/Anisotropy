from __future__ import division
from numba import cuda
import numpy
import math
import numba




# CUDA kernel
@cuda.jit()
def innerProductCuda(A, C,End,Start,Thresh):
    """Perform inner product of components of A
    """
    Thresh2 = Thresh*Thresh;
    row, col = cuda.grid(2);
    if row < C.shape[0] and col < C.shape[1]:
        for i in range(C.shape[0]):
            Prod11 = A[row,0]*A[row,0] + A[row,1]*A[row,1] + A[row,2]*A[row,2];

            #Prod11 = kX1*KX1 + kY1*kY1 + kZ1*kZ1

            if Prod11==0:
                C[row,col] = 0;
                continue;


            cuda.syncthreads()
            for col in range(End-Start):
                #Prod12 = kX1*kX2 + kY1*kY2 + kZ1*kZ2
                #Prod22 = kX2*kX2 + kY2*kY2 + kZ2*kZ2
                Prod12 = A[row,0]*A[(col+Start),0] + A[row,1]*A[(col+Start),1] + A[row,2]*A[(col+Start),2];
                #print(A[row,0]*A[(col+Start),0],A[row,1]*A[(col+Start),1],A[row,2]*A[(col+Start),2],Prod12)
                
                Prod22 = A[(col+Start),0]*A[(col+Start),0] + A[(col+Start),1]*A[(col+Start),1] + A[(col+Start),2]*A[(col+Start),2];
                if Prod22==0:
                    C[row,col] = 0;
                    continue
#                else:
#                    Inner2 = Prod12*Prod12/(Prod11*Prod22)
                Inner2 = Prod12*Prod12/(Prod11*Prod22);

                if Inner2>Thresh2:
                    C[row,col] = 1;
                else:
                    C[row,col] = 0;
                #C[row,col] = Inner2
                cuda.syncthreads()

if __name__ == "__main__":
# Host code
    NumOnSurf = 3
    End = 3
    Start = 0
    Thresh = 0.93

    A = numpy.full((NumOnSurf, 3), 1.1, numpy.int)
    A = numpy.array([[1,2,3],[4,5,6],[7,8,9]])

# Copy the arrays to the device
    A_global_mem = cuda.to_device(A)

# Allocate memory on the device for the result
#C_global_mem = cuda.device_array((NumOnSurf,End-Start))
    C_global_mem = cuda.device_array((NumOnSurf,End-Start))
#    C_global_mem = np.zeros(NumOnSurf,End-Start,dtype=np.int32)

# Configure the blocks
    threadsperblock = (32,32)
    blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
    blockspergrid = (blockspergrid_x)

    print("blockspergrid_x = ",blockspergrid_x)
    print("blockspergrid = ",blockspergrid)

# Start the kernel 
    innerProductCuda[blockspergrid, threadsperblock](A_global_mem, C_global_mem,End,Start,Thresh)

# Copy the result back to the host
    C = C_global_mem.copy_to_host()

    print(C)
    print("shape of A is ",numpy.shape(A))



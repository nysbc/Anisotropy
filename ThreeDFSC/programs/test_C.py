from jittest import AveragesOnShellsInnerLogicC
import math
import numpy as np
from numba import cuda,float32
import time


@cuda.jit
def filter_and_sum(retNowR,retNowI,n1Now,n2Now,NumAtROutPre,reduced,End,Start):

    x = cuda.grid(1)
    if (x >= retNowR.shape[0]):
      return

    retofROutRPre = 0
    retofROutIPre = 0
    n1ofROutPre = 0
    n2ofROutPre = 0
    for i in range((End-Start)):
        MultVec = NumAtROutPre[:,i]
        retofROutRPre += retNowR[x,i]*MultVec[i]
        retofROutIPre += retNowI[x,i]*MultVec[i]
        n1ofROutPre += n1Now[x,i]*MultVec[i]
        n2ofROutPre += n2Now[x,i]*MultVec[i]
    reduced[x,0] = retofROutRPre
    reduced[x,1] = retofROutIPre
    reduced[x,2] = n1ofROutPre
    reduced[x,3] = n2ofROutPre

# initiate data
MATRIX_M = 5
MATRIX_N = 5
retNowR = np.random.randint(0,9,(MATRIX_M,MATRIX_N))
retNowI = np.random.randint(0,9,(MATRIX_M,MATRIX_N))
n1Now = np.random.randint(0,9,(MATRIX_M,MATRIX_N))
n2Now = np.random.randint(0,9,(MATRIX_M,MATRIX_N))
NumAtROutPre = np.random.randint(0,2,(MATRIX_M,MATRIX_N))
End = MATRIX_N
Start = 0

print("NumAtROutPre: \n",NumAtROutPre)
print("retNowR: \n",retNowR)
print("retNowI: \n",retNowI)
print("n1Now: \n",n1Now)
print("n2Now: \n",n2Now)

# configure threads and blocks
threadsperblock = (32,1,1)
blockspergrid_x = int(math.ceil(retNowR.shape[0]/threadsperblock[0]))
blockspergrid = (blockspergrid_x,1,1)
print("blockspergrid is",blockspergrid)

start_cuda = time.time()
# set up stream
stream = 0
NumAtROutPre_global_mem = cuda.to_device(NumAtROutPre, stream)
retNowR_global_mem = cuda.to_device(retNowR, stream)
retNowI_global_mem = cuda.to_device(retNowI, stream)
n1Now_global_mem = cuda.to_device(n1Now, stream)
n2Now_global_mem = cuda.to_device(n2Now, stream)
reduced_global_mem = cuda.device_array(((End-Start),4),stream=stream)

# launch  kernel
start_kernel = time.time()

filter_and_sum[threadsperblock,blockspergrid,stream](retNowR_global_mem,retNowI_global_mem,n1Now_global_mem,n2Now_global_mem,NumAtROutPre_global_mem,reduced_global_mem,End,Start)

end_kernel = time.time()
# move results back to host
start_copy_to_host = time.time()
reduced = reduced_global_mem.copy_to_host(stream=stream)

end_copy_to_host = time.time()

end_cuda = time.time()

start_cpu_time = time.time()
[retofROutRPre, retofROutIPre, n1ofROutPre,n2ofROutPre] = AveragesOnShellsInnerLogicC(retNowR,retNowI,n1Now, n2Now,Start, End ,NumAtROutPre)
end_cpu_time = time.time()
print("Expected: ")
print([retofROutRPre, retofROutIPre, n1ofROutPre,n2ofROutPre][0])

print("Got: ")
print(reduced)

print("Time for Total CUDA calculation: ",end_cuda - start_cuda)
print("Time to run kernel: ",end_kernel - start_kernel)
print("Time to copy to host: ",end_copy_to_host - start_copy_to_host)

print("Time to compute on CPU: ",end_cpu_time - start_cpu_time)

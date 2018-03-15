from __future__ import division
import math
import numba
from numba import cuda,float32,float64
import numpy as np
import os
import time
import pdb

@cuda.jit('float32(float32,float32,float32,float32,float32,float32)',device=True)
def calculate_distance(p1_0,p1_1,p1_2,p2_0,p2_1,p2_2):
        return math.sqrt( (p1_0 - p2_0)**2 + (p1_1 - p2_1)**2 + (p1_2 - p2_2)**2)

@cuda.jit
def calcNeighborsKernel(points_array,\
                    center,\
                    cutoff_fsc,\
                    cutoff_binarize,\
                    highpassfilter,\
                    min_cutoff,\
                    inputmrc,\
                    memory_inmrc_thresholded,\
                    memory_inmrc_thresholdedbinarized,\
                    outarraythresholded,\
                    outarraythresholdedbinarized):
    # NOTE: using i as the thread index, NOT x, so as to mimic original CPU code more easily
    i = cuda.grid(1)

    if i >= points_array.shape[0]:
        return
    point = points_array[i]
    x = int(point[1])
    y = int(point[2])
    z = int(point[3])
    twentysix_neighboring_points = cuda.local.array((26,3),dtype=numba.int64)
    twentysix_neighboring_points_dist = cuda.local.array(26,dtype=numba.float32)

    if float32(point[0]) < float32(highpassfilter):
        outarraythresholded[x][y][z] = inputmrc[x][y][z]
        outarraythresholdedbinarized[x][y][z] = 1
        memory_inmrc_thresholded[x][y][z] = 1
        memory_inmrc_thresholdedbinarized[x][y][z] = 1

    elif float32(memory_inmrc_thresholded[x][y][z]) < float32(min_cutoff):
        outarraythresholded[x][y][z] = 0
        outarraythresholdedbinarized[x][y][z] = 0
        memory_inmrc_thresholded[x][y][z] = 0
        memory_inmrc_thresholdedbinarized[x][y][z] = 0

    else:
        twentysix_neighboring_points[0] = (x-1,y,z)
        twentysix_neighboring_points[1] = (x,y-1,z)
        twentysix_neighboring_points[2] = (x,y,z-1)
        twentysix_neighboring_points[3] = (x-1,y-1,z)
        twentysix_neighboring_points[4] = (x-1,y,z-1)
        twentysix_neighboring_points[5] = (x,y-1,z-1)
        twentysix_neighboring_points[6] = (x-1,y-1,z-1)

        twentysix_neighboring_points[7] = (x+1,y,z)
        twentysix_neighboring_points[8] = (x,y+1,z)
        twentysix_neighboring_points[9] = (x,y,z+1)
        twentysix_neighboring_points[10] = (x+1,y+1,z)
        twentysix_neighboring_points[11] = (x+1,y,z+1)
        twentysix_neighboring_points[12] = (x,y+1,z+1)
        twentysix_neighboring_points[13] = (x+1,y+1,z+1)

        twentysix_neighboring_points[14] = (x+1,y-1,z)
        twentysix_neighboring_points[15] = (x+1,y,z-1)
        twentysix_neighboring_points[16] = (x+1,y-1,z-1)
        twentysix_neighboring_points[17] = (x-1,y+1,z)
        twentysix_neighboring_points[18] = (x,y+1,z-1)
        twentysix_neighboring_points[19] = (x-1,y+1,z-1)
        twentysix_neighboring_points[20] = (x-1,y,z+1)
        twentysix_neighboring_points[21] = (x,y-1,z+1)
        twentysix_neighboring_points[22] = (x-1,y-1,z+1)

        twentysix_neighboring_points[23] = (x+1,y+1,z-1)
        twentysix_neighboring_points[24] = (x+1,y-1,z+1)
        twentysix_neighboring_points[25] = (x-1,y+1,z+1)


        twentysix_neighboring_points_dist[0] = calculate_distance(float32(x-1),float32(y),float32(z),center[0],center[1],center[2])
        twentysix_neighboring_points_dist[1] = calculate_distance(float32(x),float32(y-1),float32(z),center[0],center[1],center[2])
        twentysix_neighboring_points_dist[2] = calculate_distance(float32(x),float32(y),float32(z-1),center[0],center[1],center[2])
        twentysix_neighboring_points_dist[3] = calculate_distance(float32(x-1),float32(y-1),float32(z),center[0],center[1],center[2])
        twentysix_neighboring_points_dist[4] = calculate_distance(float32(x-1),float32(y),float32(z-1),center[0],center[1],center[2])
        twentysix_neighboring_points_dist[5] = calculate_distance(float32(x),float32(y-1),float32(z-1),center[0],center[1],center[2])
        twentysix_neighboring_points_dist[6] = calculate_distance(float32(x-1),float32(y-1),float32(z-1),center[0],center[1],center[2])

        twentysix_neighboring_points_dist[7] = calculate_distance(float32(x+1),float32(y),float32(z),center[0],center[1],center[2])
        twentysix_neighboring_points_dist[8] = calculate_distance(float32(x),float32(y+1),float32(z),center[0],center[1],center[2])
        twentysix_neighboring_points_dist[9] = calculate_distance(float32(x),float32(y),float32(z+1),center[0],center[1],center[2])
        twentysix_neighboring_points_dist[10] = calculate_distance(float32(x+1),float32(y+1),float32(z),center[0],center[1],center[2])
        twentysix_neighboring_points_dist[11] = calculate_distance(float32(x+1),float32(y),float32(z+1),center[0],center[1],center[2])
        twentysix_neighboring_points_dist[12] = calculate_distance(float32(x),float32(y+1),float32(z+1),center[0],center[1],center[2])
        twentysix_neighboring_points_dist[13] = calculate_distance(float32(x+1),float32(y+1),float32(z+1),center[0],center[1],center[2])

        twentysix_neighboring_points_dist[14] = calculate_distance(float32(x+1),float32(y-1),float32(z),center[0],center[1],center[2])
        twentysix_neighboring_points_dist[15] = calculate_distance(float32(x+1),float32(y),float32(z-1),center[0],center[1],center[2])
        twentysix_neighboring_points_dist[16] = calculate_distance(float32(x+1),float32(y-1),float32(z-1),center[0],center[1],center[2])
        twentysix_neighboring_points_dist[17] = calculate_distance(float32(x-1),float32(y+1),float32(z),center[0],center[1],center[2])
        twentysix_neighboring_points_dist[18] = calculate_distance(float32(x),float32(y+1),float32(z-1),center[0],center[1],center[2])
        twentysix_neighboring_points_dist[19] = calculate_distance(float32(x-1),float32(y+1),float32(z-1),center[0],center[1],center[2])
        twentysix_neighboring_points_dist[20] = calculate_distance(float32(x-1),float32(y),float32(z+1),center[0],center[1],center[2])
        twentysix_neighboring_points_dist[21] = calculate_distance(float32(x),float32(y-1),float32(z+1),center[0],center[1],center[2])
        twentysix_neighboring_points_dist[22] = calculate_distance(float32(x-1),float32(y-1),float32(z+1),center[0],center[1],center[2])

        twentysix_neighboring_points_dist[23] = calculate_distance(float32(x+1),float32(y+1),float32(z-1),center[0],center[1],center[2])
        twentysix_neighboring_points_dist[24] = calculate_distance(float32(x+1),float32(y-1),float32(z+1),center[0],center[1],center[2])
        twentysix_neighboring_points_dist[25] = calculate_distance(float32(x-1),float32(y+1),float32(z+1),center[0],center[1],center[2])

        closest_point = twentysix_neighboring_points[0]
        closest_dist = float32(twentysix_neighboring_points_dist[0])

        for j,k in enumerate(twentysix_neighboring_points_dist):
            if float32(k) < float32(closest_dist):

                closest_dist = k
                closest_point = twentysix_neighboring_points[j]

        closest_x = closest_point[0]
        closest_y = closest_point[1]
        closest_z = closest_point[2]

        if float32(memory_inmrc_thresholded[x][y][z]) < float32(cutoff_fsc):
                outarraythresholded[x][y][z] = 0
                memory_inmrc_thresholded[x][y][z] = 0
        elif float32(memory_inmrc_thresholded[closest_x][closest_y][closest_z]) < float32(cutoff_fsc):
                outarraythresholded[x][y][z] = 0
                memory_inmrc_thresholded[x][y][z] = 0
        else:
                outarraythresholded[x][y][z] = inputmrc[x][y][z]

        if float32(memory_inmrc_thresholdedbinarized[x][y][z]) < float32(cutoff_binarize):
                outarraythresholdedbinarized[x][y][z] = 0
                memory_inmrc_thresholdedbinarized[x][y][z] = 0
        elif float32(memory_inmrc_thresholdedbinarized[closest_x][closest_y][closest_z]) < float32(cutoff_binarize):
                outarraythresholdedbinarized[x][y][z] = 0
                memory_inmrc_thresholdedbinarized[x][y][z] = 0
        else:
                outarraythresholdedbinarized[x][y][z] = 1

def calcNeighbors(  points_array,\
                    center,\
                    cutoff_fsc,\
                    cutoff_binarize,\
                    highpassfilter,\
                    min_cutoff,\
                    inputmrc,\
                    memory_inmrc_thresholded,\
                    memory_inmrc_thresholdedbinarized,\
                    outarraythresholded,\
                    outarraythresholdedbinarized):
    threadsperblock = (512,1,1)
    blockspergrid = (math.ceil(points_array.shape[0]/threadsperblock[0]),1,1)
    start_time = time.time()
    g_points_array = cuda.to_device(points_array)
    stream = cuda.stream()
    g_memory_inmrc_thresholded = cuda.to_device(memory_inmrc_thresholded)
    g_memory_inmrc_thresholdedbinarized = cuda.to_device(memory_inmrc_thresholdedbinarized)
    g_outarraythresholded = cuda.to_device(outarraythresholded)
    g_outarraythresholdedbinarized = cuda.to_device(outarraythresholdedbinarized)
    g_inputmrc = cuda.to_device(np.copy(inputmrc))
    c = cuda.to_device(np.array([np.float32(center[0]),np.float32(center[1]),np.float32(center[2])]))
    cuda.synchronize()
    calcNeighborsKernel[blockspergrid,threadsperblock](g_points_array,\
                                                        c,\
                                                        cutoff_fsc,\
                                                        cutoff_binarize,\
							highpassfilter,\
							min_cutoff,\
                                                        g_inputmrc,\
							g_memory_inmrc_thresholded,\
							g_memory_inmrc_thresholdedbinarized,\
							g_outarraythresholded,\
							g_outarraythresholdedbinarized)
    stream = cuda.stream()
    outarraythresholded = g_outarraythresholded.copy_to_host(stream=stream)
    outarraythresholdedbinarized = g_outarraythresholdedbinarized.copy_to_host(stream=stream)
    print("Time to complete calcNeighbors on GPU is ",time.time() - start_time)   

    cuda.synchronize()
    return outarraythresholded,outarraythresholdedbinarized

@cuda.jit
def calcDistanceKernel(boxsize,center,out):
    x,y,z = cuda.grid(3)

    blockId = cuda.blockIdx.x \
            + cuda.blockIdx.y * cuda.gridDim.x \
            + cuda.gridDim.x * cuda.gridDim.y * cuda.blockIdx.z;

    idx = blockId * (cuda.blockDim.x *cuda.blockDim.y*cuda.blockDim.z) \
          + (cuda.threadIdx.z * (cuda.blockDim.x * cuda.blockDim.y)) \
          + (cuda.threadIdx.y * cuda.blockDim.x) \
          + cuda.threadIdx.x;

    if idx >= out.shape[0]:
        return
    out[idx,1] = (idx // (boxsize*boxsize))
    out[idx,2] = (idx %(boxsize*boxsize)) // boxsize
    out[idx,3] = idx % boxsize

    out[idx,0] = math.sqrt( (out[idx,1] - numba.float32(center[0]))**2 + (out[idx,2] - numba.float32(center[1]))**2 + (out[idx,3] - numba.float32(center[2]))**2 )



def calcDistance(boxsize,center):
    c = np.array([center[0],center[1],center[2]])
    threadsperblock = (32,1,1)
    start_time = time.time()
    blockspergrid = (math.ceil(boxsize/threadsperblock[0]), \
                     math.ceil(boxsize/threadsperblock[1]), \
                     math.ceil(boxsize/threadsperblock[2]))
    cuda.synchronize()
    out = np.zeros((boxsize*boxsize*boxsize,4))
    gout = cuda.device_array((boxsize*boxsize*boxsize,4))
    calcDistanceKernel[blockspergrid,threadsperblock](boxsize,c,gout)

    out = gout.copy_to_host()
    print("Time to compute calcDistance on GPU is ",time.time() - start_time)
    return out


def sum_rows(\
             NumAtROutPre_global_mem,Start,End):

    threadsperblock = (32,1,1)
    blockspergrid_x = (math.ceil(NumAtROutPre_global_mem.shape[0]/threadsperblock[0]))
    blockspergrid = (blockspergrid_x,1,1)

    sum_array_global_mem = cuda.device_array((End-Start))
    sum_rowsKernel[blockspergrid,threadsperblock](NumAtROutPre_global_mem,sum_array_global_mem,Start,End)

    return sum_array_global_mem.copy_to_host()

@cuda.jit
def sum_rowsKernel(\
             NumAtROutPre_global_mem,
             sum_array_global_mem,Start,End):

  x = cuda.grid(1)
  if (x>= (NumAtROutPre_global_mem.shape[1])):
    return

  temp_sum = 0
  for i in range(NumAtROutPre_global_mem.shape[0]):
    temp_sum += NumAtROutPre_global_mem[i,x]

  sum_array_global_mem[x] = temp_sum

@cuda.jit
def filter_and_sum(\
                   retofRR_global_mem,\
                   retofRI_global_mem,\
                   n1ofR_global_mem,\
                   n2ofR_global_mem,\
                   NumAtROutPre,\
                   reduced,\
                   End,\
                   Start,\
                   NumOnSurf,\
                   r):
    #retNowR = retofRR_global_mem[r][:NumOnSurf]
    #retNowI = retofRI_global_mem[r][:NumOnSurf]
    #n1Now = n1ofR_global_mem[r][:NumOnSurf]
    #n2Now = n2ofR_global_mem[r][:NumOnSurf]

    x = cuda.grid(1)
    if (x >= (End - Start)):
      return
    retofROutRPre = 0
    retofROutIPre = 0
    n1ofROutPre = 0
    n2ofROutPre = 0

    MultVec = NumAtROutPre[:,x]

    for i in range(MultVec.shape[0]):

 
        retofROutRPre += retofRR_global_mem[r][i]*MultVec[i]
        retofROutIPre += retofRI_global_mem[r][i]*MultVec[i]
        n1ofROutPre += n1ofR_global_mem[r][i]*MultVec[i]
        n2ofROutPre += n2ofR_global_mem[r][i]*MultVec[i]

    reduced[0,x] = retofROutRPre
    reduced[1,x] = retofROutIPre
    reduced[2,x] = n1ofROutPre
    reduced[3,x] = n2ofROutPre



# CUDA kernel
@cuda.jit()
def cuda_calcProd11(\
                    kXofR_global_mem,\
                    kYofR_global_mem,\
                    kZofR_global_mem,\
                    Prod11,\
                    NumOnSurf,\
                    r):
    """Calculate partial product Prod11
    """
    x = cuda.grid(1)
    #if x >= kXofR_global_mem[r][:NumOnSurf].shape[0]:
    if x >= kXofR_global_mem[:NumOnSurf].shape[0]:
        return

    #Prod11[x] = kXofR_global_mem[r][x]*kXofR_global_mem[r][x] +\
    #            kYofR_global_mem[r][x]*kYofR_global_mem[r][x] +\
    #            kZofR_global_mem[r][x]*kZofR_global_mem[r][x]

    Prod11[x] = kXofR_global_mem[x]*kXofR_global_mem[x] +\
                kYofR_global_mem[x]*kYofR_global_mem[x] +\
                kZofR_global_mem[x]*kZofR_global_mem[x]

@cuda.jit()
def cuda_calcInner2(\
                    kXofR_global_mem,\
                    kYofR_global_mem,\
                    kZofR_global_mem,\
                    Prod11,\
                    C,\
                    End,\
                    Start,\
                    Thresh,\
                    NumOnSurf,\
                    r):
    """Calculate Prod12
    """
    #kXNow = kXofR_global_mem[r][:NumOnSurf]
    #kYNow = kYofR_global_mem[r][:NumOnSurf]
    #kZNow = kZofR_global_mem[r][:NumOnSurf]

    x = cuda.grid(1)
    if x >= Prod11.shape[0]:
        return

    Thresh2 = Thresh*Thresh

    for i in range(End-Start):
        C[x,i] = 0
        if Prod11[x]==0:
            return

        else:

            #Prod12 = kX1*kX2 + kY1*kY2 + kZ1*kZ2
            #Prod12 = kXNow[x]*kXNow[(i+Start)] + kYNow[x]*kYNow[(i+Start)] + kZNow[x]*kZNow[(i+Start)]
            Prod12 = kXofR_global_mem[x]*kXofR_global_mem[(i+Start)] + \
                     kYofR_global_mem[x]*kYofR_global_mem[(i+Start)] + \
                     kZofR_global_mem[x]*kZofR_global_mem[(i+Start)]


            #Prod22 = kX2*kX2 + kY2*kY2 + kZ2*kZ2
            #Prod22 = kXNow[(i+Start)]*kXNow[(i+Start)] + kYNow[(i+Start)]*kYNow[(i+Start)] + kZNow[(i+Start)]*kZNow[(i+Start)]
            Prod22 = kXofR_global_mem[(i+Start)]*kXofR_global_mem[(i+Start)] + \
                     kYofR_global_mem[(i+Start)]*kYofR_global_mem[(i+Start)] + \
                     kZofR_global_mem[(i+Start)]*kZofR_global_mem[(i+Start)]
            if Prod22==0:
                return
            else:

                Inner2 = Prod12*Prod12/(Prod11[x]*Prod22)

                if Inner2>Thresh2:
                    C[x,i] = 1

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

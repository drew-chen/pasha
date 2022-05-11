#include <cuda.h>
#include <iostream>
#include "hitting_num.cuh"

using namespace std;
using unsigned_int = uint64_t;
using byte = uint8_t;

#define NUM_THREADS 256


__constant__ unsigned_int d_vertexExp;
unsigned_int vertexExp;
unsigned_int L;
unsigned_int dSize;
// total number of edges before any removal (does not change)
unsigned_int numEdges;

// L-k+1 rows, vertexExp columns
byte* edgeArray_gpu;
float* Fprev_gpu;
float* Fcurr_gpu;
// a host pointer pointing to the copy of D on the gpu
float* D_gpu;


// assumes already inited
__device__ float D_get(float* D, int row, int col) {
    return D[row*d_vertexExp + col];
}
__device__ void D_set(float* D, int row, int col, float val) {
    D[row*d_vertexExp + col] = val;
}

void initHittingNum(unsigned_int LParam, unsigned_int vertexExpParam, unsigned_int dSizeParam, unsigned_int numEdgesParam, byte* edgeArray) {
    L = LParam;
    vertexExp = vertexExpParam;
    dSize = dSizeParam;
    numEdges = numEdgesParam;

    cudaMalloc((void**)&edgeArray_gpu, numEdges*sizeof(byte)); 
    cudaMalloc((void**)&D_gpu, dSize*sizeof(float));
    cudaMalloc((void**)&Fprev_gpu, vertexExp*sizeof(float));
    cudaMalloc((void**)&Fcurr_gpu, vertexExp*sizeof(float));

    // MemcpyToSymbol is for consts
    cudaMemcpyToSymbol(d_vertexExp, &vertexExpParam, sizeof(unsigned_int));
}

void finalizeHittingNum() {
    cudaFree(edgeArray_gpu);
    cudaFree(D_gpu);
    cudaFree(Fprev_gpu);
    cudaFree(Fcurr_gpu);
}

__global__ void setInitialDFprev_gpu(float* D, float* Fprev) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= d_vertexExp) return;
    D_set(D, 0, tid, 1.4e-45);
    Fprev[tid] = 1.4e-45;
}

__global__ void calcNumStartingPathsOneIter_gpu(float* D, byte* edgeArray, int j) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= d_vertexExp) return;
    unsigned_int vertexExp2 = d_vertexExp * 2;
    unsigned_int vertexExp3 = d_vertexExp * 3;
    
    D_set(D, j, i, 
        edgeArray[i]*D_get(D, j-1, (i >> 2))
            + edgeArray[i + d_vertexExp]*D_get(D, j-1,((i + d_vertexExp) >> 2))
            + edgeArray[i + vertexExp2]*D_get(D, j-1,((i + vertexExp2) >> 2))
            + edgeArray[i + vertexExp3]*D_get(D, j-1,((i + vertexExp3) >> 2))
    );
}

void calcNumStartingPaths(byte* edgeArray, float* D, float* Fprev) {
    /**
    * This function generates D. D(v,i): # of i long paths starting from v after decycling
    */
    // want tid range [0, vertexExp)
    // edgeArray changes outside of this func so must cpy
    cudaMemcpy(edgeArray_gpu, edgeArray, numEdges*sizeof(byte), cudaMemcpyHostToDevice);

    int grid_size = 1 + ((vertexExp - 1) / NUM_THREADS);
    setInitialDFprev_gpu<<<grid_size, NUM_THREADS>>>(D_gpu, Fprev_gpu); 

    // THE ISSUE IS HERE
    // // TODO: replace loop with this https://towardsdatascience.com/gpu-optimized-dynamic-programming-8d5ba3d7064f
    for (unsigned_int j = 1; j <= L; j++) {
        calcNumStartingPathsOneIter_gpu<<<grid_size, NUM_THREADS>>>(D_gpu, edgeArray_gpu, j); 
    }



    cudaMemcpy(D, D_gpu, dSize*sizeof(float),  cudaMemcpyDeviceToHost);
    cudaMemcpy(Fprev, Fprev_gpu, vertexExp*sizeof(float),  cudaMemcpyDeviceToHost);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) 
    //     printf("Error: %s\n", cudaGetErrorString(err));
    // printf("---END---\n");
}

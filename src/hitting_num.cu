#include <cuda.h>
#include <iostream>
#include "hitting_num.cuh"

using namespace std;
using unsigned_int = uint64_t;
using byte = uint8_t;

#define NUM_THREADS 256
#define ALPHABET_SIZE 4
// 2^(−149)
#define MINPOS_FLOAT 1.4e-45

__constant__ unsigned_int vertexExp_gpu;
unsigned_int vertexExp;
unsigned_int L;
unsigned_int dSize;
// total number of edges before any removal (does not change)
unsigned_int numEdges;

// L-k+1 rows, vertexExp columns
byte* edgeArray_gpu;
float* Fprev_gpu;
float* Fcurr_gpu;
float* D_gpu;
double* hittingNumArray_gpu;

__device__ float D_get(float* D, int row, int col) {
    return D[row*vertexExp_gpu + col];
}
__device__ void D_set(float* D, int row, int col, float val) {
    D[row*vertexExp_gpu + col] = val;
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
    cudaMalloc((void**)&hittingNumArray_gpu, numEdges*sizeof(double));

    // MemcpyToSymbol is for consts
    cudaMemcpyToSymbol(vertexExp_gpu, &vertexExpParam, sizeof(unsigned_int));
}

void finalizeHittingNum() {
    cudaFree(edgeArray_gpu);
    cudaFree(D_gpu);
    cudaFree(Fprev_gpu);
    cudaFree(Fcurr_gpu);
}

__global__ void setInitialDFprev_gpu(float* D, float* Fprev) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= vertexExp_gpu) return;
    D_set(D, 0, i, MINPOS_FLOAT);
    Fprev[i] = MINPOS_FLOAT;
}

__global__ void calcNumStartingPathsOneIter_gpu(float* D, byte* edgeArray, int j) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= vertexExp_gpu) return;
    unsigned_int vertexExp2 = vertexExp_gpu * 2;
    unsigned_int vertexExp3 = vertexExp_gpu * 3;
    
    D_set(D, j, i, 
        edgeArray[i]*D_get(D, j-1, (i >> 2))
            + edgeArray[i + vertexExp_gpu]*D_get(D, j-1,((i + vertexExp_gpu) >> 2))
            + edgeArray[i + vertexExp2]*D_get(D, j-1,((i + vertexExp2) >> 2))
            + edgeArray[i + vertexExp3]*D_get(D, j-1,((i + vertexExp3) >> 2))
    );
}

void calcNumStartingPaths(byte* edgeArray, float* D, float* Fprev) {
    /**
    * This function generates D. D(v,i): # of i long paths starting from v after decycling
    */
    // want tid range [0, vertexExp)

    int grid_size = 1 + ((vertexExp - 1) / NUM_THREADS);
    setInitialDFprev_gpu<<<grid_size, NUM_THREADS>>>(D_gpu, Fprev_gpu); 

    // TODO: replace loop with this https://towardsdatascience.com/gpu-optimized-dynamic-programming-8d5ba3d7064f
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

__global__ void calcCurrNumEndingPaths(byte* edgeArray, float* Fprev, float* Fcurr) {
    /**
    * This function generates F. F(v,i): # of i long paths ending at v after decycling
    */
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= vertexExp_gpu) return;

    unsigned_int vertexExpMask = vertexExp_gpu - 1;

    unsigned_int index = i*4;
    Fcurr[i] = (edgeArray[index]*Fprev[index & vertexExpMask]
         + edgeArray[index + 1]*Fprev[(index + 1) & vertexExpMask]
         + edgeArray[index + 2]*Fprev[(index + 2) & vertexExpMask]
         + edgeArray[index + 3]*Fprev[(index + 3) & vertexExpMask]);

}

__global__ void calcHittingNums(double* hittingNums, byte* edgeArray, float* D, float* Fprev, int dRow, unsigned_int numEdges) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= numEdges) return;
    if (edgeArray[i] == 0) {
        hittingNums[i] = 0;
        return;
    }
    hittingNums[i] += (Fprev[i % vertexExp_gpu] / MINPOS_FLOAT)
        * (D_get(D, dRow, i / ALPHABET_SIZE) / MINPOS_FLOAT);
}

void calcNumPaths(byte* edgeArray, double* hittingNumArray) {

    // edgeArray changes outside of this func so must cpy
    cudaMemcpy(edgeArray_gpu, edgeArray, numEdges*sizeof(byte), cudaMemcpyHostToDevice);
    cudaMemset(hittingNumArray_gpu, 0, numEdges*sizeof(double));
    calcNumStartingPaths(edgeArray_gpu, D_gpu, Fprev_gpu);

    int grid_size;
    for (int l = L - 1; l >= 0; --l) {
        grid_size = 1 + ((vertexExp - 1) / NUM_THREADS);
        calcCurrNumEndingPaths<<<grid_size, NUM_THREADS>>>(edgeArray_gpu, Fprev_gpu, Fcurr_gpu);

        grid_size = 1 + ((numEdges - 1) / NUM_THREADS);
        calcHittingNums<<<grid_size, NUM_THREADS>>>(hittingNumArray_gpu,
            edgeArray_gpu,
            D_gpu,
            Fprev_gpu,
            l,
            numEdges
        );
        cudaMemcpy(Fprev_gpu, Fcurr_gpu, vertexExp*sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(edgeArray, edgeArray_gpu, numEdges*sizeof(byte), cudaMemcpyDeviceToHost);
    cudaMemcpy(hittingNumArray, hittingNumArray_gpu, numEdges*sizeof(double), cudaMemcpyDeviceToHost);
}

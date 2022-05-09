#include <cuda.h>
#include "hitting_num.cuh"

using namespace std;
using unsigned_int = uint64_t;
using byte = uint8_t;

#define NUM_THREADS 256


__constant__ unsigned_int d_vertexExp;
unsigned_int vertexExp;
unsigned_int L;
unsigned_int dSize;

// L-k+1 rows, vertexExp columns
byte* edgeArray_gpu;
float* Fprev_gpu;
float* Fcurr_gpu;
// a host pointer pointing to the copy of D on the gpu
float* D_gpu;
/*
A device pointer pointing to D_gpu.
These consts are created to avoid repeated kernel params.
*/
__constant__ float* d_D_gpu;
__constant__ byte* d_edgeArray_gpu;
__constant__ float* d_Fprev_gpu;
__constant__ float* d_Fcurr_gpu;


// assumes already inited
__device__ float D_get(int row, int col) {
    return d_D_gpu[row*d_vertexExp + col];
}
__device__ void D_set(int row, int col, float val) {
    d_D_gpu[row*d_vertexExp + col] = val;
}

void init_hitting_num(unsigned_int LParam, unsigned_int vertexExpParam, byte* edgeArray, unsigned_int numEdges, float* D, unsigned_int dSizeParam) {
    vertexExp = vertexExpParam;
    L = LParam;
    dSize = dSizeParam;

    cudaMalloc((void**)&edgeArray_gpu, numEdges*sizeof(byte)); 
    cudaMalloc((void**)&D_gpu, dSize*sizeof(float));
    cudaMalloc((void**)&Fprev_gpu, numEdges*sizeof(float));
    cudaMalloc((void**)&Fcurr_gpu, numEdges*sizeof(float));

    cudaMemcpy(edgeArray_gpu, edgeArray, numEdges*sizeof(byte), cudaMemcpyHostToDevice);
    cudaMemcpy(D_gpu, D, dSize*sizeof(float), cudaMemcpyHostToDevice);

    // MemcpyToSymbol is for consts
    cudaMemcpyToSymbol(d_vertexExp, &vertexExpParam, sizeof(unsigned_int));
    cudaMemcpyToSymbol(d_D_gpu, &D_gpu, sizeof(float *));
    cudaMemcpyToSymbol(d_edgeArray_gpu, &edgeArray_gpu, sizeof(byte *));
    cudaMemcpyToSymbol(d_Fprev_gpu, &Fprev_gpu, sizeof(float *));
    cudaMemcpyToSymbol(d_Fcurr_gpu, &Fcurr_gpu, sizeof(float *));

}

void finalize_hitting_num() {
    cudaFree(edgeArray_gpu);
    cudaFree(D_gpu);
    cudaFree(Fprev_gpu);
    cudaFree(Fcurr_gpu);
}

__global__ void set_initial_D_Fprev_gpu() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= d_vertexExp) return;
    D_set(0, tid, 1.4e-45);
    d_Fprev_gpu[tid] = 1.4e-45;
}

__global__ void calc_num_starting_paths_one_iter_gpu(int j) {
    unsigned_int vertexExp2 = d_vertexExp * 2;
    unsigned_int vertexExp3 = d_vertexExp * 3;

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    D_set(
        j, 
        i, 
        d_edgeArray_gpu[i]*D_get(j-1, (i >> 2)) + d_edgeArray_gpu[i + d_vertexExp]*D_get(j-1,((i + d_vertexExp) >> 2)) + d_edgeArray_gpu[i + vertexExp2]*D_get(j-1,((i + vertexExp2) >> 2)) + d_edgeArray_gpu[i + vertexExp3]*D_get(j-1,((i + vertexExp3) >> 2))
    );
}

void calc_num_starting_paths(float* host_D) {
  /**
   * This function generates D.
   *
   * @param L: sequence length, vertexExp: pow(ALPHABET_SIZE, k-1),
   *  edgeArray: 1 if edge exists, 0 else,
   *  D(v,i): # of i long paths starting from v after decycling
   */
   // want tid range [0, vertexExp)

   
    int grid_size = 1 + ((vertexExp - 1) / NUM_THREADS);
    set_initial_D_Fprev_gpu<<<grid_size, NUM_THREADS>>>(); 
    for (unsigned_int j = 1; j <= L; j++) {
        calc_num_starting_paths_one_iter_gpu<<<grid_size, NUM_THREADS>>>(j); 
    }
    cudaMemcpy(host_D, D_gpu, dSize*sizeof(float),  cudaMemcpyDeviceToHost);
}

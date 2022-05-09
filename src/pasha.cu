#include <cuda.h>

using namespace std;
using unsigned_int = uint64_t;
using byte = uint8_t;

#define NUM_THREADS 256


__constant__ int* d_edgeArray;
__constant__ unsigned_int d_vertexExp;
unsigned_int vertexExp;
// L-k+1 rows, vertexExp columns
__device__ float* d_D;
__device__ float* d_Fprev;
__device__ float* d_Fcurr;

// assumes already inited
__device__ float D_get(int row, int col) {
  return d_D[row*d_vertexExp + col];
}
__device__ void D_set(int row, int col, float val) {
  d_D[row*d_vertexExp + col] = val;
}

void init(unsigned_int L, unsigned_int vertexExpParam, byte* edgeArray, unsigned_int numEdges, float* D, unsigned_int dSize) {
  vertexExp = vertexExpParam;
  
  cudaMalloc((void**)&d_edgeArray, numEdges*sizeof(byte)); 
  cudaMalloc((void**)&d_D, dSize*sizeof(float));
  cudaMalloc((void**)&d_Fprev, numEdges*sizeof(float));
  cudaMalloc((void**)&d_Fcurr, numEdges*sizeof(float));

  // MemcpyToSymbol is for consts
  cudaMemcpyToSymbol(d_vertexExp, vertexExpParam, sizeof(unsigned_int));
  cudaMemcpyToSymbol(d_edgeArray, edgeArray, numEdges*sizeof(byte));
  cudaMemcpy(d_D, D, dSize*sizeof(float), cudaMemcpyHostToDevice);
}

__device__ void set_initial_D_Fprev_gpu() {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= d_vertexExp) return;
  D_set(0, tid, 1.4e-45);
  d_Fprev[tid] = 1.4e-45;
}

__device__ calc_num_starting_paths_one_iter_gpu(int j) {
  unsigned_int vertexExp2 = d_vertexExp * 2;
  unsigned_int vertexExp3 = d_vertexExp * 3;

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  D_set(
      j, 
      i, 
      edgeArray[i]*D_get(j-1, (i >> 2)) + edgeArray[i + d_vertexExp]*D_get(j-1,((i + d_vertexExp) >> 2)) + edgeArray[i + vertexExp2]*D_get(j-1,((i + vertexExp2) >> 2)) + edgeArray[i + vertexExp3]*D_get(j-1,((i + vertexExp3) >> 2))
  );
}

__global__ calc_num_starting_paths(unsigned_int L, byte* edgeArray, float** D) {
  /**
   * This function generates D.
   *
   * @param L: sequence length, vertexExp: pow(ALPHABET_SIZE, k-1),
   *  edgeArray: 1 if edge exists, 0 else,
   *  D(v,i): # of i long paths starting from v after decycling
   */
   // want tid range [0, vertexExp)
   int num_blocks = 1 + ((vertexExp - 1) / NUM_THREADS)
   set_initial_D_Fprev_gpu<<<num_blocks, NUM_THREADS>>>(); 
   for (unsigned_int j = 1; j <= L; j++) {
     calc_num_starting_paths_gpu<<<num_blocks, NUM_THREADS>>>(j); 
   }
}
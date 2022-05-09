#include <cuda.h>

using namespace std;
using unsigned_int = uint64_t;
using byte = uint8_t;

__constant__ int* d_edgeArray;
__constant__ unsigned_int d_vertexExp;
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

void init(unsigned_int L, unsigned_int vertexExp, byte* edgeArray, unsigned_int numEdges, float* D, unsigned_int dSize) {
  cudaMalloc((void**)&d_edgeArray, numEdges*sizeof(byte)); 
  cudaMalloc((void**)&d_D, dSize*sizeof(float));
  cudaMalloc((void**)&d_Fprev, numEdges*sizeof(float));
  cudaMalloc((void**)&d_Fcurr, numEdges*sizeof(float));

  // MemcpyToSymbol is for consts
  cudaMemcpyToSymbol(d_vertexExp, vertexExpParam, sizeof(unsigned_int));
  cudaMemcpyToSymbol(d_edgeArray, edgeArray, numEdges*sizeof(byte));
  cudaMemcpy(d_D, D, dSize*sizeof(float), cudaMemcpyHostToDevice);
}


__device__ calc_num_starting_paths_gpu() {
  unsigned_int vertexExp2 = vertexExp * 2;
  unsigned_int vertexExp3 = vertexExp * 3;
  #pragma omp parallel for num_threads(threads)
  for (unsigned_int i = 0; i < vertexExp; i++) {D_set(0, i, 1.4e-45); Fprev[i] = 1.4e-45;}
  for (unsigned_int j = 1; j <= L; j++) {
      #pragma omp parallel for num_threads(threads)
      for (unsigned_int i = 0; i < vertexExp; i++) {
          D_set(
              j, 
              i, 
              edgeArray[i]*D_get(j-1, (i >> 2)) + edgeArray[i + vertexExp]*D_get(j-1,((i + vertexExp) >> 2)) + edgeArray[i + vertexExp2]*D_get(j-1,((i + vertexExp2) >> 2)) + edgeArray[i + vertexExp3]*D_get(j-1,((i + vertexExp3) >> 2))
          );
          //cout << (float)(Dval[j][i] * pow(2, Dexp[j][i])) << endl;
          //D[j][i] = Dval[i];
          //if ((float)(Dval[j][i] * pow(2, Dexp[j][i])) > maxD) maxD = ((float)Dval[j][i] * pow(2, Dexp[j][i])); 
      }
      //if (maxD > std::numeric_limits<half>::max()/4) {
       //   for (unsigned_int i = 0; i < vertexExp; i++) D[j][i] = D[j][i] * 0.5;
      //}


  }
}

__global__ calc_num_starting_paths(unsigned_int L, unsigned_int vertexExp, byte* edgeArray, float** D) {
  /**
   * This function generates D.
   *
   * @param L: sequence length, vertexExp: pow(ALPHABET_SIZE, k-1),
   *  edgeArray: 1 if edge exists, 0 else,
   *  D(v,i): # of i long paths starting from v after decycling
   */

}
#include <cuda.h>

using namespace std;
using unsigned_int = uint64_t;
using byte = uint8_t;

__constant__ int* d_edgeArray;
__constant__ unsigned_int vertexExp;

// L-k+1 rows, vertexExp columns
float** D d_D;
void init(unsigned_int L, unsigned_int vertexExp, byte* edgeArray, unsigned_int, edge_num, float** D) {
  unsigned_int vertexExp2 = vertexExp * 2;
  unsigned_int vertexExp3 = vertexExp * 3;
  cudaMalloc((void**)d_edgeArray, edgeNum*sizeof(byte)) 
  
}

__global__ calc_num_starting_paths_gpu(unsigned_int L, unsigned_int vertexExp, byte* edgeArray, float** D) {
  /**
   * This function generates D.
   *
   * @param L: sequence length, vertexExp: pow(ALPHABET_SIZE, k-1),
   *  edgeArray: 1 if edge exists, 0 else,
   *  D(v,i): # of i long paths starting from v after decycling
   */
  unsigned_int vertexExp2 = vertexExp * 2;
  unsigned_int vertexExp3 = vertexExp * 3;
  cudaMalloc((void**))
  #pragma omp parallel for num_threads(threads)
  for (unsigned_int i = 0; i < vertexExp; i++) {D[0][i] = 1.4e-45; Fprev[i] = 1.4e-45;}
  for (unsigned_int j = 1; j <= L; j++) {
      #pragma omp parallel for num_threads(threads)
      for (unsigned_int i = 0; i < vertexExp; i++) {
          D[j][i] = edgeArray[i]*D[j-1][(i >> 2)] + edgeArray[i + vertexExp]*D[j-1][((i + vertexExp) >> 2)] + edgeArray[i + vertexExp2]*D[j-1][((i + vertexExp2) >> 2)] + edgeArray[i + vertexExp3]*D[j-1][((i + vertexExp3) >> 2)];
      }
  }
  #pragma omp parallel for num_threads(threads)
  for (unsigned_int i = 0; i < (unsigned_int)edgeNum; i++) hittingNumArray[i] = 0;
}
#ifndef __HITTING_NUM_CUH__
#define __HITTING_NUM_CUH__

/** Initialize cuda stuff */
void init_hitting_num_gpu(unsigned_int L, unsigned_int vertexExpParam, byte* edgeArray, unsigned_int numEdges, float* D, unsigned_int dSize);

void calc_num_starting_paths();

/** Cleanup, frees resources used by the device. */
void finalize_hitting_num_gpu();

#endif

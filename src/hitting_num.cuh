#ifndef __HITTING_NUM_CUH__
#define __HITTING_NUM_CUH__

using unsigned_int = uint64_t;
using byte = uint8_t;

/** Initialize cuda stuff */
void initHittingNum(unsigned_int L, unsigned_int vertexExpParam, byte* edgeArray, unsigned_int numEdges, float* D, unsigned_int dSize);

void calcNumStartingPaths(float* D, float* Fprev);

/** Cleanup, frees resources used by the device. */
void finalizeHittingNum();

#endif

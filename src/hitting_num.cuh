#ifndef __HITTING_NUM_CUH__
#define __HITTING_NUM_CUH__

using unsigned_int = uint64_t;
using byte = uint8_t;

/** Initialize cuda stuff */
void initHittingNum(unsigned_int LParam, unsigned_int vertexExpParam, unsigned_int dSizeParam, unsigned_int numEdgesParam, byte* edgeArray);


void calcNumPaths(byte* edgeArray, double* hittingNumArray);

/** Cleanup, frees resources used by the device. */
void finalizeHittingNum();

#endif

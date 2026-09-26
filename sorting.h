#ifndef SORTING_H
#define SORTING_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#include "types.h"
#include "macros.h"

__global__ void reorderParticles(const float4* __restrict__ posMassValsIn, const float3* __restrict__ velValsIn, float4* posMassValsOut, float3* velValsOut, uint32_t* originalIndex, int starCount);

extern "C" {

radixSorter* sorterCreate(int maxCount);

void radixSortPairs(radixSorter* sorter, const uint64_t* keysIn, uint64_t* keysOut,
                                         const uint32_t* valsIn, uint32_t* valsOut,
                                         int numCount);

void sorterDestroyer(radixSorter* s);
}
#endif
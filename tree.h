#ifndef TREE_H
#define TREE_H


#include <stdint.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#include "types.h"

__global__ void mortonEncode(const float4* posMassVals, uint64_t* mortonCodes, uint64_t* originalIndex, int NStars, worldBox rootBox);

extern "C" {

size_t sortQueryTemp(int n);
void sortPairs(uint64_t* keysIn, uint64_t* keysOut,
            uint64_t* valsIn, uint64_t* valsOut,
            int n, void* temp, size_t tempBytes);

}

#endif
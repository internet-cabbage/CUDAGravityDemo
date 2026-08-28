#ifndef TREE_H
#define TREE_H


#include <stdint.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#include "types.h"


#define BOXTHREADS 256
#define BOXBLOCKS 256

__global__ void mortonEncode(const float4* posMassVals, uint64_t* mortonCodes, uint32_t* originalIndex, int NStars, worldBox rootBox);

__global__ void localMinMaxFinder(const float4* __restrict__ posMassVals, unsigned int N, float3* minCorner, float3* maxCorner);

__global__ void globalMinMaxReducer(const float3* __restrict__ minCorner, const float3* __restrict__ maxCorner, int count, float3* outMin, float3* outMax);

__global__ void identifyNodesAtLevel(uint64_t* mortonCodes, int nodeLevel, int starCount, int* nodeFlags);

extern "C" {

typedef struct {
    void* tempStorage;
    size_t tempStorageBytes;
    int capacity;

    uint32_t* flags;
    uint32_t* offsetsPing;
    uint32_t* offsetsPong;
} treeBuilder;

treeBuilder* sumCreate(int maxCount);

void prefixSum(treeBuilder* builder, int* flags, int* offsets, int maxCount);

}

#endif
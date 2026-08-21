#include "tree.h"



// Built in radix sort 
#include <cub/cub.cuh>



/*

# Source - https://stackoverflow.com/a/18529061
# Posted by Gabriel, modified by community. See post 'Timeline' for change history
# Retrieved 2026-08-17, License - CC BY-SA 3.0


I do not know how these magic numbers actually work, but I understand what it produces.

Basicaly if you have a sequence of 21 bits like (110011...) the code then expands these bits by padding two zeros after every bit.
i.e. 110011... becomes 100100000000100100...
*/
__device__ uint64_t expandBits(uint64_t num) {
    num &= 0x1fffff;
    num = (num | num << 32) & 0x1f00000000ffff;
    num= (num| num<< 16) & 0x1f0000ff0000ff;
    num= (num| num<< 8) & 0x100f00f00f00f00f;
    num= (num| num<< 4) & 0x10c30c30c30c30c3;
    num= (num| num<< 2) & 0x1249249249249249;
    return num;
}

// Morton codes are given in order (x_1, y_1, z_1...)
__device__ uint64_t mortonIndex(uint64_t x, uint64_t y, uint64_t z) {
    return (expandBits(x)<<2) | (expandBits(y)<<1) | (expandBits(z));
}

__global__ void mortonEncode(const float4* posMassVals, uint64_t* mortonCodes, uint64_t* originalIndex, int NStars, worldBox rootBox) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    // Checks if the thread is assigned to a star
    if (threadId >= NStars) {
        return;
    }
    float4 pos = posMassVals[threadId];
    /*
    To perform morton ordering I have to map all the positions to a range [0,1.0], and then map that to a series of discrete grid points.

    So to do this I first make the coordinates positive my subtracting the minimum x val of the root box from them, then scaling them down
    */

    uint32_t x = (uint32_t) ((pos.x - rootBox.minX) * rootBox.scale);
    uint32_t y = (uint32_t) ((pos.y - rootBox.minY) * rootBox.scale);
    uint32_t z = (uint32_t) ((pos.z - rootBox.minZ) * rootBox.scale);

    mortonCodes[threadId] = mortonIndex(x,y,z);
    originalIndex[threadId] = threadId;
}

/*
Code modified based off of the example code given in the CUDA documentation:
    -   https://gevtushenko.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html

*/
void radixSortInitialiser(int* mortonCodesIn, int* mortonCodesOut, int* originalIndexIn, int* originalIndexOut, int NItems) {
    // Temporary storage requirements
    void* tempStorage = nullptr;
    size_t tempStorageBytes = 0;
    cub::DeviceRadixSort::SortPairs(tempStorage,tempStorageBytes,mortonCodesIn,mortonCodesOut,originalIndexIn,originalIndexOut,NItems);

    cudaMalloc(&tempStorage, tempStorageBytes);
}

void radixSort(int* mortonCodesIn, int* mortonCodesOut, int* originalIndexIn, int* originalIndexOut, int NItems) {
    //cub::DeviceRadixSort::SortPairs()
}



// =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
// Bounding box calculation code
// =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+


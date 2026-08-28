#include "sorting.h"
#include <cub/cub.cuh>

/*
A function to sort the array based on the value of the digit at index equal to exp

i.e. for the number (800), the [0]th value is 8, the [1]th value is 0 etc

*/

// =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
// Radix sort initialisation
// =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
    
radixSorter* sorterCreate(int maxCount) {
    if (maxCount <= 0) {
        printf("ERROR: Attempted to create a radix sort over an array of <= 0 elements.\n");
        fflush(stdout);
        exit(-1);
    }

    // Memory allocate the sorter
    radixSorter* sorter = (radixSorter*) calloc(1, sizeof(radixSorter));

    // Check if sorter was created properly
    if (sorter == NULL) {
        printf("ERROR: Radix sorter creation failed.\n");
        fflush(stdout);
        exit(-1);
    }

    // Gives CUB the type of the values being sorted, and the number of items being sorted, so that it can initialise the sorting algorithm
    uint64_t* nullKeys = nullptr;
    uint64_t* nullVals = nullptr;

    // Initialise the radix sorter, and check it for errors (I should probably do this more)

    cudaError_t radixError = cub::DeviceRadixSort::SortPairs(nullptr,sorter->tempStorageBytes,
            nullKeys,nullKeys,
            nullVals,nullVals,maxCount);

    if (radixError != 0) {
        printf("ERROR: Sorter CUB query failed: %s\n", cudaGetErrorString(radixError));
        exit(-1);
    }

    // Malloc data, and check for errors
    cudaError_t mallocErr = cudaMalloc(&sorter->tempStorage,sorter->tempStorageBytes);
    if (mallocErr != 0) {
        printf("ERROR: Failed to malloc data for CUB radix sort. %s\n", cudaGetErrorString(mallocErr));
        exit(-1);
    }

    sorter->capacity = maxCount;
    return sorter;
}

void radixSortPairs(radixSorter* sorter, const uint64_t* keysIn, uint64_t* keysOut,
                                         const uint32_t* valsIn, uint32_t* valsOut,
                                         int numCount) {

    if (sorter == NULL) {
        printf("ERROR: Sorter struct passed to radix sort was uninitialised");
        exit(-1);
    }
    if (numCount <= 0 || numCount > sorter->capacity) {
        printf("ERROR: Length of array to sort conflicts with capacity of radix sorter.\n");
        exit(-1);
    }
    cudaError_t sorterError = cub::DeviceRadixSort::SortPairs(sorter->tempStorage,sorter->tempStorageBytes,keysIn,keysOut,valsIn,valsOut,numCount);

    if(sorterError != 0) {
        printf("\nERROR: Error occured with performing radix sort: %s\n", cudaGetErrorString(sorterError));
        exit(-1);
    }
}

void sorterDestroyer(radixSorter* sorter) {
    if (sorter == NULL) {
        printf("ERROR: Attempted to delete non-existant sorter.\n");
        exit(-1);
    }
    cudaFree(sorter->tempStorage);
    // Makes the crash more obvious if it is called elsewhere
    sorter->tempStorage = NULL;
    sorter->tempStorageBytes = (size_t) 0;
    sorter->capacity = 0;
    free(sorter);
}

/*
Takes the positions and velocities of the input particles, and uses the sorted original incides to know
where to place them in the new position and velocity arrays.

*/
__global__ void reorderParticles(const float4* __restrict__ posMassValsIn, const float3* __restrict__ velValsIn, float4* posMassValsOut, float3* velValsOut, uint32_t* originalIndex, int starCount) {
    int threadNum = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadNum >= starCount) {
        return;
    }

    posMassValsOut[threadNum] = posMassValsIn[originalIndex[threadNum]];
    velValsOut[threadNum] = velValsIn[originalIndex[threadNum]];
}
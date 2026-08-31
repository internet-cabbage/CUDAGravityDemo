#include "tree.h"



// Built in radix sort 
#include <cub/cub.cuh>



/*
Morton code bit extraction.

So the morton codes store data interleaved in the order: x_20, y_20, z_20, x_19 ... x_0, y_0, z_0
To determine what octrant (my new name for octtree node) the star lies within, I have to extract a set of 3 bits from the encoded value.

The first 3 bits (x_20,y_20,z_20) determine what octrant of the root node the star lies within, and the second 3 bits determine the octrant of that etc...

To extract the bits we can just right shift the bits, as it discards bits to the right.

The 63 comes from the fact that the 64th bit doesn't store anything anyways
*/

__device__ uint64_t extractBitsFromLevel(uint64_t mortonCode, int extractLevel) {
    uint64_t extractedBits = (mortonCode >> (63 - (3*extractLevel)));
    return extractedBits;
}

/*
Morton insertion

If two stars lie in the same octrant of the root node, then they will share the first 3 bits of their morton code.
Likewise if any two stars lie in the same octrant of a node at depth N, they will share the first N*3 bits of their morton code.

Since particles are sorted according to morton code, all the child-particles of a node at level N share the first N digits

*/

/*
I 
*/

__global__ void identifyNodesAtLevel(uint64_t* mortonCodes, int nodeLevel, int starCount, uint32_t* nodeFlags) {
    int threadNum = threadIdx.x + (blockDim.x * blockIdx.x);

    if (threadNum >= starCount) {return;}
    if (threadNum == 0) {nodeFlags[0]=1;return;} // The first particle has to start a node

    uint64_t currentOctrant = extractBitsFromLevel(mortonCodes[threadNum],nodeLevel);
    uint64_t previousOctrant = extractBitsFromLevel(mortonCodes[threadNum-1],nodeLevel);

    if (currentOctrant != previousOctrant) {
        nodeFlags[threadNum] = 1;
    }
    else {
        nodeFlags[threadNum] = 0;
    }
}

// Code to actually produce the nodes
/*
This function takes the morton codes, node flags, node offsets, tree level, total star count, and maximum node count of the tree, and uses them to write to a single great array that spans all nodes.


Each node in the tree is all a part of one giant array. So in order to differentiate the different levels of the tree, we just add an offset to each nodeIndex that is equal to the nodeCount
of the tree at that level. We will call this offset 'arrayLevelOffset' as we already have a variable called offset.
*/

__global__ void updateTreeMass(node* nodes,const float4* positionMassVals, int level, int levelNodes, int levelStartIndex) {
    int threadNum = threadIdx.x + (blockDim.x * blockIdx.x); 
    int myNode = threadNum + levelStartIndex;
    if (threadNum >= levelNodes) {return;}
    
    /* Check if this node is a leaf node, if so then the total mass is just the sum of the mass of all contained stars etc, and same goes for the CoM
    just being the average position of the masses, weighted by mass.
    */
    bool isLeafNode = true;
    for (int i = 0; i < 8; i++) {
        if (nodes[myNode].child[i] != -1) {
            isLeafNode = false;
            break;
        }
    }
    int particleCount = nodes[myNode].particleCount;
    if (isLeafNode == true) {
        int starIndex = 0;
        // Iterates over all particles contained within the node
        float3 weightedPositionVals = {0.0,0.0,0.0};
        float totalMass = 0.0;

        for (int i = 0; i < particleCount; i++) {
            starIndex = nodes[myNode].firstParticleIndex + i;
            // Copy data values to the temporary array
            weightedPositionVals.x += positionMassVals[starIndex].x * positionMassVals[starIndex].w;
            weightedPositionVals.y += positionMassVals[starIndex].y * positionMassVals[starIndex].w;
            weightedPositionVals.z += positionMassVals[starIndex].z * positionMassVals[starIndex].w;

            totalMass += positionMassVals[starIndex].w;
        }
        // Calculates the cente of mass
        float4 massVals;
        if (totalMass > 0.0) {
            massVals.x = weightedPositionVals.x / totalMass;
            massVals.y = weightedPositionVals.y / totalMass;
            massVals.z = weightedPositionVals.z / totalMass;
            massVals.w = totalMass; // Total mass is just the sum of all masses

            // Now gives the node the mass value
            nodes[myNode].massData = massVals;
        }
        else {
            printf("ERROR: Total mass of node %d is zero.\n", threadNum);
            nodes[myNode].massData = {0.0,0.0,0.0,0.0};
        }

    }

    // Else, the node is not a leaf node, so we iterate over all its child nodes instead
    else {
        int childNodeCount = 0;
        float4 nodeMassVals = {0.0,0.0,0.0,0.0};

        // Node data array
        int nodeIndexes[8];
        for (int i = 0; i < 8; i++) {
            if ((nodes[myNode]).child[i] != -1) {
                // Stores the index of the child node into the nodeIndexes arra
                nodeIndexes[childNodeCount] = (nodes[myNode]).child[i];
                childNodeCount++;
                // Add total mass
            }
        }
        // Calculate total mass of node
        for (int i = 0; i < childNodeCount; i++) {
            // Updates the total mass by the mass of the child nodes
            nodeMassVals.w += (nodes[nodeIndexes[i]]).massData.w;
        }
        // Calculates the centre of mass of the node
        for (int i = 0; i < childNodeCount; i++) {
            float4 nodeMass = (nodes[nodeIndexes[i]]).massData;
            nodeMassVals.x += (nodeMass.x * nodeMass.w);
            nodeMassVals.y += (nodeMass.y * nodeMass.w);
            nodeMassVals.z += (nodeMass.z * nodeMass.w);
        }
        if (nodeMassVals.w > 0.0) {
            nodeMassVals.x /= nodeMassVals.w;
            nodeMassVals.y /= nodeMassVals.w;
            nodeMassVals.z /= nodeMassVals.w;

            nodes[myNode].massData = nodeMassVals;
        }
        else {
            printf("ERROR: Total mass of node %d is zero.\n", threadNum);
            nodes[myNode].massData = {0.0,0.0,0.0,0.0};
        }

    }

}

__global__ void updateTreeParticles(node* nodes, int levelStartIndex, int levelNodeCount, int starCount) {
    int threadNum = threadIdx.x + (blockDim.x * blockIdx.x);
    if (threadNum >= levelNodeCount) {return;}

    int firstParticle = nodes[levelStartIndex + threadNum].firstParticleIndex;
    int nextParticle = 0;

    // Check if the 'firstParticle' is the last node in the array
    if (threadNum == levelNodeCount - 1) {
        nextParticle = starCount; // The last node contains all particles between the start of this node and the end of the array
    }
    // If 'firstParticle' is not the last node in the array, we find the index of the next node
    else {
        // The index position of the first particle in the next node
        nextParticle = nodes[levelStartIndex + threadNum + 1].firstParticleIndex;
        // The total particles in the node is just the difference between this and the first particle index of the node being checked
    }

    nodes[levelStartIndex + threadNum].particleCount = (nextParticle - firstParticle);
}

__global__ void createNodes(const uint64_t* mortonCodes, const uint32_t* flags,const uint32_t* previousFlags, const uint32_t* offsets, const uint32_t* offsetsPrev, node* nodes, int level, int arrayLevelOffset, int prevArrayLevelOffset, int starCount, int maxNodes) {
    int threadNum = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadNum >= starCount) {return;} // Check if the thread corresponds to a star
    if (flags[threadNum]==0) {return;} // Not the start of a node

    // Retrieve the star's tree path, so it can be added to the node it is inputted into
    uint64_t pathFromRoot = extractBitsFromLevel(mortonCodes[threadNum],level); 
    
    int nodeIndex = offsets[threadNum] + arrayLevelOffset; // The nodeIndex is basically just a value which identifies each node at a given level of the tree
    if (nodeIndex >= maxNodes) {return;}

    // Writes all the node's data to the node array
    nodes[nodeIndex].nodePathFromRoot = pathFromRoot;
    nodes[nodeIndex].treeLevel = level;
    nodes[nodeIndex].firstParticleIndex = threadNum;
    for (int kid = 0; kid < 8; kid++) {nodes[nodeIndex].child[kid] = -1;} // Initialise all the nodes kids to -1 to signify they are currently empty

    // Add info to parent node
    int parent; // The index location of the parent node
    if (level == 1) { // The node is the level before the root, so its parent is just the root node
        parent = 0;
    }
    else {
        // The offsets value is equal to the parent node of the node at that index, as long as the flag value is 1. Otherwise it is 1 too high
        // This corrects that, so that the node can know what it's parent is.
        parent = prevArrayLevelOffset + offsetsPrev[threadNum] + previousFlags[threadNum] - 1;
    }
    int childIndexNumber = pathFromRoot & 7; // The child index number is the bottom 3 bits of the morton coded path, so we just chop off every bit which isnt them.
    nodes[parent].child[childIndexNumber] = nodeIndex;
    nodes[nodeIndex].parentIndex = parent;
}  

// =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
// Prefix sum wrappers 
// =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+

treeBuilder* sumCreate(int maxCount) {
    if (maxCount <= 0) {
        printf("ERROR: Attempted to initialise a prefix sum over an array of <= 0 elements.\n");
        fflush(stdout);
        exit(-1);
    }
    // Allocate memory for the prefix sum
    treeBuilder* sum = (treeBuilder*) calloc(1, sizeof(treeBuilder));

    // Check if the memory was allocated correctly etc
    if (sum==NULL) {printf("ERROR: Prefix sum could not be initialised.\n");exit(-1);}
    
    uint32_t* nullArray = nullptr; // To initialise the summation, it just needs to know the type

    cudaError_t sumError = cub::DeviceScan::ExclusiveSum(nullptr,sum->tempStorageBytes,nullArray,nullArray,maxCount);

    if (sumError != 0) {printf("ERROR: Prefix sum initialisation failed: %s\n",cudaGetErrorString(sumError));exit(-1);}

    // Allocate memory for prefix sum
    cudaError_t mallocErr = cudaMalloc(&sum->tempStorage,sum->tempStorageBytes);
    
    if (mallocErr != 0) {printf("ERROR: Failed to malloc data for the exclusive sum");exit(-1);}
    
    sum->capacity = maxCount;

    // =======================
    // Now I allocate the rest of the arrays for the treeBuilder:
    // flags,offsetsPing,offsetsPong,

    uint32_t* flagsPing; uint32_t* flagsPong; uint32_t* offsetsPing; uint32_t* offsetsPong;
    size_t arraySize = sizeof(uint32_t) * maxCount;

    cudaMalloc(&flagsPing,arraySize);
    cudaMalloc(&flagsPong,arraySize);
    cudaMalloc(&offsetsPing,arraySize);
    cudaMalloc(&offsetsPong,arraySize);

    sum->flagsPing = flagsPing;
    sum->flagsPong = flagsPong;
    sum->offsetsPing = offsetsPing;
    sum->offsetsPong = offsetsPong;

    return sum;
}   

void prefixSum(treeBuilder* builder, uint32_t* flags, uint32_t* offsets, int maxCount) {
    if (builder == NULL) {
        printf("ERROR: treeBuilder struct passed to prefixSum() was uninitialised.\n");
        exit(-1);
    }

    if (maxCount <= 0 || maxCount > builder->capacity) {
        printf("ERROR: Length of array to sum conflicts with capacity of treeBuilder struct.\n");
        exit(-1);
    }

    cudaError_t sumError = cub::DeviceScan::ExclusiveSum(builder->tempStorage,builder->tempStorageBytes,flags,offsets,maxCount);
    if (sumError != 0) {
        printf("ERROR: Error occured while performing prefix sum: %s\n",cudaGetErrorString(sumError));
        exit(-1);
    }
}

void sumDestroyer(treeBuilder* sum) {
    if (sum == NULL) {printf("ERROR: Attempted to delete non-existant prefix sum object.\n"); exit(-1);}

    // Free cuda arrays
    cudaFree(sum->flagsPing);
    cudaFree(sum->flagsPong);
    cudaFree(sum->offsetsPing);
    cudaFree(sum->offsetsPong);
    
    // Dereference pointers
    sum->capacity = 0;
    sum->tempStorage = NULL;
    sum->tempStorageBytes = (size_t) 0;
    free(sum);
}

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
__device__ uint64_t mortonIndex(uint32_t x, uint32_t y, uint32_t z) {
    return (expandBits(x)<<2) | (expandBits(y)<<1) | (expandBits(z));
}

__global__ void mortonEncode(const float4* posMassVals, uint64_t* mortonCodes, uint32_t* originalIndex, int NStars, worldBox rootBox) {
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




// =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
// Bounding box calculation code
// =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+

/*
Call __syncThreads() before running this function
*/
__device__ void blockReduceMinMax(float3* threadMin, float3* threadMax, int interiorID) {
    for (int i = BOXTHREADS / 2; i > 0; i /= 2) {
        if (interiorID < i) {
            // This makes it so that on the first run threads 0 to 127 compare all values between 0 and 256, then condense them all into 128 values at the bottom of the array.
            // This halving is repeated until only 1 thread exists.
            threadMin[interiorID].x = fminf(threadMin[interiorID].x, threadMin[interiorID + i].x);
            threadMin[interiorID].y = fminf(threadMin[interiorID].y, threadMin[interiorID + i].y);
            threadMin[interiorID].z = fminf(threadMin[interiorID].z, threadMin[interiorID + i].z);

            threadMax[interiorID].x = fmaxf(threadMax[interiorID].x, threadMax[interiorID + i].x);
            threadMax[interiorID].y = fmaxf(threadMax[interiorID].y, threadMax[interiorID + i].y);
            threadMax[interiorID].z = fmaxf(threadMax[interiorID].z, threadMax[interiorID + i].z);
        }
        // Important addition to stop the code from racing
        __syncthreads();
    }
}

/*
This was a source I used in order to ensure my algorithm was quick:

https://medium.com/@rimikadhara/7-step-optimization-of-parallel-reduction-with-cuda-33a3b2feafd8

Parameters:
    -*posMassVals: An array of float4's containing the position info and mass info of all stars, stored as floats
    -N: The number of stars which the reduction algorithm is acting over
    -*minCorner: An array of float3's containing the local minimum values from each thread 
    -*maxCorner: An array of float3's containing the local maximum values from each thread 
*/
__global__ void localMinMaxFinder(const float4* __restrict__ posMassVals, unsigned int N, float3* minCorner, float3* maxCorner) {
    // Shared memory to store all the values:
    __shared__ float3 threadMin[BOXTHREADS];
    __shared__ float3 threadMax[BOXTHREADS];

    int interiorID = threadIdx.x;
    unsigned int exteriorID = blockIdx.x * blockDim.x + threadIdx.x; // The global thread index
    unsigned int threadCount = gridDim.x * blockDim.x;

    float3 lowerVals = make_float3(INFINITY,INFINITY,INFINITY);
    float3 upperVals = make_float3(-INFINITY,-INFINITY,-INFINITY);

    // Now perform the reduction
    for (unsigned int i = exteriorID; i < N; i+=threadCount) {
        float4 localPos = posMassVals[i];
        lowerVals.x = fminf(lowerVals.x,localPos.x); upperVals.x = fmaxf(upperVals.x,localPos.x);
        lowerVals.y = fminf(lowerVals.y,localPos.y); upperVals.y = fmaxf(upperVals.y,localPos.y);
        lowerVals.z = fminf(lowerVals.z,localPos.z); upperVals.z = fmaxf(upperVals.z,localPos.z);
    }

    // Write the local maxima and minima to the shared memory

    threadMin[interiorID] = lowerVals;
    threadMax[interiorID] = upperVals;

    // Make sure threadMin and threadMax have been filled before reducing them
    __syncthreads();

    // Then condense the threadMin and threadMax values down into global minima and maxima values
    blockReduceMinMax(threadMin,threadMax,interiorID);

    // Now since at this point all the values assigned to the block have been condensed (Or in other words... reduced...), we can just write the local minima and maxima to the block minima and maxima arrays
    if (interiorID == 0) {
        minCorner[blockIdx.x] = threadMin[0];
        maxCorner[blockIdx.x] = threadMax[0];
    }
    
} 


__global__ void globalMinMaxReducer(const float3* __restrict__ minCorner, const float3* __restrict__ maxCorner, int count, float3* outMin, float3* outMax) {
    // Shared memory to store all the values:
    __shared__ float3 threadMin[BOXTHREADS];
    __shared__ float3 threadMax[BOXTHREADS];

    int interiorID = threadIdx.x;

    float3 lowerVals = make_float3(INFINITY,INFINITY,INFINITY);
    float3 upperVals = make_float3(-INFINITY,-INFINITY,-INFINITY);

    // Check if the thread even corresponds to an id in the 
    if (interiorID < count) {
        lowerVals = minCorner[interiorID];
        upperVals = maxCorner[interiorID];
    }
    threadMin[interiorID] = lowerVals;
    threadMax[interiorID] = upperVals;

    // Make sure threadMin and threadMax have been filled before reducing them
    __syncthreads();

    blockReduceMinMax(threadMin,threadMax,interiorID);

    // Now since at this point all the values assigned to the block have been condensed (Or in other words... reduced...), we can just write the local minima and maxima to the block minima and maxima arrays
    if (interiorID == 0) {
        *outMin = threadMin[0];
        *outMax = threadMax[0];
    }
}



// =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
// Tree traversal code
// =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+

__global__ void calculateForce(node* nodes) {
    return;
}
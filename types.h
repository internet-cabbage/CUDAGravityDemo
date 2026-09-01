#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>
#include <cuda_runtime.h>

typedef struct __attribute__((aligned(16))) {
    float x,y,z,m;
} sVec4;

typedef struct {
    float x,y,z;
} sVec3;

typedef struct {
    uint8_t R,G,B;
} RGB;

typedef struct {
    void* tempStorage;
    size_t tempStorageBytes;
    int capacity;
} radixSorter;

typedef struct {
    // The coordinates of the corner with the minimum values
    float minX, minY, minZ;

    // The extent is the largest of the width/height/depth values of the cube.
    // The scale is then multiplied by 2^21, which is equal to a multiplication by 1 << 21.
    float scale;
} worldBox;

/*
This struct contains all the data used to describe and handle nodes in the program.

To understand this it is best to think about what the morton indices actually encode. A 3D morton code basically already describes an octtree, with the
first 3 bits determining which octrant the particle falls within with respect to the root node. And every successive 3 bits determine its position inside of that node (if it exists).

For the sake of simplicity we will define each 3 bits sequentially from the left, as being a distinct 'level' in the octtree, so for example the particle's octrant wrt the root node is its level 1 position etc.
Since we have 3 interleaved integers stored in a 64 bit number, we will only have 21 different levels.

fields:
    -nodePathFromRoot; The path that must be taken from the root node to reach this node
    -firstParticleIndex; The index in the main particle array at which point the first particle in the node is located
    -particleCount; The number of particles in the node

*/

typedef struct {
    uint64_t nodePathFromRoot;
    int parentIndex;
    int firstParticleIndex; // The index in the sorted morton code array, at which point the particle(s) within this node start
    int particleCount; // How many particles are within the node
    
    int child[8]; // The index positions of the child nodes
    // If the child[i] value is equal to -1, then the child is empty

    int treeLevel;
    float4 massData; // The centre of mass of the node (first 3 elements), as well as the total mass (4th) element
} node;

typedef struct {
    void* tempStorage;
    size_t tempStorageBytes;
    int capacity;

    uint32_t* flagsPing;
    uint32_t* flagsPong;
    uint32_t* offsetsPing;
    uint32_t* offsetsPong;
} treeBuilder;

typedef struct {
    node* nodes;
    int maxNodes;
    uint64_t* mortonIn; uint64_t* mortonOut;
    uint32_t* indexIn; uint32_t* indexOut;
    float3* boundingData[4]; // An array of pointers to the bounding data stored on the GPU
    radixSorter* sorter;
    treeBuilder* builder;
    int levelOffsets[22];
    int levelCount[22];
    int deepestLevel;
    int treeBlocks; // Blocks to run on the GPU
    int starCount;
    float4* sortedPosMassVals;
    float3* sortedVelVals;
} treeState;




#endif
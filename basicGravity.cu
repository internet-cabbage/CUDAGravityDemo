#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>

// Include C libraries
extern "C" {
    #include "types.h"
    #include "inputOutput.h"
    #include "initialConditions.h"
}

#include "tree.h"
#include "sorting.h"
#include "units.h"
#include "macros.h"


#define THREADCOARSENING 2
#define THREADPERBLOCK 256

void createWorldBox(const float4* cudaPositionMassVals, const int starCount, 
                         float3* cudaMinCorner, float3* cudaMaxCorner,
                         float3* cudaOutputMin, float3* cudaOutputMax,
                         worldBox* outputWorldBox) {
    float3 minCorner, maxCorner;
    localMinMaxFinder<<<BOXBLOCKS,BOXTHREADS>>>(cudaPositionMassVals,starCount,cudaMinCorner,cudaMaxCorner);
    globalMinMaxReducer<<<1,BOXTHREADS>>>(cudaMinCorner,cudaMaxCorner,BOXBLOCKS,cudaOutputMin,cudaOutputMax);
    cudaMemcpy(&minCorner,cudaOutputMin,sizeof(float3),cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxCorner,cudaOutputMax,sizeof(float3),cudaMemcpyDeviceToHost);

    float width = maxCorner.x - minCorner.x;
    float height = maxCorner.y - minCorner.y;
    float depth = maxCorner.z - minCorner.z;
    float extent = fmaxf(width, fmaxf(height,depth));

    /*
    The binary representation of the integer 1 is: 0000...0001
    Since I only care about the first 21 bits of the integer, if I left shift it by 21 bits, I get: 1000...0000 (decimal value: 2,097,152)
    
    If I subtract 1 from it, it ends up getting the binary representation: 0111...1111, which is the largest value representable in 21 bits
    */
    int biggest21BitNum = (1u<<21)-1;
    // Scale is a way of mapping between the float coordinates, and an integer grid of range [0,2^21]
    float scale = (float)(biggest21BitNum) / extent;

    // Now we actually assign these values to the worldBox

    outputWorldBox->minX = minCorner.x;
    outputWorldBox->minY = minCorner.y;
    outputWorldBox->minZ = minCorner.z;
    outputWorldBox->scale = scale;
}

// Helper function
sVec3* randomGen(int lower, int upper, size_t N) {
    // Adress at which the array is saved at
    sVec3* address = (sVec3*) calloc(N, sizeof(sVec3));

    int intN = (int) N;
    for (int i = 0; i < intN; i++) {
        float xVal = (rand() % (upper-lower+1)) + lower;
        float yVal = (rand() % (upper-lower+1)) + lower;
        float zVal = (rand() % (upper-lower+1)) + lower;

        (address + i)->x = xVal;
        (address + i)->y = yVal;
        (address + i)->z = zVal;
    }
    return address;
}


/*
Each thread holds ownership of the stars at index: (threadTaskMapping, threadTaskMapping + blockDim.x, threadTaskMapping + 2 * blockDim.x, threadTaskMapping + 3 * blockDim.x)
This means that block 0 now holds ownership of the same stars that blocks: 0,1,2,3 would have held (using a thread coarsening factor of 4)
And likewise block 1 holds ownership of what used to be blocks: 4,5,6,7 etc
*/
__global__ void forceCalc(const float4* __restrict__ positionMassArray, float3 (*accelArray), int starCount, int paddedStarCount, float antiSingularitySquared) {
    // In order to give each thread several tasks, I assign each thread ownership of 4 tasks using the following formula
    int threadTaskMapping = blockIdx.x * blockDim.x * THREADCOARSENING + threadIdx.x;

    if (threadTaskMapping + ((THREADCOARSENING - 1) * blockDim.x) >= paddedStarCount) {
        //int threadNum = blockIdx.x * blockDim.x + threadIdx.x;
        //printf("Thread number: %d, was caught.\n", threadNum);
        return;
    }

    // Acceleration and position data
    float4 starData[THREADCOARSENING];
    float3 accels[THREADCOARSENING] = {};

    #pragma unroll
    for (int i = 0; i < THREADCOARSENING; i++) {
        starData[i] = positionMassArray[threadTaskMapping + (blockDim.x * i)];
    }

    // Brute force calculate the force exerted on the particle 'threadNum'
    // Position differences
    for (int i = 0; i < starCount; i++){
        /* Turns out this isnt required, as if the particle is compared to itself, its separatino is zero anyways
        if (i == threadNum) {continue;}
        */
        float4 ithStarData = positionMassArray[i];
        #pragma unroll
        for (int j = 0; j < THREADCOARSENING; j++) {
            float dx = starData[j].x - ithStarData.x;
            float dy = starData[j].y - ithStarData.y;
            float dz = starData[j].z - ithStarData.z;
            float r2 = (dx*dx + dy*dy + dz*dz + antiSingularitySquared);

            float invR = rsqrtf(r2);

            float aMag = (G*ithStarData.w) * invR * invR;

            accels[j].x -= aMag * (dx*invR);
            accels[j].y -= aMag * (dy*invR);
            accels[j].z -= aMag * (dz*invR);
        }

        // Acceleration components

    }
    #pragma unroll
    for (int k = 0; k < THREADCOARSENING; k++) {
        accelArray[threadTaskMapping + (k * blockDim.x)] = accels[k];
    }
}




// Symplectic euler, update velocity then position
__global__ void integrateStep( float4 (*positionArray),float3 (*velocityArray),float3 (*accelArray),int starCount, float dt){
    int threadNum = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadNum >= starCount) {
        return;
        // This means an out of bounds thread tried to run
    }
    
    /*
    // Output star 1.s parameters
    if (threadNum == 1) {
        printf("\nAcceleration: (%f,%f,%f)\n",accelArray[0].x,accelArray[0].y,accelArray[0].z);
        printf("Velocity: (%f,%f,%f\n",velocityArray[0].x,velocityArray[0].y,velocityArray[0].z);
        printf("Position: (%f,%f,%f\n\n",positionArray[0].x,positionArray[0].y,positionArray[0].z);
    }
    */
    
    velocityArray[threadNum].x += accelArray[threadNum].x * dt;
    positionArray[threadNum].x += velocityArray[threadNum].x * dt;

    velocityArray[threadNum].y += accelArray[threadNum].y * dt;
    positionArray[threadNum].y += velocityArray[threadNum].y * dt;

    velocityArray[threadNum].z += accelArray[threadNum].z * dt;
    positionArray[threadNum].z += velocityArray[threadNum].z * dt;

}

__global__ void threadPrinter(int* nThreads) {
    int threadNum = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread number: %d\n", threadNum);
}

int roundNumberToMultiple(int number, int multiple) {
    if (multiple == 0) {
        return number;
    }
    if (multiple < 0) {
        printf("Rounding up to a negative number is undefined (i.e. you probably made an error by even requesting this)\n");
        exit(-1);
    }

    int remainder = number % multiple;
    if (remainder == 0){ // If so then the number is already a multiple
        return number;
    }
    return number + multiple - remainder;

}

void threadSizeCalc(int* forceBlocks, int* integrateBlocks, int* paddedStarCount, int starCount) {
    int threadsPerForceBlock = THREADCOARSENING * THREADPERBLOCK;
    int threadsPerIntegrationBlock = THREADPERBLOCK;

    int forceBlockNum = -1;
    int integrateBlockNum = -1;

    *paddedStarCount = roundNumberToMultiple(starCount,threadsPerForceBlock);

    // Force block calculator
    if (starCount % threadsPerForceBlock == 0) {
        forceBlockNum = *paddedStarCount/threadsPerForceBlock;
    }
    else {
        forceBlockNum = (*paddedStarCount/threadsPerForceBlock) + 1;
    }

    // Integration block calculator
    if (starCount % threadsPerIntegrationBlock == 0) {
        integrateBlockNum = starCount/threadsPerIntegrationBlock;
    }
    else {
        integrateBlockNum = (starCount/threadsPerIntegrationBlock) + 1;
    }
    *forceBlocks = forceBlockNum;
    *integrateBlocks = integrateBlockNum;
    *paddedStarCount = roundNumberToMultiple(starCount,threadsPerForceBlock);
}

int main(void) {
    srand(time(0));
    //int cpuNThreads = 256;
    int tSteps = 10000;
    int starCount = 50000;
    int framesPerWrite = 5;

    // Size of star box

    /*
    Calculate the number of blocks to allocate to the calculations. 
    Since force calculation is just a memory read, a memory write and a bunch of repeated calculations. We can save processor time by performing
    several memory reads at the same time, then performing all the calculations. This means that less time is spent waiting for the data to transmit from memory, and more
    time is spent calculating!!!

    On the other hand the integration is mainly limited by the memory access time, so that is unecessary. So therefore we use different block counts for both in order to support
    the thread coarsening I implemented.
    */
    int forceBlocks, integrateBlocks, paddedStarCount;

    threadSizeCalc(&forceBlocks,&integrateBlocks,&paddedStarCount,starCount);

    printf("Blocks allocated for force calculation: %d/80\n",forceBlocks);
    printf("Blocks allocated for integration calculation: %d/80\n",integrateBlocks);
    printf("Total number of threads for force calculation: %d\n", starCount/THREADCOARSENING);
    printf("Threads per block: %d\n", THREADPERBLOCK);
    printf("Star count: %d, Padded star count: %d.\n",starCount, paddedStarCount);
    
    unsigned long int NSquared = ((unsigned long int)starCount*(unsigned long int)starCount);
    printf("\nInteractions total: %lu\n\n", (long int)tSteps * NSquared);

    // Put parameters and constants in array;
    float dt = 0.04;
    float antiSingularity = 0.2; // Can never be zero, otherwise an optimisation assumption breaks and the code will calculate NaN for acceleration

    // =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
    // Define arrays on CPU
    // =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+



    // Creates other arrays
    float3* cpuVelocityVals = (float3*)calloc(paddedStarCount, sizeof(float3));
    sVec3* cpuAccelerationVals = (sVec3*)calloc(paddedStarCount, sizeof(sVec3));

    float* cpuMassVals = (float*)calloc(paddedStarCount,sizeof(float));
    float totalMass = 5.0f;


    // Initialise arrays
    for (int i = 0; i < (int) starCount; i++) {
        cpuMassVals[i] = totalMass/starCount;

        cpuVelocityVals[i].x = 0.0f;
        cpuVelocityVals[i].y = 0.0f;
        cpuVelocityVals[i].z = 0.0f;

        cpuAccelerationVals[i] = (sVec3) {0.0,0.0,0.0};
    }

    // Padd out the empty arrays with zeros
    for (int i = starCount; i < paddedStarCount; i++) {
        cpuMassVals[i] = 0.0;

        cpuVelocityVals[i].x = 0.0f;
        cpuVelocityVals[i].y = 0.0f;
        cpuVelocityVals[i].z = 0.0f;
        
        cpuAccelerationVals[i] = (sVec3) {0.0,0.0,0.0};
    }


    /*
     Since GPU's are optimised to load 16 bytes of memory at a go, I pack the mass with the position to hopefully take advantage of this
    */
    float4* cpuPositionMassVals = (float4*)calloc(paddedStarCount,sizeof(float4));
    // Generates position data
    //sVec3* cpuPositionVals = (sVec3*)calloc(cpuNThreadsVal, sizeof(sVec3));


    // Initialise galaxy conditions
    //void generateLocalCoordsDisk(galaxyDiskParameters* parameters,sVec4* posMassVals, sVec3* velVals, int spawnNumber)
    galaxyDiskParameters parameters;
    parameters.galacticCentreLocation = (sVec3) {0.0,0.0,0.0};
    parameters.galacticNetVelocity = (sVec3) {0.0,0.0,0.0};
    parameters.rotationAngles = (sVec3) {0.0,0.0,0.0};
    parameters.scaleLength = (float) 4.0f;
    parameters.scaleThickness = (float) 0.3f;
    parameters.totalDiskMass = (float) totalMass;

    generateLocalCoordsDisk(&parameters,cpuMassVals,cpuPositionMassVals,cpuVelocityVals,starCount);

    free(cpuMassVals);


    // =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
    // Define equivalent arrays on GPU
    // =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+

    float4* cudaPositionMassVals = 0;
    float4* cudaPositionMassValsSorted = 0;

    float3* cudaVelocityVals = 0;
    float3* cudaVelocityValsSorted = 0;

    float3* cudaAccelerationVals = 0;

    // Allocate memory on GPU
    size_t posSize = sizeof(cpuPositionMassVals[0]) * paddedStarCount;
    size_t unpaddedPosSize = sizeof(cpuPositionMassVals[0]) * starCount;
    size_t velSize = sizeof(cpuVelocityVals[0]) * paddedStarCount;
    size_t accelSize = sizeof(cpuAccelerationVals[0]) * paddedStarCount;

    printf("Memory allocated to following arrays (in bytes):\n\n\tPosition: %zu\n\tVelocity: %zu\n\tAcceleration: %zu\n",posSize,velSize,accelSize);
    fflush(stdout);

    // Allocate arrays
    /*
    Two position and velocity arrays are needed, as after the particles have been sorted in morton order, we need to rearrange the position and velocity arrays.
    In order to do this we need a separate array, as an in-place reordering would have many race conditions and would be a mess.

    The acceleration array does not need an equivalent duplicate, as it is overwritten each frame.
    */
    cudaMalloc(&cudaPositionMassVals,posSize);
    cudaMalloc(&cudaPositionMassValsSorted,posSize);

    cudaMalloc(&cudaVelocityVals,velSize);
    cudaMalloc(&cudaVelocityValsSorted,velSize);

    cudaMalloc(&cudaAccelerationVals,accelSize);

    // Error checking
    if (cudaPositionMassVals == NULL || cudaVelocityVals == NULL || cudaAccelerationVals == NULL) {
        printf("Attempt to allocate memory on the GPU failed.\n");
        printf("\n\tPosition: %p\n\tVelocity: %p\n\tAcceleration: %p\n",cudaPositionMassVals,cudaVelocityVals,cudaAccelerationVals);
        fflush(stdout);
        exit(-1);
    } 

    printf("Memory transferred to GPU\n");

    // Transfer arrays
    cudaMemcpy(cudaPositionMassVals,cpuPositionMassVals,posSize,cudaMemcpyHostToDevice);
    cudaMemcpy(cudaVelocityVals,cpuVelocityVals,velSize,cudaMemcpyHostToDevice);
    cudaMemcpy(cudaAccelerationVals,cpuAccelerationVals,accelSize,cudaMemcpyHostToDevice);

    // =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
    // Core logic
    // =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+

    // Run function
    //<<<Number of blocks, Number of threads per block>>>
    //threadPrinter<<<numBlocks,threadsPerBlock>>>(cudaNThreads);

    // File writing prequisites
    FILE *fptr;
    fptr = fopen("outputDump.bin","wb");
    if (fptr == NULL) {
        printf("File pointer null\n");
        fflush(stdout);
        exit(-1);
    }


    // Write rendering information to file
    fwrite(&starCount,sizeof(starCount),1,fptr);
    int writeSteps = (tSteps/framesPerWrite);
    fwrite(&writeSteps,sizeof(writeSteps),1,fptr);

    printf("Write frames: %d\n", writeSteps);

    // Write temporary colour vals to file
    RGB* colourVals = (RGB*)calloc(starCount, sizeof(RGB));
    for (int i = 0; i < (int) starCount; i++) {
        colourVals[i] = {100,100,100};
    }
    fwrite(colourVals, sizeof(colourVals[0]),starCount,fptr);

    printf("Size of colour vals: %zu\n\n\n", sizeof(colourVals));

    float* frameBuffer = (float*) calloc(starCount,sizeof(sVec3));
    
    // Check if pointer is null
    if (frameBuffer == NULL) {
        printf("ERROR: frameBuffer pointer is Null\n");
        exit(-1);
    }
    for (int i = 0; i < starCount*3; i++) {
        frameBuffer[i] = 0.0;
    }

    // Output initial couple of values

    printf("\nNumber of stars: (%d)\n", (int) starCount);

    printf("Started to calculate force\n");

    // =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
    // Morton coding initialisation
    // =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+

    // For the radix sorter to work, it requires an array of the morton codes as input, and an array to write the sorted array to
    // The 'in' variables are basically just the input, and the 'out' variables are the sorted output
    uint64_t* cudaMortonCodesIn;
    uint64_t* cudaMortonCodesOut;
    uint32_t* cudaOriginalIndexIn; // The index / star id of the star whose morton code is being found
    uint32_t* cudaOriginalIndexOut;
    

    cudaMalloc(&cudaMortonCodesIn,sizeof(uint64_t)*starCount);
    cudaMalloc(&cudaMortonCodesOut,sizeof(uint64_t)*starCount);

    cudaMalloc(&cudaOriginalIndexIn,sizeof(uint32_t)*starCount);
    cudaMalloc(&cudaOriginalIndexOut,sizeof(uint32_t)*starCount);

    // Allocating memory for the bounding boxes
    float3* cudaMinCorner; float3* cudaMaxCorner; float3* cudaOutputMin; float3* cudaOutputMax;
    cudaMalloc(&cudaMinCorner,sizeof(float3)*BOXBLOCKS);
    cudaMalloc(&cudaMaxCorner,sizeof(float3)*BOXBLOCKS);
    cudaMalloc(&cudaOutputMin,sizeof(float3));
    cudaMalloc(&cudaOutputMax,sizeof(float3));

    // Define the rootbox
    worldBox rootBox;
    rootBox.minX = 0.0; rootBox.minY = 0.0; rootBox.minZ = 0.0;
    rootBox.scale = 0.0;
    createWorldBox(cudaPositionMassVals,starCount,cudaMinCorner,cudaMaxCorner,cudaOutputMin,cudaOutputMax,&rootBox);

    if (rootBox.scale == 0.0) {
        printf("ERROR: rootBox failed to initialise.\n\tminXCoord: %f\n\tminYCoord %f\n\tminZCoord: %f\n\tscale: %f\n",rootBox.minX,rootBox.minY,rootBox.minZ,rootBox.scale);
        fflush(stdout);
        exit(-1);
    }



    // Calculate the morton codes for the stars
    mortonEncode<<<integrateBlocks,THREADPERBLOCK>>>(cudaPositionMassVals,cudaMortonCodesIn,cudaOriginalIndexIn,starCount,rootBox);
    cudaDeviceSynchronize();

    localMinMaxFinder<<<BOXBLOCKS,BOXTHREADS>>>(cudaPositionMassVals,starCount,cudaMinCorner,cudaMaxCorner);
    globalMinMaxReducer<<<1,BOXTHREADS>>>(cudaMinCorner,cudaMaxCorner,BOXBLOCKS,cudaOutputMin,cudaOutputMax);

    cudaDeviceSynchronize();

    // Test sorter
    radixSorter* tempSorter = sorterCreate(starCount);
    radixSortPairs(tempSorter,cudaMortonCodesIn,cudaMortonCodesOut,cudaOriginalIndexIn,cudaOriginalIndexOut,starCount);

    // Extract values from GPU to CPU so I can print them
    float3 minCorner, maxCorner;

    cudaMemcpy(&minCorner,cudaOutputMin,sizeof(float3),cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxCorner,cudaOutputMax,sizeof(float3),cudaMemcpyDeviceToHost);

    printf("Minimum corner: (%f,%f,%f)\nMaximum corner: (%f,%f,%f)\n",minCorner.x,minCorner.y,minCorner.z, maxCorner.x,maxCorner.y,maxCorner.z);




    // Test of tree functions

    treeBuilder* prefixSumObject = sumCreate(starCount);

    identifyNodesAtLevel<<<BOXBLOCKS,BOXTHREADS>>>(cudaMortonCodesOut,8,starCount,prefixSumObject->flags);

    uint32_t* CPUNodeFlags = (uint32_t*) calloc(starCount,sizeof(uint32_t));
    uint64_t* CPUMortonCodes = (uint64_t*) calloc(starCount,sizeof(uint64_t));
    cudaMemcpy(CPUMortonCodes,cudaMortonCodesOut,sizeof(uint64_t)*starCount,cudaMemcpyDeviceToHost);

    cudaMemcpy(CPUNodeFlags,prefixSumObject->flags,sizeof(uint32_t)*starCount,cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("MORTON CODE [%d]: %lu\n", i, CPUMortonCodes[i]);
        fflush(stdout);
    }

    int flagCount = 0;
    for (int i = 0; i < starCount; i++) {
        flagCount += CPUNodeFlags[i];
    }
    printf("Flag count: %d\n\n", flagCount);

    int maxNodes = 8 * starCount;
    node* CPUNodes = (node*) calloc(maxNodes,sizeof(node));
    uint32_t* CPUFlags = (uint32_t*) calloc(starCount,sizeof(uint32_t));

    
    
    node* CudaNodes = 0;
    cudaMalloc(&CudaNodes,sizeof(node)*maxNodes);

    // Create root node;

    node rootNode;
    rootNode.nodePathFromRoot = (uint64_t) 0;
    rootNode.treeLevel = (int) 0;
    rootNode.firstParticleIndex = (int) 0;
    for (int c = 0; c < 8; c++) {
        rootNode.child[c] = -1;
    }
    // Send root node to GPU memory

    CUDA_CHECK(cudaMemcpy(CudaNodes, &rootNode,sizeof(node),cudaMemcpyHostToDevice));

    int arrayLevelOffset = 1;
    int previousArrayLevelOffset = 0;

    // How many nodes exist in total
    int totalNodes = 0;

    uint32_t* offsets = prefixSumObject->offsetsPing;
    uint32_t* previousOffsets = prefixSumObject->offsetsPong;

    for (int levels = 1; levels <= 21; levels++) {
        // Mark the index boundaries at which point the nodes start
        identifyNodesAtLevel<<<BOXBLOCKS,BOXTHREADS>>>(cudaMortonCodesOut,levels,starCount,prefixSumObject->flags);
        cudaDeviceSynchronize();

        // Use the index boundaries to tag each star to say which node it lies at
        prefixSum(prefixSumObject,prefixSumObject->flags,offsets,starCount);
        cudaDeviceSynchronize();

        // Use the node boundaries to produce the node array
        createNodes<<<BOXBLOCKS,BOXTHREADS>>>(cudaMortonCodesOut, (const uint32_t*) (prefixSumObject->flags), offsets, previousOffsets, CudaNodes, levels, arrayLevelOffset, previousArrayLevelOffset, starCount, maxNodes);
        cudaDeviceSynchronize();



        // Calculate how many nodes were spawned in the last createNodes call
        int nodeCount;



        // the prefixSum at index i tells you how many particles occured before index i, so the total nodes is just the last element of prefixSum plus the last element of flags.
        uint32_t lastFlagVal; uint32_t lastOffsetVal;
        CUDA_CHECK(cudaMemcpy(&lastFlagVal,&(prefixSumObject->flags[starCount-1]),sizeof(uint32_t),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&lastOffsetVal,&(offsets[starCount-1]),sizeof(uint32_t),cudaMemcpyDeviceToHost));
        nodeCount = lastFlagVal + lastOffsetVal;
        printf("Node count at level %d is: %d\n",levels, nodeCount);


        totalNodes += nodeCount;
        // Check if tree is complete, if so we can stop building it
        if (nodeCount == starCount) {break;}
        
        // Check for an overflow
        if (arrayLevelOffset + nodeCount > maxNodes) {
            printf("ERROR: Node array overflow at level: %d",levels);
            fflush(stdout);
            exit(-1);
        }
        

        // update array offsets
        previousArrayLevelOffset = arrayLevelOffset;
        arrayLevelOffset += nodeCount;

        uint32_t* tempVal =  previousOffsets;
        previousOffsets = offsets; // This makes the previousOffsets point to the offsets array (which is now the previous offsets array)
        offsets = tempVal; // This sets the offsets pointer to the now useless previousOffsets array, which will be overwritten
        
    }
    cudaDeviceSynchronize();

    printf("Starting to copy data from GPU to CPU\n");
    fflush(stdout);
    // Copy back the nodes array and perform a check to see all the pointers are done properly
    cudaMemcpy(CPUNodes,CudaNodes,sizeof(node)*maxNodes,cudaMemcpyDeviceToHost);

    printf("Copied memory from GPU to CPU\n");
    printf("Total nodes in tree: %d\n", totalNodes);
    fflush(stdout);
    int childrenThatExist = 0;
    for (int i = 0; i < totalNodes; i++) {
        for (int j = 0; j < 8; j++) {
            // If the child does not exist, break
            if (CPUNodes[i].child[j] != -1) {
                childrenThatExist -=-1;
            }
        }
    }
    printf("DRUMROLL, bmd bdm bdm bdm, there are %d many kids in the tree.\n",childrenThatExist);

    printf("\n\nIF THE TOTAL NODES DIFFERS FROM THE NUMBER OF KIDS, SOMETHING IS GOING WRONG!!!\n\n\n\n");

    exit(23);

    // CPU MIN MAX

    cudaEvent_t startTime, finishTime, mathStartTime, mathEndTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&finishTime);

    // Functions to profile the maths speed
    cudaEventCreate(&mathStartTime);
    cudaEventCreate(&mathEndTime);
    float timePerStep = 2.0;
    cudaEventRecord(mathStartTime,0);

    // Record the start time
    cudaEventRecord(startTime,0);
    

    progressPrinter(tSteps,0,40,0);

    /*
    Process:
        1) Start time profiling
        2) Produce the new worldBox bounding box
        3) Calculate the morton codes of the stars
        4) Radix sort the stars via their morton codes
        5) Use the sorted codes to produce the Barnes-Hutt tree in parallel
        6) Traverse the Barnes-Hutt tree to calculate the force acting on the particles
        7) Integrate the force steps to move the stars
        8) Check if the frame should be written to data
        9) Repeat
    */

    // Create radix sorter
    radixSorter* sorter = sorterCreate(starCount);

    for (int i = 0; i < tSteps; i++) {
        // PROCESS 1)
        cudaEventRecord(mathStartTime,0);

        // PROCESS 2)
        createWorldBox(cudaPositionMassVals,starCount,cudaMinCorner,cudaMaxCorner,cudaOutputMin,cudaOutputMax,&rootBox);
        
        // PROCESS 3)
        mortonEncode<<<integrateBlocks,THREADPERBLOCK>>>(cudaPositionMassVals,cudaMortonCodesIn,cudaOriginalIndexIn,starCount,rootBox);
        cudaDeviceSynchronize();

        // PROCESS 4)
        radixSortPairs(sorter,cudaMortonCodesIn,cudaMortonCodesOut,cudaOriginalIndexIn,cudaOriginalIndexOut,starCount);

        reorderParticles<<<integrateBlocks,THREADPERBLOCK>>>(cudaPositionMassVals,cudaVelocityVals,cudaPositionMassValsSorted,cudaVelocityValsSorted,cudaOriginalIndexOut,starCount);

        // Swap arrays
        float4* temp = cudaPositionMassVals;
        cudaPositionMassVals = cudaPositionMassValsSorted;
        cudaPositionMassValsSorted = temp;
        // Check if any morton codes are duplicated
        
        /*
        uint64_t* CPUMortonCodesTemp = (uint64_t*) calloc(starCount,sizeof(uint64_t));
        cudaMemcpy(CPUMortonCodesTemp,cudaMortonCodesOut,sizeof(uint64_t)*starCount,cudaMemcpyDeviceToHost);
        int duplications = 0;
        for (int j = 0; j < starCount-1; j++) {
            if (CPUMortonCodesTemp[j] == CPUMortonCodesTemp[j+1]) {
                duplications-=-1;
            }
        }
        printf("DUPLICATE MORTON CODES: %d\n", duplications);
        free(CPUMortonCodesTemp);
        */

        // PROCESS 5)

        // TODO

        // Pre morton sorting time 64.67 seconds

        // PROCESS 6)
        forceCalc<<<forceBlocks,THREADPERBLOCK>>>(cudaPositionMassValsSorted,cudaAccelerationVals,starCount,paddedStarCount,antiSingularity*antiSingularity);
        
        // TODO

        // PROCESS 7)


        
        //cudaDeviceSynchronize();
        integrateStep<<<integrateBlocks,THREADPERBLOCK>>>(cudaPositionMassValsSorted,cudaVelocityValsSorted,cudaAccelerationVals,starCount,dt);
        cudaEventRecord(mathEndTime,0);
        cudaDeviceSynchronize();
        if (i % framesPerWrite == 0) {        
            cudaError_t mathTimeErr = cudaEventElapsedTime(&timePerStep,mathStartTime,mathEndTime);
            if (mathTimeErr != 0) {printf("CUDA ERROR: Problem with frame time calculator, %s\n",cudaGetErrorString(mathTimeErr));}
            cudaMemcpy(cpuPositionMassVals,cudaPositionMassValsSorted,unpaddedPosSize,cudaMemcpyDeviceToHost);
            writeFrame(fptr,(sVec4*)cpuPositionMassVals,starCount,frameBuffer);
            progressPrinter(tSteps,i,40,timePerStep/((float)framesPerWrite));
        }
    }
    cudaEventRecord(finishTime,0);
    cudaDeviceSynchronize();
    float timeElapsedMilliseconds = -1.1;
    cudaError_t totalElapsedTimeErr =  cudaEventElapsedTime(&timeElapsedMilliseconds,startTime,finishTime);
    if (totalElapsedTimeErr != 0) {printf("CUDA ERROR: Problem with total elapsed time calculator, %s\n", cudaGetErrorString(totalElapsedTimeErr));}


    // World box location



    cudaDeviceSynchronize();



    progressPrinter(tSteps,tSteps,40,timePerStep);
    printf("\n");

    printf("\n\nTotal time elapsed (s): %f\n\n", timeElapsedMilliseconds/(1000.0f));

    printf("\nInteractions per second: %e\n", (double) tSteps*1000*(NSquared)/(timeElapsedMilliseconds));
    return 0;
}
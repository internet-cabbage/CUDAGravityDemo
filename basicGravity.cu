#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>

// Include C libraries
extern "C" {
    #include "types.h"
    #include "inputOutput.h"
}

#define cpuNThreadsVal 1048576
#define THREADCOARSENING 2

// Helper function
vec3* randomGen(int lower, int upper, size_t N) {
    // Adress at which the array is saved at
    vec3* address = (vec3*) calloc(N, sizeof(vec3));

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
__global__ void forceCalc(const float4* __restrict__ positionMassArray, vec3 (*accelArray), int nThreads, float G, float antiSingularitySquared) {
    // In order to give each thread several tasks, I assign each thread ownership of 4 tasks using the following formula
    int threadTaskMapping = blockIdx.x * blockDim.x * THREADCOARSENING + threadIdx.x;

    if (threadTaskMapping + ((THREADCOARSENING - 1) * blockDim.x) > nThreads-1) {
        int threadNum = blockIdx.x * blockDim.x + threadIdx.x;
        printf("Thread number: %d, was caught.\n", threadNum);
        return;
    }

    // Acceleration and position data
    float4 starData[THREADCOARSENING];
    vec3 accels[THREADCOARSENING] = {};

    #pragma unroll
    for (int i = 0; i < THREADCOARSENING; i++) {
        starData[i] = positionMassArray[threadTaskMapping + (blockDim.x * i)];
    }

    // Brute force calculate the force exerted on the particle 'threadNum'
    // Position differences
    for (int i = 0; i < nThreads; i++){
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
__global__ void integrateStep( float4 (*positionArray),vec3 (*velocityArray),vec3 (*accelArray), int nThreads, float dt){
    int threadNum = blockDim.x * blockIdx.x + threadIdx.x;
    
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


void progressPrinter(int tSteps, int currentStep, int width) {
    double percentageVal = (double) currentStep / (tSteps) * 100;

    char filledBar[] = {"||||||||||||||||||||||||||||||||||||||||"};
    char emptyBar[] = {"----------------------------------------"};

    // Width of the filled bar
    int filledWidth = (int) (percentageVal / 100 * width);
    int emptyPad = width - filledWidth;
    
    // Write emptry bar
    fprintf(stdout,"%.s", emptyBar);

    // Write filled bar
    fprintf(stdout,"\r %5.2f%% [%.*s%.*s] tStep: %5.d / %d", percentageVal, filledWidth, filledBar, emptyPad, emptyBar, currentStep, tSteps);
    fflush(stdout);
}

int main(void) {
    //int cpuNThreads = 256;
    int tSteps = 1000;
    int cpuNThreads = cpuNThreadsVal;
    int framesPerWrite = 20;

    size_t starCount = cpuNThreadsVal;

    int threadsPerBlock = 256;
    /*
    Calculate the number of blocks to allocate to the calculations. 
    Since force calculation is just a memory read, a memory write and a bunch of repeated calculations. We can save processor time by performing
    several memory reads at the same time, then performing all the calculations. This means that less time is spent waiting for the data to transmit from memory, and more
    time is spent calculating!!!

    On the other hand the integration is mainly limited by the memory access time, so that is unecessary. So therefore we use different block counts for both in order to support
    the thread coarsening I implemented.
    */
    int forceBlocks = cpuNThreads / (threadsPerBlock * THREADCOARSENING);
    int integrateBlocks = cpuNThreads / (threadsPerBlock);

    printf("Blocks allocated for force calculation: %d/80\n",forceBlocks);
    printf("Blocks allocated for integration calculation: %d/80\n",integrateBlocks);
    printf("Total number of threads for force calculation: %d\n", cpuNThreads/THREADCOARSENING);
    printf("Threads per block: %d\n", threadsPerBlock);
    
    unsigned long int NSquared = ((unsigned long int)cpuNThreadsVal*(unsigned long int)cpuNThreadsVal);
    printf("\nInteractions total: %lu\n\n", (long int)tSteps * NSquared);

    if (cpuNThreads % (threadsPerBlock * THREADCOARSENING) != 0) {
        printf("ERROR: Number of stars is not cleanly divisible into threads\n");
        printf("I have not implemented the fix for this yet");
        exit(-1);
    }

    // Put parameters and constants in array;
    float G = 50.0;
    float dt = 0.15;
    float antiSingularity = 4.0; // Can never be zero, otherwise an optimisation assumption breaks and the code will calculate NaN for acceleration

    // =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
    // Define arrays on CPU
    // =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+



    // Creates other arrays
    vec3* cpuVelocityVals = (vec3*)calloc(cpuNThreadsVal, sizeof(vec3));
    vec3* cpuAccelerationVals = (vec3*)calloc(cpuNThreadsVal, sizeof(vec3));

    float* cpuMassVals = (float*)calloc(cpuNThreads,sizeof(float));

    // Initialise arrays
    for (int i = 0; i < cpuNThreads; i++) {
        cpuMassVals[i] = 2.0;

        cpuVelocityVals[i] = (vec3) {0.0,0.0,0.0};
        cpuAccelerationVals[i] = (vec3) {0.0,0.0,0.0};
    }


    /*
     Since GPU's are optimised to load 16 bytes of memory at a go, I pack the mass with the position to hopefully take advantage of this
    */
    float4* cpuPositionMassVals = (float4*)calloc(cpuNThreadsVal,sizeof(float4));
    // Generates position data
    //vec3* cpuPositionVals = (vec3*)calloc(cpuNThreadsVal, sizeof(vec3));

    // Turn pointer of array into array to make code more consistent

    vec3* posPtr = randomGen(-4000,4000,cpuNThreads);
    for (int i = 0; i < cpuNThreads; i++) {
        cpuPositionMassVals[i] = {posPtr[i].x,posPtr[i].y,posPtr[i].z,cpuMassVals[i]};
    }
    free(posPtr);
    free(cpuMassVals);


    // =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
    // Define equivalent arrays on GPU
    // =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+

    float4* cudaPositionMassVals = 0;
    vec3* cudaVelocityVals = 0;
    vec3* cudaAccelerationVals = 0;

    // Allocate memory on GPU
    size_t posSize = sizeof(cpuPositionMassVals[0]) * cpuNThreadsVal;
    size_t velSize = sizeof(cpuVelocityVals[0]) * cpuNThreadsVal;
    size_t accelSize = sizeof(cpuAccelerationVals[0]) * cpuNThreadsVal;

    printf("Memory allocated to following arrays (in bytes):\n\n\tPosition: %zu\n\tVelocity: %zu\n\tAcceleration: %zu\n",posSize,velSize,accelSize);
    fflush(stdout);

    // Allocate arrays
    cudaMalloc(&cudaPositionMassVals,posSize);
    cudaMalloc(&cudaVelocityVals,velSize);
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

    for (int i = 0; i < cpuNThreads; i++) {
        //printf("Position of particle before running %d: (%f,%f,%f)\n",i,cpuPositionVals[i].x,cpuPositionVals[i].y,cpuPositionVals[i].z);
    }

    // File writing prequisites
    FILE *fptr;
    fptr = fopen("outputDump.bin","wb");
    if (fptr == NULL) {
        printf("File pointer null\n");
        fflush(stdout);
        exit(-1);
    }


    // Write rendering information to file
    fwrite(&cpuNThreads,sizeof(cpuNThreads),1,fptr);
    int writeSteps = (tSteps/framesPerWrite);
    fwrite(&writeSteps,sizeof(writeSteps),1,fptr);

    // Write temporary colour vals to file
    RGB* colourVals = (RGB*)calloc(cpuNThreadsVal, sizeof(RGB));
    for (int i = 0; i < cpuNThreads; i++) {
        colourVals[i] = {100,100,100};
    }
    fwrite(colourVals, sizeof(colourVals[0]),cpuNThreadsVal,fptr);

    printf("Size of colour vals: %zu\n\n\n", sizeof(colourVals));

    float* frameBuffer = (float*) calloc(starCount,sizeof(vec3));
    
    // Check if pointer is null
    if (frameBuffer == NULL) {
        printf("ERROR: frameBuffer pointer is Null\n");
        exit(-1);
    }
    for (int i = 0; i < (int)starCount*3; i++) {
        frameBuffer[i] = 0.0;
    }

    // Output initial couple of values

    printf("\n NThreadsVal: (%d)\n", cpuNThreadsVal);

    printf("Started to calculate force\n");

    // =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
    // CUDA performance profiling
    // =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
    
    cudaEvent_t startTime, finishTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&finishTime);

    // Record the start time
    cudaEventRecord(startTime,0);
    

    progressPrinter(tSteps,0,40);

    for (int i = 0; i < tSteps; i++) {
        forceCalc<<<forceBlocks,threadsPerBlock>>>(cudaPositionMassVals,cudaAccelerationVals,cpuNThreadsVal,G,antiSingularity*antiSingularity);
        //cudaDeviceSynchronize();
        integrateStep<<<integrateBlocks,threadsPerBlock>>>(cudaPositionMassVals,cudaVelocityVals,cudaAccelerationVals,cpuNThreadsVal,dt);
        //cudaDeviceSynchronize();

        if (i % framesPerWrite == 0) {
            cudaMemcpy(cpuPositionMassVals,cudaPositionMassVals,posSize,cudaMemcpyDeviceToHost);
            writeFrame(fptr,(vec4*)cpuPositionMassVals,starCount,frameBuffer);
            progressPrinter(tSteps,i,40);
        }
    }
    cudaDeviceSynchronize();
    cudaEventRecord(finishTime,0);
    float timeElapsedMilliseconds = 0;
    cudaEventElapsedTime(&timeElapsedMilliseconds,startTime,finishTime);



    progressPrinter(tSteps,tSteps,40);
    printf("\n");

    printf("\n\nTotal time elapsed: %f\n\n", timeElapsedMilliseconds/1000.0);

    printf("\nInteractions per second: %e\n", (double) tSteps*1000*(NSquared)/(timeElapsedMilliseconds));
    return 0;
}
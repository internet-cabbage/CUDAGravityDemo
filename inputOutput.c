#include "inputOutput.h"

// Cute little helper function to make it easier to write data to the binary file
void writeFrame(FILE* dataFile, sVec4* positionVals, size_t N, float *frameBuffer) {
    for (int i = 0; i < (int) N; i++) {
        frameBuffer[3*i] = (float)positionVals[i].x;
        frameBuffer[(3*i)+1] = (float)positionVals[i].y;
        frameBuffer[(3*i)+2] = (float)positionVals[i].z;
    }
    fwrite(frameBuffer, sizeof(float), 3*N, dataFile);
}

/* If value 1 < value 2, the return value is negative
if value 1 = value 2, return value = 0
if value 1 > value 2, return value > 0
*/
int keyValueSorter(const sorterPair* pair1, const sorterPair* pair2) {
    if (pair1->originalIndex < pair2->originalIndex) {
        return -1;
    }
    else if (pair1->originalIndex > pair2->originalIndex) {
        return +1;
    }
    else {
        return 0;
    }
}

/* The orderings of the star data is not a constant during runtime, as stars are organised such that adjacent stars are stored contiguously in the star array.
So for example the star which is originally at index 10, may move to index 40 in the next frame, and maybe 30 the next. 

In order to give each star a colour we would either have to store the colour value as an array which is permuted each time step (increase file size by a factor of 4/3), or just sort the data 
before we write it to the file, and have a constant 'header' to the file which stores the star colours. 

This function will sort the star data each frame such that it is constant across the run.
*/

void writeOrderedFrame(FILE* datafile, float4* positionVals, uint32_t* originalIndex, size_t N, float* frameBuffer, sorterPair* pairBuffer) {

    // Input data into key value pair datastructure
    for (int i = 0; i < N; i++) {
        pairBuffer[i].positionVal = &(positionVals[i]);
        pairBuffer[i].originalIndex = &(originalIndex[i]);
    }
    // Sort stars by index
    qsort(pairBuffer,sizeof(sorterPair),N,keyValueSorter);

    // Write into the framebuffer from the sorted pairs buffer
    for (int i = 0; i < (int) N; i++) {
        frameBuffer[3*i] = (float) (pairBuffer[i]).positionVal->x;
        frameBuffer[(3*i)+1] = (float) (pairBuffer[i]).positionVal->y;
        frameBuffer[(3*i)+2] = (float) (pairBuffer[i]).positionVal->z;
    }
    fwrite(frameBuffer,sizeof(float),3*N,datafile);
}

void progressPrinter(int tSteps, int currentStep, int width, float timePerStep) {
    double percentageVal = (double) currentStep / (tSteps) * 100;

    char filledBar[] = {"||||||||||||||||||||||||||||||||||||||||"};
    char emptyBar[] = {"----------------------------------------"};

    // Width of the filled bar
    int filledWidth = (int) (percentageVal / 100 * width);
    int emptyPad = width - filledWidth;
    
    // Write emptry bar
    fprintf(stdout,"%.s", emptyBar);

    // Write filled bar
    fprintf(stdout,"\r %5.2f%% [%.*s%.*s] tStep: %5.d / %d, stepTime (ms): %5.4f", percentageVal, filledWidth, filledBar, emptyPad, emptyBar, currentStep, tSteps, timePerStep);
    fflush(stdout);
}
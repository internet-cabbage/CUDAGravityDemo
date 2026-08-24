#include "inputOutput.h"

// Cute little helper function to make it easier to write data to the binary file
void writeFrame(FILE *dataFile, sVec4* positionVals, size_t N, float *frameBuffer) {
    for (int i = 0; i < (int) N; i++) {
        frameBuffer[3*i] = (float)positionVals[i].x;
        frameBuffer[(3*i)+1] = (float)positionVals[i].y;
        frameBuffer[(3*i)+2] = (float)positionVals[i].z;
    }
    fwrite(frameBuffer, sizeof(float), 3*N, dataFile);
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
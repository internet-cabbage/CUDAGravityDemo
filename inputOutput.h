#ifndef INPUTOUTPUT_H
#define INPUTOUTPUT_H

#include <stdio.h>
#include <stdlib.h>
#include "types.h"

void writeFrame(FILE *dataFile, sVec4* positionVals, size_t N, float *frameBuffer);
void progressPrinter(int tSteps, int currentStep, int width, float timePerStep);

typedef struct {
    float4* positionVal;
    uint32_t* originalIndex;
} sorterPair;


#endif
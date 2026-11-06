#ifndef INPUTOUTPUT_H
#define INPUTOUTPUT_H

#include <stdio.h>
#include <stdlib.h>
#include "types.h"

void writeFrame(FILE *dataFile, sVec4* positionVals, size_t N, float *frameBuffer);
void writeOrderedFrame(FILE* datafile, float4* positionVals, uint32_t* originalIndex, size_t N, float* frameBuffer, sorterPair* pairBuffer);
void progressPrinter(int tSteps, int currentStep, int width, float timePerStep);

int keyValueSorter(const void* pointer1, const void* pointer2);


#endif
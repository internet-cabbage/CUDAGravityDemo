#ifndef INPUTOUTPUT_H
#define INPUTOUTPUT_H

#include <stdio.h>
#include "types.h"

void writeFrame(FILE *dataFile, sVec4* positionVals, size_t N, float *frameBuffer);
void progressPrinter(int tSteps, int currentStep, int width);

#endif
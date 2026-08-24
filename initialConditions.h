#ifndef INITIALCONDITIONS_H
#define INITIALCONDITIONS_H

#include "types.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <cglm/cglm.h>
#include <vector_types.h>
#include "units.h"

typedef struct {
    float scaleLength;
    float scaleThickness;

    float totalDiskMass;
    sVec3 rotationAngles; // [Yaw, Pitch, Roll]
    sVec3 galacticCentreLocation; 
    sVec3 galacticNetVelocity;
} galaxyDiskParameters;

void generateLocalCoordsDisk(galaxyDiskParameters* parameters,float* massVals,float4* posMassVals, float3* velVals, int spawnNumber) ;


#endif
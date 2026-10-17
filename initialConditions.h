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



void generateLocalCoordsDisk(galaxyDiskParameters* parameters,float* massVals,float4* posMassVals, float3* velVals, int spawnNumber) ;


#endif
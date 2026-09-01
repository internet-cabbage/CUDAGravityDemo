#ifndef MAIN_H
#define MAIN_H

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

#endif
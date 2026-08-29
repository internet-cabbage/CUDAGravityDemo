#ifndef MACROS_H
#define MACROS_H

#include "stdio.h"
#include "stdlib.h"
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                        \
do {                                                            \
    cudaError_t cudaErr = (call);                                   \
    if (cudaErr != cudaSuccess) {                                   \
        printf("CUDA ERROR at: %s:%d\n",__FILE__,__LINE__);     \
        printf("%s\n",cudaGetErrorString(cudaErr));                 \
        fflush(stdout);                                         \
        exit(-1);                                               \
    }                                                           \
} while (0)

#endif
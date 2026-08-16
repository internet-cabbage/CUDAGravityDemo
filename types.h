#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>

typedef struct __attribute__((aligned(16))) {
    float x,y,z,m;
} vec4;

typedef struct {
    float x,y,z;
} vec3;

typedef struct {
    uint8_t R,G,B;
} RGB;

#endif
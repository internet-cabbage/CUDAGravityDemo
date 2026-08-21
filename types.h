#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>

typedef struct __attribute__((aligned(16))) {
    float x,y,z,m;
} sVec4;

typedef struct {
    float x,y,z;
} sVec3;

typedef struct {
    uint8_t R,G,B;
} RGB;

typedef struct {
    // The coordinates of the corner with the minimum values
    float minX, minY, minZ;

    // The extent is the largest of the width/height/depth values of the cube.
    // The scale is then multiplied by 2^21, which is equal to a multiplication by 1 << 21.
    float scale;
} worldBox;

#endif
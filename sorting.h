#ifndef SORTING_H
#define SORTING_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

extern "C" {
typedef struct {
    void* tempStorage;
    size_t tempStorageBytes;
    int capacity;
} radixSorter;

radixSorter* sorterCreate(int maxCount);

void radixSortPairs(radixSorter* sorter, const uint64_t* keysIn, const uint64_t* keysOut,
                                         const uint64_t* valsIn, const uint64_t* valsOut,
                                         int numCount);

void sorterDestroyer(radixSorter* s);
}
#endif
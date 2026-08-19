#ifndef SORTING_HPP
#define SORTING_HPP

#include <stdint.h>
#include <stdio.h>

#include "types.h"


extern "C" {

size_t sortQueryTemp(int n);
void sortPairs(uint64_t* keysIn, uint64_t* keysOut,
            uint64_t* valsIn, uint64_t* valsOut,
            int n, void* temp, size_t tempBytes);

}

#endif
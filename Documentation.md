#Documentation

In this file I will document what each build file is actually contributing to the project, so I will list all the public structs, functions, and their use cases.

## Data structures (```Types.h```)

All the custom data structures in the program are stored in the ```types.h``` header file, where most of these are used in order to group related parameters into one single data structure, so that I do not need to keep track of dozens of individual parameters.

### Purposes of each struct:

* ```node```: The node data structure basically just groups together all the node information, where a node is effectively just a grouping of stars (or a single star for a leaf node), that is used to speed up the traversal of the tree and the associated force calculation on each star.
* ```radixSorter```: In order to utilise the CUDA CUB (a library of CUDA functions built by Nvidia) radix sort algorithm it needs to be initialised so that it can know how much memory to allocate. This struct contains the pointer to the memory which the sorter will be using, the amount of memory in bytes that the sorter allocated, and the total capacity / number of items which the sorter can sort. This is then just passed to my sorting function, which serves as a wrapper around the CUB radix sort caller.
* ```worldBox```: The world box describes the smallest possible cube that can surround all stars in the simulation. The struct also contains a constant which can be used to convert from the float coordinates of the stars, to an integer along a grid inside the world box.

* ```node```: 
  * ```nodePathFromRoot```: This contains the location of the node in the tree, as as series of instructions determining which octants of the tree needs to be entered starting from the root node. This is obtained based on the star's Morton code which has been truncated down to the level of the node.
  * ```parentIndex```: Contains the index of the parent node in the ```nodes``` array. Where the nodes array
  * ```treeLevel```: This is an integer between 0 and 21, where the greater the number the 'deeper' it is inside the tree. A level of 0 refers to the root node, and 1 would refer to a direct child of the root node etc.
  * ```firstParticleIndex```: The index in the star positions array which the first star in this node lies at. This is used alongside the ```particleCount``` field to determine which stars a given node contains.
  * ```particleCount```: The number of stars that a given node contains. Since each star is arranged contiguously in the star position array, any node with an index between ```firstParticleIndex``` and ```firstParticleIndex + particleCount```, is within the node.
  * ```child[8]```: This stores the index locations of this node's child nodes, inside the ```nodes``` array. A value of -1 means that it is empty. This array is not contiguous, i.e. it is possible to have values like: 10,4,-1,5... as the index value in the child array that the node is in, determines which octant the child node is in, relative to its parent node (the node with child[8] attached to it).
  * ```massData```: This is a CUDA datatype called ```float4```, which is just an array of 4 floats that the GPU is able to read from all at once. The first 3 elements (x,y,z) determine the centre of mass of the node, and the 4th element (w), is the total mass inside the node.
* ```radixSorter```:
  * ```tempStorage```: A pointer which contains the location of the chunk of memory allocated to storing the Morton codes to be sorted.
  * ```tempStorageBytes```: The number of bytes allocated for the sorting.
  * ```capacity```: The maximum number of Morton codes which can be sorted at once.
* ```worldBox```:
  * ```minX```, ```minY```, and ```minZ```: These are the coordinates of the cube's corner with the smallest value.
  * ```extent```: This is the side length of the cube.
  * ```scale```: This is a factor that can be used to convert between the float coordinates of the stars (normalised such that (0.0,0.0,0.0) is the bottom left corner), and the integer grid of coordinates.
* ```treeBuilder```:
  * ```tempStorage```: A pointer which contains the location of the chunk of memory (on the GPU), which is used to perform the prefix sum of the nodeFlags. This is used to identify which node a given star lies inside.
  * ```tempStorageBytes```: The amount of memory allocated to the sum.
  * ```capacity```: The maximum number of nodeFlags which can be summed.
  * ```flagsPing```: 
  * ```flagsPong```:
  * ```offsetsPing```:
  * ```offsetsPong```:
* ```treeState```:
  * TODO
* ```galaxyDiskParameters```:
  * TODO





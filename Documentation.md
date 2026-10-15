#Documentation

In this file I will document what each build file is actually contributing to the project, so I will list all the public structs, functions, and their use cases.

## Data structures (```Types.h```)

All the custom data structures in the program are stored in the ```types.h``` header file, where most of these are used in order to group related parameters into one single data structure, so that I do not need to keep track of dozens of individual parameters.

* ```node```:
  * '''nodePathFromRoot''': This contains the location of the node in the tree, as as series of instructions determining which octants of the tree needs to be entered starting from the root node. This is obtained based on the star's Morton code which has been truncated down to the level of the node.
  * ```parentIndex```: Contains the index of the parent node in the ```nodes``` array. Where the nodes array
  * ```treeLevel```: This is an integer between 0 and 21, where the greater the number the 'deeper' it is inside the tree. A level of 0 refers to the root node, and 1 would refer to a direct child of the root node etc.
  * '''firstParticleIndex''': The index in the star positions array which the first star in this node lies at. This is used alongside the '''particleCount''' field to determine which stars a given node contains.
  * '''particleCount''': The number of stars that a given node contains. Since each star is arranged contiguously in the star position array, any node with an index between '''firstParticleIndex''' and '''firstParticleIndex + particleCount''', is within the node.
  * '''child[8]''': This stores the index locations of this node's child nodes, inside the '''nodes''' array. A value of -1 means that it is empty. This array is not contiguous, i.e. it is possible to have values like: 10,4,-1,5... as the index value in the child array that the node is in, determines which octant the child node is in, relative to its parent node (the node with child[8] attached to it).
  * ```massData```: This is a CUDA datatype called ```float4```, which is just an array of 4 floats that the GPU is able to read from all at once. The first 3 elements (x,y,z) determine the centre of mass of the node, and the 4th element (w), is the total mass inside the node.

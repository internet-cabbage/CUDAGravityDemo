#Documentation

In this file I will document what each build file is actually contributing to the project, so I will list all the public structs, functions, and their use cases.

## Data structures (```Types.h```)

All the custom data structures in the program are stored in the ```types.h``` header file, where most of these are used in order to group related parameters into one single data structure, so that I do not need to keep track of dozens of individual parameters.

* ```node```:
  * '''nodePathFromRoot''': This contains the location of the node in the tree, as as series of instructions determining which octants of the tree needs to be entered starting from the root node. This is obtained based on the star's Morton code which has been truncated down to the level of the node.
  * ```parentIndex```: Contains the index of the parent node in the ```nodes``` array. Where the nodes array 

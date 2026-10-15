# Gravity Simulator

A N-body gravity simulation that uses CUDA to perform the tree construction, force calculation, and integration on the GPU.

**Features**

* The Barnes-Hut algorithm is used in order to group together clustered bodies, so that they can be approximated as a single body to speed up the force calculation.
* Symplectic semi-implicit Euler method used to integrate the motion of the stars in a way that conserves energy.


## Setup

This project requires CUDA and C, so it will only be able to run on a machine with an NVIDIA graphics card.

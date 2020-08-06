### CUDA Genetic Insects

The code is in Add.cu
   
An simple example designed to implement genetic algorithms in CUDA, see Requirements.txt for more information.
   
This can be compiled with 

>    nvcc add.cu -lcurand_static -lculibos -o add_cuda -arch=sm_60

This compiles on Ubuntu 16.04 LTS, with latest cuda sdk, toolkit (must be latest toolkit) and drivers.


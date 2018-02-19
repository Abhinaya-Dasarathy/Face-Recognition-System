#!/bin/bash
#$ -N FaceDetectTEST
#$ -q gpu2


# Module load Cuda Compilers and GCC
module load  cuda/6.0
module load  gcc/4.7.0
module load opencv/2.4.9
module load cmake/3.6.1

cmake .
make
./Face 

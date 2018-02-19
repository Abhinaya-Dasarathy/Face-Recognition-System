#!/bin/bash
#$ -N hpc
#$ -q gpu
#$ -l gpu=1
#$ -pe gpu-node-cores 24











module load opencv/2.4.9
module load cmake/3.6.1
module load openmpi-1.8.3/gcc-4.9.2
module load cuda/6.0
cmake .
make
mpirun -np 8  ./ParallelFaceRecog csv.ext database/s9/1.pgm>>out1.txt
mpirun -np 16 ./ParallelFaceRecog csv.ext database/s9/1.pgm>>out2.txt
mpirun -np 24 ./ParallelFaceRecog csv.ext database/s9/1.pgm>>out3.txt

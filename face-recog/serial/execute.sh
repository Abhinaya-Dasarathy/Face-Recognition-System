#!/bin/bash
#$ -N hpc
#$ -q gpu
#$ -l gpu=1
#$ -pe gpu-node-cores 1











module load opencv/2.4.9
module load cmake/3.6.1
module load openmpi-1.8.3/gcc-4.9.2
module load cuda/6.0
cmake .
make
mpirun -np 1  ./SerialFaceRecog csv.ext database/s9/1.pgm>serial_output.txt



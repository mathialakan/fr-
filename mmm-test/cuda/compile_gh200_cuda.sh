#!/bib/bash
#

module use /soft/modulefiles/

module load cuda

export COMPILER=nvcc
export API=CUDA
export SYS=gh200

make

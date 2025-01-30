#!/bib/bash


export COMPILER=llvm
export API=OMP_OL
export SYS=a100

export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/

make

#!/bib/bash
#


export COMPILER=nvcc
export API=CUDA
export SYS=a100

export CUDA_IPATH=${NVIDIA_PATH}/cuda/include
export CUDA_LPATH=${NVIDIA_PATH}/cuda/lib64

make

#!/bib/bash
#

 module use  use /soft/modulefiles/
 module load rocm
 module load cudatoolkit-standalone

 export HIP_PLATFORM=nvidia
 export HCC_HOME=/soft/compilers/rocm/6.2.4/clr-install/

 export GPU_LIBRARY_PATH=${HCC_HOME}/lib
 export GPU_INCLUDE_PATH=${HCC_HOME}/include

 export CUDA_LPATH=${CUDA_HOME}/lib64
 export CUDA_IPATH=${CUDA_HOME}/include


export COMPILER=hipcc
export API=HIP
export SYS=a100


make

#!/bib/bash


export COMPILER=nvhpc
export API=OMP_OL
export SYS=a100

export TYPE=$1

make

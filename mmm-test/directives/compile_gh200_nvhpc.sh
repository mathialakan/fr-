#!/bib/bash
#

module use /soft/modules/

module load nvhpc

export COMPILER=nvhpc
export API=OMP_OL
export SYS=gh200

export TYPE=$1

make

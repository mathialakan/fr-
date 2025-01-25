#!/bib/bash
#

module use /soft/modules/

module load nvhpc

export COMPILER=nvhpc
export API=OMP_OL

make

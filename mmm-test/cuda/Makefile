
#CC= cpp -P

CXXFLAGS = -std=c++20

SRC = $(wildcard *.cu)
INC = ../../fraction/$(wildcard *fr.h)
OBJ = $(SRC:.cu=.o)

API=cuda
ifeq ($(TYPE), int)
	TYPE_FLAG = -DINT_1D
else
ifeq ($(TYPE), float)
        TYPE_FLAG = -DFLOAT_1D	
else
ifeq ($(TYPE), double)
        TYPE_FLAG = -DDOUBLE_1D
else	
ifeq ($(TYPE), fraction)
        TYPE_FLAG = -DFRACTION_1D
else
ifeq ($(TYPE), mixed)
        TYPE_FLAG = -DMIXED_1D
endif
endif
endif
endif
endif

ifeq ($(API), cuda)
	ARCH=sm_80
	CXX = nvcc
	CXXFLAGS += -Xcompiler -Wall -arch=$(ARCH) -L${CUDA_LPATH} -I${CUDA_IPATH} -DGPU ${TYPE_FLAG}
endif

EX=mmm_$(API)_$(COMPILER)_$(SYS)_${TYPE}

$(EX): $(OBJ)
	$(CXX) -o $@ $(CXXFLAGS) $(OBJ) $(LIB)

$(OBJ): %.o: %.cu
	$(CXX) $(CXXFLAGS) -c $<

$(OBJ): $(INC)

clean:
	rm -f *.o *~ ./$(EX)



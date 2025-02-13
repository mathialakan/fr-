
#CC= cpp -P

CXXFLAGS = -std=c++20

SRC = $(wildcard *.cpp)
INC = ../../fraction/$(wildcard *fr.h)
OBJ = $(SRC:.cpp=.o)

API=hip
HIP_PATH=/soft/compilers/rocm/6.2.4/clr-install/
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

ifeq ($(API), hip)
	ARCH=sm_80
	CXX = $(HIP_PATH)/bin/hipcc
	CXXFLAGS += --gpu-architecture=compute_80  --gpu-code=$(ARCH) -L${CUDA_LPATH} -I${CUDA_IPATH} -D__HIP_PLATFORM_NVIDIA__ -DGPU ${TYPE_FLAG}
endif

EX=mmm_$(API)_$(COMPILER)_$(SYS)_$(TYPE)

$(EX): $(OBJ)
	$(CXX) -o $@ $(CXXFLAGS) $(OBJ) $(LIB)

$(OBJ): %.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

$(OBJ): $(INC)

clean:
	rm -f *.o *~ ./$(EX)



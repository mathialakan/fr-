
#CC= cpp -P

CXXFLAGS = -std=c++14

SRC = $(wildcard *.cu)
INC = ../../fraction/$(wildcard *fr.h)
OBJ = $(SRC:.cu=.o)

API=cuda

ifeq ($(API), cuda)
	ARCH=sm_80
	CXX = nvcc
	CXXFLAGS += -Xcompiler -Wall -arch=$(ARCH) -L${CUDA_LPATH} -I${CUDA_IPATH} -DGPU
endif

EX=mmm_$(API)_$(COMPILER)_$(SYS)

$(EX): $(OBJ)
	$(CXX) -o $@ $(CXXFLAGS) $(OBJ) $(LIB)

$(OBJ): %.o: %.cu
	$(CXX) $(CXXFLAGS) -c $<

$(OBJ): $(INC)

clean:
	rm -f *.o *~ ./$(EX)



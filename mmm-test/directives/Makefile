CC= cpp -P
#CXX= mpicxx

CXXFLAGS = -std=c++20

SRC = $(wildcard *.cpp)
INC = ../../fraction/$(wildcard *fr.h)
OBJ = $(SRC:.cpp=.o)

#API=OMP_OL
#COMPILER=icp
#SYS=pvc

ifeq ($(API), OMP_OL)
	CXXFLAGS += -DOMP_OL
ifeq ($(COMPILER), icp)
	CXX = mpicxx
	CXXFLAGS +=-fiopenmp -fopenmp-targets=spir64_gen -Xopenmp-target-backend=spir64_gen "-device 12.60.7"
else ifeq ($(COMPILER), nvhpc)
	CXX = nvc++
	CXXFLAGS += -std=c++17 -mp=gpu -fast -gpu=cc80 -gpu=maxrregcount:64
else ifeq ($(COMPILER), llvm)
	CXX = clang++
	CXXFLAGS += -Ofast -std=c++17 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=$(CUDA_PATH) -Xcuda-ptxas --maxrregcount=64 # -fopenmp-new-driver -foffload-lto -fopenmp-assume-no-thread-state
else ifeq ($(COMPILER), AOMP)
      CPP=clang++  -fopenmp -target x86_64-pc-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$(ARCH) #gfx908
endif
else ifeq ($(API), OMP)
        CXXFLAGS +=-fiopenmp -DOMP
endif

EX=mmm_$(API)_$(COMPILER)_$(SYS)

$(EX): $(OBJ)
	$(CXX) -o $@ $(CXXFLAGS) $(OBJ) $(LIB)

$(OBJ): %.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

$(OBJ): $(INC)

clean:
	rm -f *.o *~ ./$(EX)

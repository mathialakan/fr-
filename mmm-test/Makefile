
CC= cpp -P
CXX= mpicxx

CXXFLAGS = -std=c++20

SRC = $(wildcard *.cpp)
INC = ../fraction/$(wildcard *fr.h)
OBJ = $(SRC:.cpp=.o)

API="OMP_OL"
COMPILER=icp
SYS=pvc

ifeq ($(API), "OMP_OL")
	CXXFLAGS +=-fiopenmp -fopenmp-targets=spir64_gen -Xopenmp-target-backend=spir64_gen "-device 12.60.7"
else ifeq($(API), "OMP")
	CXXFLAGS +=-fiopenmp
endif

EX=mmm_$(API)_$(COMPILER)_$(SYS)

$(EX): $(OBJ)
	$(CXX) -o $@ $(CXXFLAGS) $(OBJ) $(LIB)

$(OBJ): %.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

$(OBJ): $(INC)

clean:
	rm -f *.o *~ ./$(EX)



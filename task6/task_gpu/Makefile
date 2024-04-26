CXX = pgc++
CXXFLAGS = -O2 -Wall -std=c++20 -fast -acc=gpu -gpu=ccall -Minfo=all
LD = pgc++
LDFLAGS = -fast -acc -gpu=ccall -lstdc++ -lboost_program_options
NVTXLIB = -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include/

SOURCES = main.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGETS = main

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(NVTXLIB) -c -o $@ $<

all: $(TARGETS)

main: $(OBJECTS)
	$(LD) -o $@ $^ $(LDFLAGS) 

clean:
	rm -f $(TARGETS) $(OBJECTS) *.mod core

.PHONY: all clean

CXX = g++
CXXFLAGS = -std=c++17 -Wall

ifdef USE_DOUBLE
CXXFLAGS += -DUSE_DOUBLE
endif

task1: task1.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< -lm

clean:
	rm -f task1
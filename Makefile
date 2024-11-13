# Compiler
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -m64 -I. # General flags
OPENMP_FLAGS = -fopenmp -Wno-unknown-pragmas # Only for OpenMP programs

# Executable targets
TARGETS = OpenMPDinic SequentialDinic EdgeOpenMPDinic

# Source files
SRCS_OPENMP = OpenMPDinics.cpp
SRCS_SEQ = SequentialDinics.cpp
SRCS_EDGE = EdgeOpenMPDinics.cpp

# Object files
OBJS_OPENMP = $(SRCS_OPENMP:.cpp=.o)
OBJS_SEQ = $(SRCS_SEQ:.cpp=.o)
OBJS_EDGE = $(SRCS_EDGE:.cpp=.o)

# Default rule to build all programs
all: $(TARGETS)

# Rule to build the sequential version
SequentialDinic: $(OBJS_SEQ)
	$(CXX) $(CXXFLAGS) -o SequentialDinic $(OBJS_SEQ)

# Rule to build the OpenMP version
OpenMPDinic: $(OBJS_OPENMP)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -o OpenMPDinic $(OBJS_OPENMP)

# Rule to build the Edge-based OpenMP version
EdgeOpenMPDinic: $(OBJS_EDGE)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -o EdgeOpenMPDinic $(OBJS_EDGE)

# Rule to compile object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up object files and the executables
clean:
	rm -f $(OBJS_OPENMP) $(OBJS_SEQ) $(OBJS_EDGE) $(TARGETS)

# Phony targets (not real files)
.PHONY: all clean OpenMPDinic SequentialDinic EdgeOpenMPDinic
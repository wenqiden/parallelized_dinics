# Compiler
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -m64 -I. -fopenmp -Wno-unknown-pragmas

# OpenMP flags
# OPENMP_FLAGS = -m64 -I. -fopenmp -Wno-unknown-pragmas

# Executable targets
TARGETS = OpenMPDinic SequentialDinic

# Source files
SRCS_OPENMP = OpenMPDinics.cpp
SRCS_SEQ = SequentialDinics.cpp

# Object files
OBJS_OPENMP = $(SRCS_OPENMP:.cpp=.o)
OBJS_SEQ = $(SRCS_SEQ:.cpp=.o)

# Default rule to build both programs
all: $(TARGETS)

# Rule to build the OpenMP version
OpenMPDinic: $(OBJS_OPENMP)
	$(CXX) $(CXXFLAGS) -o OpenMPDinic $(OBJS_OPENMP)

# Rule to build the sequential version
SequentialDinic: $(OBJS_SEQ)
	$(CXX) $(CXXFLAGS) -o SequentialDinic $(OBJS_SEQ)

# Rule to build object files for OpenMP
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to build object files for sequential
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up object files and the executables
clean:
	rm -f $(OBJS_OPENMP) $(OBJS_SEQ) $(TARGETS)

# Phony targets (not real files)
.PHONY: all clean OpenMPDinic SequentialDinic
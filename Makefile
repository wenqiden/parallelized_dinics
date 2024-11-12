# Compiler
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -g

# OpenMP flags
OPENMP_FLAGS = -fopenmp

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
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -o $@ $(OBJS_OPENMP)

# Rule to build the sequential version
SequentialDinic: $(OBJS_SEQ)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_SEQ)

# Rule to compile object files for OpenMP
OpenMPDinics.o: OpenMPDinics.cpp
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -c $< -o $@

# Rule to compile object files for sequential
SequentialDinics.o: SequentialDinics.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up object files and the executables
clean:
	rm -f $(OBJS_OPENMP) $(OBJS_SEQ) $(TARGETS)

# Phony targets (not real files)
.PHONY: all clean OpenMPDinic SequentialDinic
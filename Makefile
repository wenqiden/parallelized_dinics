SHELL := /bin/bash

# Compiler
CXX = g++
MPICXX = mpic++
CXXFLAGS = -std=c++17 -O3 -g -Wall -m64 -I. -fopenmp -Wno-unknown-pragmas
MPIFLAGS = -Wall -Wextra -O3 -std=c++20 -I. -g

# Executable targets
TARGETS = DFSOpenMPDinic OpenMPDinic SequentialDinic EdgeOpenMPDinic LazyDFSOpenMPDinic BFSMPIDinic

# Source files
SRCS_OPENMP = OpenMPDinics.cpp
SRCS_SEQ = SequentialDinics.cpp
SRCS_EDGE = EdgeOpenMPDinics.cpp
SRCS_DFS = DFSOpenMPDinics.cpp
SRCS_LAZY_DFS = LazyDFSOpenMPDinics.cpp
SRCS_BFS_MPI = 1DBFSMPIDinics.cpp

# Object files
OBJS_OPENMP = $(SRCS_OPENMP:.cpp=.o)
OBJS_SEQ = $(SRCS_SEQ:.cpp=.o)
OBJS_EDGE = $(SRCS_EDGE:.cpp=.o)
OBJS_DFS = $(SRCS_DFS:.cpp=.o)
OBJS_LAZY_DFS = $(SRCS_LAZY_DFS:.cpp=.o)
OBJS_BFS_MPI = $(SRCS_BFS_MPI:.cpp=.o)

# Default rule to build all programs
all: $(TARGETS)

# Rule to build the sequential version (no OpenMP)
SequentialDinic: $(OBJS_SEQ)
	$(CXX) $(CXXFLAGS) -o SequentialDinic $(OBJS_SEQ)

# Rule to build the OpenMP version
OpenMPDinic: $(OBJS_OPENMP)
	$(CXX) $(CXXFLAGS) -o OpenMPDinic $(OBJS_OPENMP)

# Rule to build the Edge-based OpenMP version
EdgeOpenMPDinic: $(OBJS_EDGE)
	$(CXX) $(CXXFLAGS) -o EdgeOpenMPDinic $(OBJS_EDGE)

# Rule to build the DFS-based OpenMP version
DFSOpenMPDinic: $(OBJS_DFS)
	$(CXX) $(CXXFLAGS) -o DFSOpenMPDinic $(OBJS_DFS)

LazyDFSOpenMPDinic: $(OBJS_LAZY_DFS)
	$(CXX) $(CXXFLAGS) -o LazyDFSOpenMPDinic $(OBJS_LAZY_DFS)

BFSMPIDinic: $(OBJS_BFS_MPI)
	$(MPICXX) $(MPIFLAGS) -o BFSMPIDinic $(OBJS_BFS_MPI)

# Rule to compile object files for the sequential program (no OpenMP)
%.o: %.cpp
	@if [[ "$<" == *1DBFSMPIDinics.cpp ]]; then \
		$(MPICXX) $(MPIFLAGS) -c $< -o $@; \
	else \
		$(CXX) $(CXXFLAGS) -c $< -o $@; \
	fi

# Clean up object files and the executables
clean:
	rm -f $(OBJS_OPENMP) $(OBJS_SEQ) $(OBJS_EDGE) $(OBJS_DFS) $(OBJS_LAZY_DFS) $(OBJS_BFS_MPI) $(TARGETS)

# Phony targets (not real files)
.PHONY: all clean OpenMPDinic SequentialDinic EdgeOpenMPDinic DFSOpenMPDinic LazyDFSOpenMPDinic BFSMPIDinic
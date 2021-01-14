NVCC ?= nvcc
hypre ?= true
gpu ?= false
hdf5 ?= true
config ?= production
precision ?= double

# SMALL HELPER FOR OSX USERS
ifeq ($(shell uname -s), Darwin)
include make.macos
endif

# SET FLAGS FOR COMPILER
CXX=$(MPICXX)
LD=$(MPICXX)

ifeq "$(shell $(CXX) --version | grep -ci 'icpc')" "0"
ifeq "$(shell $(CXX) --version | grep -ci 'llvm')" "0"
compiler=gnu
include make.gcc.flags
else
compiler=llvm
include make.llvm.flags
endif
else
compiler=intel
include make.icc.flags
endif

# SET FLAGS FOR CUBISM
bs ?= 32
nthreads ?= 12
CPPFLAGS+= -D_BS_=$(bs) -DCUBISM_ALIGNMENT=32 -DNTHREADS=$(nthreads) -I$(BUILDDIR)/../Cubism/include/
NVFLAGS+= -code=sm_60 -arch=compute_60 -I$(BUILDDIR)/../Cubism/include/

# SET VPATH FOR MAKE TO SEARCH FOR FILES
BUILDDIR = .
DIRS = $(sort $(dir $(wildcard ../source/*/)))
VPATH := $(DIRS) $(BUILDDIR)/../Cubism/src/

OBJECTS = \
		Simulation.o SimulationData.o Shape.o ShapeLibrary.o ShapesSimple.o \
		PressureSingle.o PressureIterator_unif.o PressureIterator_approx.o \
		HYPREdirichlet.o FFTW_freespace.o PutObjectsOnGrid.o UpdateObjects.o \
		advDiff.o BufferedLogger.o Helpers.o Glider.o Fish.o SmartCylinder.o \
		FishLibrary.o ArgumentParser.o PoissonSolver.o advDiffGravStaggered.o \
		advDiffGrav.o BlowFish.o StefanFish.o CarlingFish.o ComputeForces.o \
		HYPREdirichletVarRho.o PressureVarRho_proper.o UpdateObjectsStaggered.o \
		PutObjectsOnGridStaggered.o Naca.o CStartFish.o ZebraFish.o \
		NeuroKinematicFish.o advDiff_implicit_all.o LinearSolverDelta.o

# SET FLAGS FOR GPU-BASED SOLVERS
ifneq "$(gpu)" "false"
LIBS += -lcufft -L$(AMGX_DIR)/lib -lamgx -lcublas -lcusparse -lcusolver -lnvToolsExt
OBJECTS += CUDA_all.o AMGXdirichletVarRho.o
CPPFLAGS += -DCUDAFFT -DAMGX_POISSON -I$(AMGX_DIR)/include
endif

# SET FLAGS FOR FFTW
ifneq ($(FFTW_ROOT),)
LIBS  += -L$(FFTW_ROOT)/lib
CPPFLAGS  += -I$(FFTW_ROOT)/include
else
ERROR += "FFTW_ROOT not set"
endif

ifeq "$(precision)" "single"
CPPFLAGS += -D_FLOAT_PRECISION_
NVFLAGS += -D_FLOAT_PRECISION_
LIBS += -lfftw3f -lfftw3f_omp
else 
LIBS += -lfftw3  -lfftw3_omp
endif

# SET FLAGS FOR HYPRE
ifneq "$(hypre)" "false"
ifneq ($(HYPRE_ROOT),)
LIBS     += -L$(HYPRE_ROOT)/lib -lHYPRE
CPPFLAGS += -I$(HYPRE_ROOT)/include -DHYPREFFT
else
ERROR += "HYPRE_ROOT not set"
endif
endif

# SET FLAGS FOR GSL
ifneq ($(GSL_ROOT),)
CPPFLAGS += -I$(GSL_ROOT)/include
LIBS += -L$(GSL_ROOT)/lib -lgsl -lgslcblas
else
ERROR += "GSL_ROOT not set"
endif

# SET FLAGS FOR HDF5
ifneq "$(hdf)" "false"
ifneq ($(HDF5_ROOT),)
LIBS     += -L$(HDF5_ROOT)/lib -lhdf5
CPPFLAGS += -I$(HDF5_ROOT)/include -DCUBISM_USE_HDF
else
ERROR += "HDF5_ROOT not set"
endif
endif

ifneq ($(ERROR),)
$(error $(ERROR))
endif

all: simulation libcup.a cup.cflags.txt cup.libs.txt
.DEFAULT: all;

# COMPILATION INSTRUCTIONS FOR SMARTIES APPS
leadFollow: $(OBJECTS)
	$(CXX) $(CPPFLAGS) -I${SMARTIES_ROOT}/include -c ../source/main_RL_$@.cpp -o main_RL_$@.o
	$(LD) -o $@ $^ main_RL_$@.o $(LIBS) -L${SMARTIES_ROOT}/lib -lsmarties

blowfish: $(OBJECTS)
	$(CXX) $(CPPFLAGS) -I${SMARTIES_ROOT}/include -c ../source/main_RL_$@.cpp -o main_RL_$@.o
	$(LD) -o $@ $^ main_RL_$@.o $(LIBS) -L${SMARTIES_ROOT}/lib -lsmarties

cStart: $(OBJECTS)
	$(CXX) $(CPPFLAGS) -I${SMARTIES_ROOT}/include -c ../source/main_RL_$@.cpp -o main_RL_$@.o
	$(LD) -o $@ $^ main_RL_$@.o $(LIBS) -L${SMARTIES_ROOT}/lib -lsmarties

zebrafish: $(OBJECTS)
	$(CXX) $(CPPFLAGS) -I${SMARTIES_ROOT}/include -c ../source/main_RL_$@.cpp -o main_RL_$@.o
	$(LD) -o $@ $^ main_RL_$@.o $(LIBS) -L${SMARTIES_ROOT}/lib -lsmarties

neurokinematicfish: $(OBJECTS)
	$(CXX) $(CPPFLAGS) -I${SMARTIES_ROOT}/include -c ../source/main_RL_$@.cpp -o main_RL_$@.o
	$(LD) -o $@ $^ main_RL_$@.o $(LIBS) -L${SMARTIES_ROOT}/lib -lsmarties

smartCyl: $(OBJECTS)
	$(CXX) $(CPPFLAGS) -I${SMARTIES_ROOT}/include -c ../source/main_RL_$@.cpp -o main_RL_$@.o
	$(LD) -o $@ $^ main_RL_$@.o $(LIBS) -L${SMARTIES_ROOT}/lib -lsmarties

glider: $(OBJECTS)
	$(CXX) $(CPPFLAGS) -I${SMARTIES_ROOT}/include -c ../source/main_RL_$@.cpp -o main_RL_$@.o
	$(LD) -o $@ $^ main_RL_$@.o $(LIBS) -L${SMARTIES_ROOT}/lib -lsmarties

# COMPILATION INSTRUCTIONS FOR STANDALONE APPLICATION OR AS LIBRARY
simulation: main.o $(OBJECTS)
	$(LD) main.o $(OBJECTS) $(LIBS) -o $@

libcup.a: $(OBJECTS)
	ar rcs $@ $(OBJECTS)

cup.cflags.txt:
	echo $(CPPFLAGS) > cup.cflags.txt

cup.libs.txt:
	echo $(LIBS) > cup.libs.txt

# COMPILATION INSTRUCTIONS FOR OBJECT FILES
%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(CXX) $(CPPFLAGS) -c -MD $<

CUDA_all.o: CUDA_all.cu
	nvcc $(NVFLAGS) -c $< -o $@

# COMPILATION INSTRUCTION FOR CLEANING BUILD
clean:
	rm -f simulation libcup.a cup.cflags.txt cup.libs.txt
	rm -f *.o *.d

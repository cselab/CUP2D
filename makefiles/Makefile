config ?= production
precision ?= double
bs ?= 8
gpu ?= false
openmp ?= false
profile ?= false

# SMALL HELPER FOR OSX USERS
ifeq ($(shell uname -s), Darwin)
  include make.macos
endif

# SET FLAGS FOR COMPILER
ifneq ($(MPICXX),)
  CXX=$(MPICXX)
else
  ERROR += "MPICXX not set"
endif

ifneq ($(ERROR),)
	$(error $(ERROR))
endif

# LOAD FLAGS -- GCC NEEDED
CPPFLAGS+= -std=c++17 -Wall
CPPFLAGS+= -Wextra -Wfloat-equal -Wcast-align -Woverloaded-virtual
CPPFLAGS+= -Wlogical-op -Wmissing-declarations -Wredundant-decls -Wshadow
CPPFLAGS+= -Wwrite-strings -Wno-unused-parameter #-Wno-cpp
CPPFLAGS+= -Wno-float-equal
CPPFLAGS+= -Wno-redundant-decls

ifeq "$(openmp)" "true"
	CPPFLAGS+= -fopenmp
endif

# FLAGS FOR EXTERNAL LIBRARIES
LIBS+= -lgsl -lgslcblas -fopenmp -lhdf5

# ADD LIBRARY PATHS IF GIVEN
ifneq ($(HDF5_ROOT),)
	LIBS     += -L$(HDF5_ROOT)/lib
	CPPFLAGS += -I$(HDF5_ROOT)/include
endif

ifneq ($(GSL_ROOT),)
	CPPFLAGS += -I$(GSL_ROOT)/include
	LIBS += -L$(GSL_ROOT)/lib
endif

# ENABLE OPTIMIZATION/DEBUG FLAGS IF WISHED
ifeq "$(findstring prod,$(config))" ""
	CPPFLAGS+= -g -O0
	ifeq "$(config)" "segf"
		CPPFLAGS+= -fsanitize=address
		LIBS+= -fsanitize=address -static-libasan
	endif
	ifeq "$(config)" "nans"
		CPPFLAGS+= -fsanitize=undefined
		LIBS+= -fsanitize=undefined
	endif
else
	CPPFLAGS+= -DNDEBUG -O3 -fstrict-aliasing -march=native -mtune=native -falign-functions -ftree-vectorize -fmerge-all-constants
	CPPFLAGS+= -ffast-math
	# CPPFLAGS+= -DCUP2D_PRESERVE_SYMMETRY
endif

# USE MEMORY SAVING FOR MANY LEVEL RUNS
usemap ?= true
ifeq "$(usemap)" "true"
  CPPFLAGS += -DCUBISM_USE_MAP
endif

# SET FLOATING POINT ACCURACY
ifeq "$(precision)" "single"
  CPPFLAGS += -D_FLOAT_PRECISION_
else ifeq "$(precision)" "double"
  CPPFLAGS += -D_DOUBLE_PRECISION_
else ifeq "$(precision)" "long_double"
  CPPFLAGS += -D_LONG_DOUBLE_PRECISION_
endif

# SET VPATH FOR MAKE TO SEARCH FOR FILES
BUILDDIR = .
DIRS = $(sort $(dir $(wildcard ../source/*/)))
VPATH := $(DIRS) $(BUILDDIR)/../Cubism/src/

# SET FLAGS FOR CUBISM
CPPFLAGS+= -D_BS_=$(bs) -DCUBISM_ALIGNMENT=32 -I$(BUILDDIR)/../Cubism/include/ -DDIMENSION=2

OBJECTS = \
		Simulation.o SimulationData.o BufferedLogger.o Helpers.o ArgumentParser.o \
		PressureSingle.o PutObjectsOnGrid.o advDiff.o ComputeForces.o\
		AdaptTheMesh.o AMRSolver.o Shape.o ShapeLibrary.o ShapesSimple.o \
		Fish.o FishData.o SmartCylinder.o StefanFish.o CarlingFish.o  \
		Naca.o CStartFish.o ZebraFish.o NeuroKinematicFish.o  Windmill.o \
		Waterturbine.o Teardrop.o ExperimentFish.o Base.o Forcing.o advDiffSGS.o

# ENABLE GPU SUPPORT
ifeq ("$(gpu)", "true")
	OBJECTS += ExpAMRSolver.o BiCGSTAB.o LocalSpMatDnVec.o
	CPPFLAGS += -fopenmp -DGPU_POISSON
	NVCCFLAGS := -std=c++17 -O3 --use_fast_math -Xcompiler "$(CPPFLAGS)" -DGPU_POISSON
	LIBS += -lcudart -lcublas -lcusparse
  
  ifeq ("$(profile)", "true")
    CPPFLAGS += -DBICGSTAB_PROFILER
  endif
else
  CPPFLAGS += -Wno-unknown-pragmas
endif

# DEFINE COMPILATION TARGETS
all: debugRL simulation libcup.a cup.cflags.txt cup.libs.txt
.DEFAULT: all;

# COMPILATION INSTRUCTIONS FOR APPLICATION THAT CAN REPRODUCE AN RL RUN
debugRL: debugRL.o $(OBJECTS)
	$(CXX) debugRL.o $(OBJECTS) $(LIBS) -o $@

# COMPILATION INSTRUCTIONS FOR APPLICATION AND LIBRARY
simulation: main.o $(OBJECTS)
	$(CXX) main.o $(OBJECTS) $(LIBS) -o $@

libcup.a: $(OBJECTS)
	ar rcs $@ $(OBJECTS)

cup.cflags.txt:
	echo '$(CPPFLAGS)' > cup.cflags.txt

cup.libs.txt:
	echo '$(LIBS)' > cup.libs.txt

# COMPILATION INSTRUCTIONS FOR OBJECT FILES
%.o: %.cu
	$(NVCC) -ccbin=$(CXX) $(NVCCFLAGS) -c $< -o $@

%.d: %.cu
	$(NVCC) -ccbin=$(CXX) $(NVCCFLAGS) -c -MD $<

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(CXX) $(CPPFLAGS) -c -MD $<

# COMPILATION INSTRUCTION FOR CLEANING BUILD
clean:
	rm -f debugRL simulation libcup.a cup.cflags.txt cup.libs.txt
	rm -f *.o *.d

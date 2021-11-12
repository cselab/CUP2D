hdf5 ?= true
config ?= production
precision ?= double
bs ?= 8

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
CPPFLAGS+= -D_BS_=$(bs) -DCUBISM_ALIGNMENT=32 -I$(BUILDDIR)/../Cubism/include/ -DDIMENSION=2

# SET VPATH FOR MAKE TO SEARCH FOR FILES
BUILDDIR = .
DIRS = $(sort $(dir $(wildcard ../source/*/)))
VPATH := $(DIRS) $(BUILDDIR)/../Cubism/src/

OBJECTS = \
		Simulation.o SimulationData.o BufferedLogger.o Helpers.o ArgumentParser.o \
		PressureSingle.o PutObjectsOnGrid.o advDiff.o ComputeForces.o\
		AdaptTheMesh.o AMRSolver.o Shape.o ShapeLibrary.o ShapesSimple.o \
		Fish.o FishData.o SmartCylinder.o StefanFish.o CarlingFish.o  \
		Naca.o CStartFish.o ZebraFish.o NeuroKinematicFish.o  Windmill.o \
		Waterturbine.o Teardrop.o ExperimentFish.o

ifeq "$(precision)" "single"
CPPFLAGS += -D_FLOAT_PRECISION_
NVFLAGS += -D_FLOAT_PRECISION_
endif

usemap ?= true
ifeq "$(usemap)" "true"
CPPFLAGS += -DCUBISM_USE_MAP
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
	echo $(CPPFLAGS) > cup.cflags.txt

cup.libs.txt:
	echo $(LIBS) > cup.libs.txt

# COMPILATION INSTRUCTIONS FOR OBJECT FILES
%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(CXX) $(CPPFLAGS) -c -MD $<

# COMPILATION INSTRUCTION FOR CLEANING BUILD
clean:
	rm -f simulation libcup.a cup.cflags.txt cup.libs.txt
	rm -f *.o *.d

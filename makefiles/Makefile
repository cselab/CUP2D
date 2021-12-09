hdf5 ?= true
config ?= production
precision ?= double

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
CPPFLAGS+= -D_BS_=8 -DCUBISM_ALIGNMENT=32 -I$(BUILDDIR)/../Cubism/include/ -DDIMENSION=2

# SET VPATH FOR MAKE TO SEARCH FOR FILES
BUILDDIR = .
DIRS = $(sort $(dir $(wildcard ../source/*/)))
VPATH := $(DIRS) $(BUILDDIR)/../Cubism/src/

OBJECTS = \
		Simulation.o SimulationData.o Shape.o ShapeLibrary.o ShapesSimple.o \
		PressureSingle.o PutObjectsOnGrid.o advDiff.o BufferedLogger.o Helpers.o Fish.o SmartCylinder.o \
		FishData.o ArgumentParser.o StefanFish.o CarlingFish.o ComputeForces.o \
		Naca.o CStartFish.o ZebraFish.o NeuroKinematicFish.o AdaptTheMesh.o AMRSolver.o Windmill.o


ifeq "$(precision)" "single"
CPPFLAGS += -D_FLOAT_PRECISION_
NVFLAGS += -D_FLOAT_PRECISION_
endif

usemap ?= false
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

all: debugRL simulation cudasimulation libcup.a cup.cflags.txt cup.libs.txt
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
	$(CXX) $(CPPFLAGS) $(LIBS) -c $< -o $@

%.d: %.cpp
	$(CXX) $(CPPFLAGS) $(LIBS) -c -MD $<

# COMPILATION INSTRUCTIONS FOR CUDA ENABLED CODE
NVOBJECTS = \
		cudaSimulation.o cudaSimulationData.o cudaShape.o cudaShapeLibrary.o \
		cudaShapesSimple.o cudaPressureSingle.o cudaPutObjectsOnGrid.o cudaadvDiff.o \
		cudaBufferedLogger.o cudaHelpers.o cudaFish.o cudaSmartCylinder.o \
		cudaFishData.o cudaArgumentParser.o cudaStefanFish.o cudaCarlingFish.o \
		cudaComputeForces.o cudaNaca.o cudaCStartFish.o cudaZebraFish.o \
		cudaNeuroKinematicFish.o cudaAdaptTheMesh.o cudaWindmill.o
NVOBJECTS += cuAMRSolver.o # give a different prefix to differentiate %.cpp generated .o

# Set $(CXX) to do host-side compilation and pass its flags
NVFLAGS += -ccbin=$(CXX) -Xcompiler "$(CPPFLAGS)"
NVLIBS += -Xcompiler "$(LIBS)"
NVLIBS += -lcudart -lcublas -lcusparse
# Program specific definitions 
NVFLAGS += -DGPU_POISSON -DBICGSTAB_PROFILER
# Arch flags for maximum compatibility
#NVFLAGS += -arch=sm_52 \
#					 -gencode=arch=compute_52,code=sm_52 \
#					 -gencode=arch=compute_60,code=sm_60 \
#					 -gencode=arch=compute_61,code=sm_61 \
#					 -gencode=arch=compute_70,code=sm_70 \
#					 -gencode=arch=compute_75,code=sm_75 \
#					 -gencode=arch=compute_80,code=sm_80 \
#					 -gencode=arch=compute_86,code=sm_86 \
#					 -gencode=arch=compute_86,code=compute_86
# Arch flags for maximum performance on a particular system
NVFLAGS+= -code=sm_60 -arch=compute_60

cuAMRSolver.o: AMRSolver.cu 
	$(NVCC) $(NVFLAGS) $(NVLIBS) -c $< -o $@

cuda%.o : %.cpp
	$(NVCC) $(NVFLAGS) $(NVLIBS) -c $< -o $@

cudasimulation: cudamain.o $(NVOBJECTS)
	$(NVCC) cudamain.o $(NVOBJECTS) $(NVFLAGS) $(NVLIBS) -o $@

# COMPILATION INSTRUCTION FOR CLEANING BUILD
clean:
	rm -f simulation debugRL cudasimulation libcup.a cup.cflags.txt cup.libs.txt
	rm -f *.o *.d

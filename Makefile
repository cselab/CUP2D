config ?= production
precision ?= double
bs ?= 8
gpu ?= false
openmp ?= false
profile ?= false
symmetry ?= false
cylinder_ref ?= false

ifeq "$(precision)" "single"
	CPPFLAGS += -D_FLOAT_PRECISION_
else ifeq "$(precision)" "double"
	CPPFLAGS += -D_DOUBLE_PRECISION_
else ifeq "$(precision)" "long_double"
	CPPFLAGS += -D_LONG_DOUBLE_PRECISION_
endif

ifeq "$(symmetry)" "true"
	CPPFLAGS += -DCUP2D_PRESERVE_SYMMETRY
else ifneq "$(findstring prod,$(config))" ""
	CPPFLAGS+= -ffast-math
endif
ifeq "$(cylinder_ref)" "true"
	CPPFLAGS += -DCUP2D_CYLINDER_REF
endif
CPPFLAGS+= -D_BS_=$(bs) -DCUBISM_ALIGNMENT=32
CPPFLAGS += -ICubism/include -DDIMENSION=2
OBJECTS = \
Simulation.o \
Cubism/src/ArgumentParser.o \
Obstacles/CarlingFish.o \
Obstacles/CStartFish.o \
Obstacles/CylinderNozzle.o \
Obstacles/ExperimentFish.o \
Obstacles/FishData.o \
Obstacles/Fish.o \
Obstacles/Naca.o \
Obstacles/NeuroKinematicFish.o \
Obstacles/ShapeLibrary.o \
Obstacles/ShapesSimple.o \
Obstacles/SmartCylinder.o \
Obstacles/SmartNaca.o \
Obstacles/StefanFish.o \
Obstacles/Teardrop.o \
Obstacles/Waterturbine.o \
Obstacles/Windmill.o \
Obstacles/ZebraFish.o \
Operators/AdaptTheMesh.o \
Operators/advDiff.o \
Operators/advDiffSGS.o \
Operators/ComputeForces.o \
Operators/Forcing.o \
Operators/Helpers.o \
Operators/PressureSingle.o \
Operators/PutObjectsOnGrid.o \
Poisson/AMRSolver.o \
Poisson/Base.o \
Shape.o \
SimulationData.o \
Utils/BufferedLogger.o \

NVCC ?= nvcc
NVCCFLAGS ?= -code=sm_60 -arch=compute_60
ifeq ("$(gpu)", "true")
	OBJECTS += ExpAMRSolver.o BiCGSTAB.o LocalSpMatDnVec.o
	CPPFLAGS += -fopenmp -DGPU_POISSON -Wno-shadow -Wno-undef -Wno-float-equal -Wno-redundant-decls
	NVCCFLAGS += -std=c++17 -O3 --use_fast_math -Xcompiler "$(CPPFLAGS)" -DGPU_POISSON
	LIBS += -lcudart -lcublas -lcusparse
	ifeq ("$(profile)", "true")
		NVCCFLAGS += -DBICGSTAB_PROFILER
	endif
else
  CPPFLAGS += -Wno-unknown-pragmas
endif

all: debugRL simulation libcup.a
.DEFAULT: all
debugRL: debugRL.o $(OBJECTS)
	$(CXX) debugRL.o $(OBJECTS) $(LIBS) -o $@

simulation: main.o $(OBJECTS)
	$(CXX) main.o $(OBJECTS) $(LIBS) -o $@
libcup.a: $(OBJECTS)
	ar rcs $@ $(OBJECTS)

%.o: %.cu
	$(NVCC) -ccbin=$(CXX) $(NVCCFLAGS) -c $< -o $@
%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

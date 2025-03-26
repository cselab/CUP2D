bs = 8
gpu = false
NVCC = nvcc
LINK = $(CXX)

CPPFLAGS = \
-D_BS_=$(bs) \
-DCUBISM_ALIGNMENT=32 \
-D_DOUBLE_PRECISION_ \
-DDIMENSION=2 \
-I.

OBJECTS = \
Simulation.o \
Cubism/ArgumentParser.o \
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

ifeq ("$(gpu)", "true")
	CPPFLAGS += -DGPU_POISSON
	NVCCFLAGS += -std=c++17 -O3 --use_fast_math -DGPU_POISSON
	OBJECTS += \
Poisson/BiCGSTAB.o \
Poisson/ExpAMRSolver.o \
Poisson/LocalSpMatDnVec.o \

endif

all: debugRL simulation libcup.a
.DEFAULT: all
debugRL: debugRL.o $(OBJECTS)
	$(LINK) debugRL.o $(OBJECTS) $(LIBS) -o $@

simulation: main.o $(OBJECTS)
	$(LINK) main.o $(OBJECTS) $(LIBS) -o $@
libcup.a: $(OBJECTS)
	ar rcs $@ $(OBJECTS)

%.o: %.cu
	$(NVCC) -ccbin=$(CXX) $(NVCCFLAGS) -c $< -o $@
%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

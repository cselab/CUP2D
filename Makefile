gpu = false
MPICXX = mpicxx
NVCC = nvcc
FLAGS = \
-D_BS_=8 \
-DCUBISM_ALIGNMENT=32 \
-DDIMENSION=2 \
-D_DOUBLE_PRECISION_ \
-I. \

ifeq ($(gpu), true)
	LINK = $(NVCC)  -ccbin='$(MPICXX)'
	L = -Xcompiler '$(LIBS)' -Xcompiler -fopenmp -lcublas -lcusparse
	FLAGS += -DGPU_POISSON
	NVCCFLAGS = -std=c++17 -O3 --use_fast_math
	C = \
Poisson/BiCGSTAB.o \
Poisson/ExpAMRSolver.o \
Poisson/LocalSpMatDnVec.o \

else
	LINK = $(MPICXX)
	L = $(LIBS) -fopenmp
endif

O = \
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

main: main.o $O $C
	$(LINK) -o $@ main.o $O $C $L
%.o: %.cu
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler '$(FLAGS) $(CXXFLAGS)' -c $<
%.o: %.cpp
	$(MPICXX) $(FLAGS) -fopenmp $(CXXFLAGS) -c $< -o $@

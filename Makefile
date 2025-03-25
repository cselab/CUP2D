config ?= production
precision ?= double
bs ?= 8
gpu ?= false
openmp ?= false
profile ?= false
onetbb ?= false
symmetry ?= false
cylinder_ref ?= false

CPPFLAGS+= -std=c++17 -Wall -g
CPPFLAGS+= -Wextra -Wfloat-equal -Wcast-align -Woverloaded-virtual
CPPFLAGS+= -Wlogical-op -Wmissing-declarations -Wredundant-decls -Wshadow
CPPFLAGS+= -Wwrite-strings -Wno-unused-parameter
CPPFLAGS+= -Wno-float-equal
CPPFLAGS+= -Wno-redundant-decls

ifeq "$(openmp)" "true"
	CPPFLAGS+= -fopenmp
endif

LIBS+= -fopenmp

ifeq "$(onetbb)" "true"
	LIBS     += -L$(ONETBBROOT)/lib64 -ltbb
	CPPFLAGS += -I$(ONETBBROOT)/include
	CPPFLAGS += -DCUBISM_USE_ONETBB
endif


ifeq "$(findstring prod,$(config))" ""
	CPPFLAGS+= -O0
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
endif

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
		Simulation.o SimulationData.o BufferedLogger.o Helpers.o ArgumentParser.o \
		PressureSingle.o PutObjectsOnGrid.o advDiff.o ComputeForces.o\
		AdaptTheMesh.o AMRSolver.o Shape.o ShapeLibrary.o ShapesSimple.o \
		Fish.o FishData.o SmartCylinder.o StefanFish.o CarlingFish.o  \
		Naca.o CStartFish.o ZebraFish.o NeuroKinematicFish.o  Windmill.o \
		Waterturbine.o Teardrop.o ExperimentFish.o Base.o Forcing.o advDiffSGS.o CylinderNozzle.o \
		SmartNaca.o

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

all: debugRL simulation libcup.a cup.cflags.txt cup.libs.txt
.DEFAULT: all
debugRL: debugRL.o $(OBJECTS)
	$(CXX) debugRL.o $(OBJECTS) $(LIBS) -o $@

simulation: main.o $(OBJECTS)
	$(CXX) main.o $(OBJECTS) $(LIBS) -o $@
libcup.a: $(OBJECTS)
	ar rcs $@ $(OBJECTS)

%.o: %.cu
	$(NVCC) -ccbin=$(CXX) $(NVCCFLAGS) -c $< -o $@
%.d: %.cu
	$(NVCC) -ccbin=$(CXX) $(NVCCFLAGS) -c -MD $<
%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@
%.d: %.cpp
	$(CXX) $(CPPFLAGS) -c -MD $<

clean:
	rm -f debugRL simulation libcup.a cup.cflags.txt cup.libs.txt
	rm -f *.o *.d

//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "Simulation.h"

#include <Cubism/HDF5Dumper.h>

#include "Operators/Helpers.h"
#include "Operators/PressureSingle.h"
#include "Operators/PutObjectsOnGrid.h"
#include "Operators/ComputeForces.h"
#include "Operators/advDiff.h"
#include "Operators/advDiffSGS.h"
#include "Operators/AdaptTheMesh.h"
#include "Operators/Forcing.h"

#include "Utils/FactoryFileLineParser.h"
#include "Utils/StackTrace.h"

#include "Obstacles/ShapesSimple.h"
#include "Obstacles/CarlingFish.h"
#include "Obstacles/StefanFish.h"
#include "Obstacles/CStartFish.h"
#include "Obstacles/ZebraFish.h"
#include "Obstacles/NeuroKinematicFish.h"
#include "Obstacles/SmartCylinder.h"
#include "Obstacles/Naca.h"
#include "Obstacles/Windmill.h"
#include "Obstacles/Teardrop.h"
#include "Obstacles/Waterturbine.h"
#include "Obstacles/ExperimentFish.h"

#include <algorithm>
#include <iterator>

// to test reward function of windmill
// #include <random>

using namespace cubism;

BCflag cubismBCX;
BCflag cubismBCY;

static const char kHorLine[] = 
    "=======================================================================\n";

static inline std::vector<std::string> split(const std::string&s,const char dlm)
{
  std::stringstream ss(s); std::string item; std::vector<std::string> tokens;
  while (std::getline(ss, item, dlm)) tokens.push_back(item);
  return tokens;
}

Simulation::Simulation(int argc, char ** argv, MPI_Comm comm) : parser(argc,argv)
{
  enableStackTraceSignalHandling();
  sim.comm = comm;
  int size;
  MPI_Comm_size(sim.comm,&size);
  MPI_Comm_rank(sim.comm,&sim.rank);
  if (sim.rank == 0)
  {
    std::cout <<"=======================================================================\n";
    std::cout <<"    CubismUP 2D (velocity-pressure 2D incompressible Navier-Stokes)    \n";
    std::cout <<"=======================================================================\n";
    parser.print_args();
    #pragma omp parallel
    {
      int numThreads = omp_get_num_threads();
      #pragma omp master
      printf("[CUP2D] Running with %d rank(s) and %d thread(s).\n", size, numThreads);
    }
  }
}

Simulation::~Simulation() = default;

void Simulation::insertOperator(std::shared_ptr<Operator> op)
{
  pipeline.push_back(std::move(op));
}
void Simulation::insertOperatorAfter(
    std::shared_ptr<Operator> op, const std::string &name)
{
  for (size_t i = 0; i < pipeline.size(); ++i) {
    if (pipeline[i]->getName() == name) {
      pipeline.insert(pipeline.begin() + i + 1, std::move(op));
      return;
    }
  }
  std::string msg;
  msg.reserve(300);
  msg += "operator '";
  msg += name;
  msg += "' not found, available: ";
  for (size_t i = 0; i < pipeline.size(); ++i) {
    if (i > 0)
      msg += ", ";
    msg += pipeline[i]->getName();
  }
  msg += " (ensure that init() is called before inserting custom operators)";
  throw std::runtime_error(std::move(msg));
}

void Simulation::init()
{
  // parse field variables
  if ( sim.rank == 0 && sim.verbose )
    std::cout << "[CUP2D] Parsing Simulation Configuration..." << std::endl;
  parseRuntime();
  // allocate the grid
  if( sim.rank == 0 && sim.verbose )
    std::cout << "[CUP2D] Allocating Grid..." << std::endl;
  sim.allocateGrid();
  // create shapes
  if( sim.rank == 0 && sim.verbose )
    std::cout << "[CUP2D] Creating Shapes..." << std::endl;
  createShapes();
  // impose field initial condition
  if( sim.rank == 0 && sim.verbose )
    std::cout << "[CUP2D] Imposing Initial Conditions..." << std::endl;
  if( sim.ic == "random" )
  {
    randomIC ic(sim);
    ic(0);
  }
  else
  {
    IC ic(sim);
    ic(0);
  }
  // create compute pipeline
  if( sim.rank == 0 && sim.verbose )
    std::cout << "[CUP2D] Creating Computational Pipeline..." << std::endl;

  if( sim.smagorinskyCoeff == 0 )
    pipeline.push_back(std::make_shared<advDiff>(sim));
  else
    pipeline.push_back(std::make_shared<advDiffSGS>(sim));
  if( sim.bForcing )
    pipeline.push_back(std::make_shared<Forcing>(sim));
  pipeline.push_back(std::make_shared<PressureSingle>(sim));
  pipeline.push_back(std::make_shared<ComputeForces>(sim));
  pipeline.push_back(std::make_shared<AdaptTheMesh>(sim));
  pipeline.push_back(std::make_shared<PutObjectsOnGrid>(sim));

  if( sim.rank == 0 && sim.verbose )
  {
    std::cout << "[CUP2D] Operator ordering:\n";
    for (size_t c=0; c<pipeline.size(); c++)
      std::cout << "[CUP2D] - " << pipeline[c]->getName() << "\n";
  }

  // Put Object on Intially defined Mesh and impose obstacle velocities
  startObstacles();
}

void Simulation::parseRuntime()
{
  // restart the simulation?
  sim.bRestart = parser("-restart").asBool(false);

  /* parameters that have to be given */
  /************************************/
  parser.set_strict_mode();

  // set initial number of blocks
  sim.bpdx = parser("-bpdx").asInt();
  sim.bpdy = parser("-bpdy").asInt();

  // maximal number of refinement levels
  sim.levelMax = parser("-levelMax").asInt();

  // refinement/compression tolerance for vorticity magnitude
  sim.Rtol = parser("-Rtol").asDouble(); 
  sim.Ctol = parser("-Ctol").asDouble();

  parser.unset_strict_mode();
  /************************************/
  /************************************/

  // refiment according to Qcriterion instead of |omega|
  sim.Qcriterion = parser("-Qcriterion").asBool(false);

  // check for refinement every this many timesteps
  sim.AdaptSteps = parser("-AdaptSteps").asInt(20);

  // boolean to switch between refinement according to chi or grad(chi)
  sim.bAdaptChiGradient = parser("-bAdaptChiGradient").asInt(1);

  // initial level of refinement
  sim.levelStart = parser("-levelStart").asInt(-1);
  if (sim.levelStart == -1) sim.levelStart = sim.levelMax - 1;

  // simulation extent
  sim.extent = parser("-extent").asDouble(1);

  // timestep / CFL number
  sim.dt = parser("-dt").asDouble(0);
  sim.CFL = parser("-CFL").asDouble(0.2);
  sim.rampup = parser("-rampup").asInt(0);

  // simulation ending parameters
  sim.nsteps = parser("-nsteps").asInt(0);
  sim.endTime = parser("-tend").asDouble(0);

  // penalisation coefficient
  sim.lambda = parser("-lambda").asDouble(1e7);

  // constant for explicit penalisation lambda=dlm/dt
  sim.dlm = parser("-dlm").asDouble(0);

  // kinematic viscocity
  sim.nu = parser("-nu").asDouble(1e-2);

  // forcing
  sim.bForcing = parser("-bForcing").asInt(0);
  sim.forcingWavenumber = parser("-forcingWavenumber").asDouble(4);
  sim.forcingCoefficient = parser("-forcingCoefficient").asDouble(4);

  // Smagorinsky Model
  sim.smagorinskyCoeff = parser("-smagorinskyCoeff").asDouble(0);
  sim.bDumpCs = parser("-dumpCs").asInt(0);

  // Flag for initial condition
  sim.ic = parser("-ic").asString("");

  // Boundary conditions (freespace or periodic)
  std::string BC_x = parser("-BC_x").asString("freespace");
  std::string BC_y = parser("-BC_y").asString("freespace");
  cubismBCX = string2BCflag(BC_x);
  cubismBCY = string2BCflag(BC_y);

  // poisson solver parameters
  sim.poissonSolver = parser("-poissonSolver").asString("iterative");
  sim.PoissonTol = parser("-poissonTol").asDouble(1e-6);
  sim.PoissonTolRel = parser("-poissonTolRel").asDouble(0);
  sim.maxPoissonRestarts = parser("-maxPoissonRestarts").asInt(30);
  sim.maxPoissonIterations = parser("-maxPoissonIterations").asInt(1000);
  sim.bMeanConstraint = parser("-bMeanConstraint").asInt(0);
  sim.GuessDpDt = parser("-GuessDpDt").asBool(false);

  // output parameters
  sim.profilerFreq = parser("-profilerFreq").asInt(0);
  sim.dumpFreq = parser("-fdump").asInt(0);
  sim.dumpTime = parser("-tdump").asDouble(0);
  sim.path2file = parser("-file").asString("./");
  sim.path4serialization = parser("-serialization").asString(sim.path2file);
  sim.verbose = parser("-verbose").asInt(1);
  sim.muteAll = parser("-muteAll").asInt(0);
  sim.DumpUniform = parser("-DumpUniform").asBool(false);
  if(sim.muteAll) sim.verbose = 0;
}

void Simulation::createShapes()
{
  const std::string shapeArg = parser("-shapes").asString("");
  std::stringstream descriptors( shapeArg );
  std::string lines;

  while (std::getline(descriptors, lines))
  {
    std::replace(lines.begin(), lines.end(), '_', ' ');
    const std::vector<std::string> vlines = split(lines, ',');

    for (const auto& line: vlines)
    {
      std::istringstream line_stream(line);
      std::string objectName;
      if( sim.rank == 0 && sim.verbose )
        std::cout << "[CUP2D] " << line << std::endl;
      line_stream >> objectName;
      // Comments and empty lines ignored:
      if(objectName.empty() or objectName[0]=='#') continue;
      FactoryFileLineParser ffparser(line_stream);
      Real center[2] = {
        ffparser("-xpos").asDouble(.5*sim.extents[0]),
        ffparser("-ypos").asDouble(.5*sim.extents[1])
      };
      //ffparser.print_args();
      Shape* shape = nullptr;
      if (objectName=="disk")
        shape = new Disk(             sim, ffparser, center);
      else if (objectName=="smartDisk")
        shape = new SmartCylinder(    sim, ffparser, center);
      else if (objectName=="halfDisk")
        shape = new HalfDisk(         sim, ffparser, center);
      else if (objectName=="ellipse")
        shape = new Ellipse(          sim, ffparser, center);
      else if (objectName=="rectangle")
        shape = new Rectangle(        sim, ffparser, center);
      else if (objectName=="stefanfish")
        shape = new StefanFish(       sim, ffparser, center);
      else if (objectName=="cstartfish")
        shape = new CStartFish(       sim, ffparser, center);
      else if (objectName=="zebrafish")
          shape = new ZebraFish(      sim, ffparser, center);
      else if (objectName=="neurokinematicfish")
          shape = new NeuroKinematicFish(      sim, ffparser, center);
      else if (objectName=="carlingfish")
        shape = new CarlingFish(      sim, ffparser, center);
      else if ( objectName=="NACA" )
        shape = new Naca(             sim, ffparser, center);
      else if (objectName=="windmill")
        shape = new Windmill(         sim, ffparser, center);
      else if (objectName=="teardrop")
        shape = new Teardrop(         sim, ffparser, center);
      else if (objectName=="waterturbine")
        shape = new Waterturbine(     sim, ffparser, center);
      else if (objectName=="experimentFish")
        shape = new ExperimentFish(   sim, ffparser, center);
      else
        throw std::invalid_argument("unrecognized shape: " + objectName);
      sim.addShape(std::shared_ptr<Shape>{shape});
    }
  }

  if( sim.shapes.size() ==  0 && sim.rank == 0)
    std::cout << "Did not create any obstacles." << std::endl;
}

void Simulation::reset()
{
  // reset field variables and shapes
  if( sim.rank == 0 && sim.verbose )
    std::cout << "[CUP2D] Resetting Simulation..." << std::endl;
  sim.resetAll();
  // impose field initial condition
  if( sim.rank == 0 && sim.verbose )
    std::cout << "[CUP2D] Imposing Initial Conditions..." << std::endl;
  IC ic(sim);
  ic(0);
  // Put Object on Intially defined Mesh and impose obstacle velocities
  startObstacles();
}

void Simulation::resetRL()
{
  // reset simulation (not shape)
  if( sim.rank == 0 && sim.verbose )
    std::cout << "[CUP2D] Resetting Simulation..." << std::endl;
  sim.resetAll();
  // impose field initial condition
  if( sim.rank == 0 && sim.verbose )
    std::cout << "[CUP2D] Imposing Initial Conditions..." << std::endl;
  IC ic(sim);
  ic(0);
}

void Simulation::startObstacles()
{
  Checker check (sim);

  // put obstacles to grid and compress
  if( sim.rank == 0 && sim.verbose && !sim.bRestart)
    std::cout << "[CUP2D] Initial PutObjectsOnGrid and Compression of Grid\n";
  PutObjectsOnGrid * const putObjectsOnGrid = findOperator<PutObjectsOnGrid>();
  AdaptTheMesh * const adaptTheMesh = findOperator<AdaptTheMesh>();
  assert(putObjectsOnGrid != nullptr && adaptTheMesh != nullptr);
  if( not sim.bRestart )
  for( int i = 0; i<sim.levelMax; i++ )
  {
    (*putObjectsOnGrid)(0.0);
    (*adaptTheMesh)(0.0);
  }
  (*putObjectsOnGrid)(0.0);

  // impose velocity of obstacles
  if( not sim.bRestart )
  {
    if( sim.rank == 0 && sim.verbose )
      std::cout << "[CUP2D] Imposing Initial Velocity of Objects on field\n";
    ApplyObjVel initVel(sim);
    initVel(0);
  }
}

void Simulation::simulate() {
  if (sim.rank == 0 && !sim.muteAll)
    std::cout << kHorLine << "[CUP2D] Starting Simulation...\n" << std::flush;

  while (1)
	{
    Real dt = calcMaxTimestep();

    bool done = false;

    // Ignore the final time step if `dt` is way too small.
    if (!done || dt > 2e-16)
      advance(dt);

    if (!done)
      done = sim.bOver();

    if (sim.rank == 0 && sim.profilerFreq > 0 && sim.step % sim.profilerFreq == 0)
      sim.printResetProfiler();

    if (done)
    {
      const bool bDump = sim.bDump();
      if( bDump ) {
        if( sim.rank == 0 && sim.verbose )
          std::cout << "[CUP2D] dumping field...\n";
        sim.registerDump();
        sim.dumpAll("avemaria_");
      }
      if (sim.rank == 0 && !sim.muteAll)
      {
        std::cout << kHorLine << "[CUP2D] Simulation Over... Profiling information:\n";
        sim.printResetProfiler();
        std::cout << kHorLine;
      }
      break;
    }
  }
}

Real Simulation::calcMaxTimestep()
{
  sim.dt_old2 = sim.dt_old;
  sim.dt_old = sim.dt;
  Real CFL = sim.CFL;
  const Real h = sim.getH();
  const auto findMaxU_op = findMaxU(sim);
  sim.uMax_measured = findMaxU_op.run();

  if( CFL > 0 )
  {
    const Real dtDiffusion = 0.25*h*h/(sim.nu+0.25*h*sim.uMax_measured);
    const Real dtAdvection = h / ( sim.uMax_measured + 1e-8 );
    
    //non-constant timestep introduces a source term = (1-dt_new/dt_old) \nabla^2 P_{old}
    //in the Poisson equation. Thus, we try to modify the timestep less often
    if (sim.step < sim.rampup)
    {
      const Real x = (sim.step + 1.0)/sim.rampup;
      const Real rampupFactor = std::exp(std::log(1e-3)*(1-x));
      sim.dt = rampupFactor*std::min({ dtDiffusion, CFL * dtAdvection});
    }
    else
    {
      sim.dt = std::min({ dtDiffusion, CFL * dtAdvection});
    }
  }

  if( sim.dt <= 0 ){
    std::cout << "[CUP2D] dt <= 0. Aborting..." << std::endl;
    fflush(0);
    abort();
  }

  if(sim.dlm > 0) sim.lambda = sim.dlm / sim.dt;
  return sim.dt;
}

void Simulation::advance(const Real dt)
{

  const Real CFL = ( sim.uMax_measured + 1e-8 ) * sim.dt / sim.getH();
  if (sim.rank == 0 && !sim.muteAll)
  {
    std::cout << kHorLine;
    printf("[CUP2D] step:%d, blocks:%zu, time:%f, dt=%f, uinf:[%f %f], maxU:%f, CFL:%f\n",
           sim.step, sim.chi->getBlocksInfo().size(),
           (double)sim.time, (double)dt,
           (double)sim.uinfx, (double)sim.uinfy, (double)sim.uMax_measured, (double)CFL);
  }

  // dump field
  const bool bDump = sim.bDump();
  if( bDump ) {
    if( sim.rank == 0 && sim.verbose )
      std::cout << "[CUP2D] dumping field...\n";
    sim.registerDump();
    sim.dumpAll("avemaria_");
  }

  for (size_t c=0; c<pipeline.size(); c++) {
    if( sim.rank == 0 && sim.verbose )
      std::cout << "[CUP2D] running " << pipeline[c]->getName() << "...\n";
    (*pipeline[c])(dt);
  }
  sim.time += dt;
  sim.step++;
}

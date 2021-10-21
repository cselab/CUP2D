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
#include "Operators/AdaptTheMesh.h"

#include "Utils/FactoryFileLineParser.h"

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

#include <algorithm>
#include <iterator>

// to test reward function of windmill
// #include <random>

using namespace cubism;

static inline std::vector<std::string> split(const std::string&s,const char dlm)
{
  std::stringstream ss(s); std::string item; std::vector<std::string> tokens;
  while (std::getline(ss, item, dlm)) tokens.push_back(item);
  return tokens;
}

Simulation::Simulation(int argc, char ** argv) : parser(argc,argv)
{
  int size;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&sim.rank);
  if (sim.rank == 0)
  {
    std::cout <<"=======================================================================\n";
    std::cout <<"    CubismUP 2D (velocity-pressure 2D incompressible Navier-Stokes)    \n";
    std::cout <<"=======================================================================\n";
    parser.print_args();
    std::cout <<"[CUP2D] Running with " << size << " ranks.\n";
  }


}

Simulation::~Simulation()
{
  while( not pipeline.empty() ) {
    Operator * g = pipeline.back();
    pipeline.pop_back();
    if(g not_eq nullptr) delete g;
  }
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
  IC ic(sim);
  ic(0);
  // create compute pipeline
  if( sim.rank == 0 && sim.verbose )
    std::cout << "[CUP2D] Creating Computational Pipeline..." << std::endl;

  pipeline.push_back( new advDiff(sim) );
  pipeline.push_back( new PressureSingle(sim) );
  pipeline.push_back( new ComputeForces(sim) );
  pipeline.push_back( new AdaptTheMesh(sim) );
  pipeline.push_back( new PutObjectsOnGrid(sim) );

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

  // refinement/compression tolerance for voriticy magnitude
  sim.Rtol = parser("-Rtol").asDouble(); 
  sim.Ctol = parser("-Ctol").asDouble();

  parser.unset_strict_mode();
  /************************************/
  /************************************/

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

  // simulation ending parameters
  sim.nsteps = parser("-nsteps").asInt(0);
  sim.endTime = parser("-tend").asDouble(0);

  // penalisation coefficient
  sim.lambda = parser("-lambda").asDouble(1e7);

  // constant for explicit penalisation lambda=dlm/dt
  sim.dlm = parser("-dlm").asDouble(0);

  // kinematic viscocity
  sim.nu = parser("-nu").asDouble(1e-2);

  // poisson solver parameters
  sim.PoissonTol = parser("-poissonTol").asDouble(1e-6);
  sim.PoissonTolRel = parser("-poissonTolRel").asDouble(1e-4);
  sim.maxPoissonRestarts = parser("-maxPoissonRestarts").asInt(30);

  // output parameters
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
  unsigned k = 0;

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
      double center[2] = {
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
      assert(shape not_eq nullptr);
      shape->obstacleID = k++;
      sim.shapes.push_back(shape);
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
  if( sim.rank == 0 && sim.verbose )
    std::cout << "[CUP2D] Initial PutObjectsOnGrid and Compression of Grid\n";
  for( int i = 0; i<sim.levelMax; i++ )
  {
    // PutObjectsOnGrid
    (*pipeline[pipeline.size()-1])(0);

    // AdaptTheMesh
    (*pipeline[pipeline.size()-2])(0);
  }
  // PutObjectsOnGrid
   (*pipeline[pipeline.size()-1])(0);
  // impose velocity of obstacles
  if( sim.rank == 0 && sim.verbose )
    std::cout << "[CUP2D] Imposing Initial Velocity of Objects on field\n";
  ApplyObjVel initVel(sim);
  initVel(0); }

void Simulation::simulate() { if(sim.rank == 0) { std::cout
	<<"=======================================================================\n";
	std::cout << "[CUP2D] Starting Simulation..." << std::endl; } while (1)
	{
    // sim.startProfiler("DT");
    const double dt = calcMaxTimestep();
    // sim.stopProfiler();
    if (advance(dt)) break; } }

double Simulation::calcMaxTimestep()
{
  sim.dt_old = sim.dt;
  double CFL = sim.CFL;
  const double h = sim.getH();
  const auto findMaxU_op = findMaxU(sim);
  sim.uMax_measured = findMaxU_op.run();

  if( CFL > 0 )
  {
    const double dtDiffusion = 0.25*h*h/(sim.nu+0.125*h*sim.uMax_measured);
    const double dtAdvection = h / ( sim.uMax_measured + 1e-8 );
    // ramp up CFL
    const int rampup = 100;
    if (sim.step < rampup)
    {
      const double x = (sim.step + 1.0)/rampup;
      const double rampupFactor = std::exp(std::log(1e-3)*(1-x));
      sim.dt = rampupFactor*std::min({ dtDiffusion, CFL * dtAdvection});
    }
    else //if (sim.time < 0.5)
    {
      sim.dt = std::min({dtDiffusion, CFL * dtAdvection});
    }
    //else 
    //{
    //  const double CFL_ramp = (sim.time < 1.5) ? (0.1-sim.CFL)*(sim.time-0.5) + sim.CFL : 0.1;
    //  sim.dt = std::min({dtDiffusion, CFL_ramp * dtAdvection});
    //}
  }

  if( sim.dt <= 0 ){
    std::cout << "[CUP2D] dt <= 0. Aborting..." << std::endl;
    fflush(0);
    abort();
  }

  if(sim.dlm > 0) sim.lambda = sim.dlm / sim.dt;
  return sim.dt;
}

bool Simulation::advance(const double dt)
{
  MPI_Barrier(MPI_COMM_WORLD);

  const double CFL = ( sim.uMax_measured + 1e-8 ) * sim.dt / sim.getH();
  if (sim.rank == 0)
  {
    std::cout
  <<"=======================================================================\n";
    printf("[CUP2D] step:%d, time:%f, dt=%f, uinf:[%f %f], maxU:%f, CFL:%f\n",
      sim.step, sim.time, dt, sim.uinfx, sim.uinfy, sim.uMax_measured, CFL); 
  }

  assert(dt>2.2e-16);
  if( sim.step == 0 ){
    if ( sim.rank == 0 && sim.verbose )
      std::cout << "[CUP2D] dumping IC...\n";
    sim.dumpAll("IC");
  }

  const bool bDump = sim.bDump();

  // For debuging state function LEAD-FOLLOW
  // Shape *object = getShapes()[0];
  // StefanFish *agent = dynamic_cast<StefanFish *>(getShapes()[1]);
  // auto state = agent->state(object);
  // std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;
  // std::cout << "[CUP2D] Computed state:" << std::endl;
  // for( auto s : state )
  //   std::cout << s << std::endl;
  // std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;

  // // For debuging state function SMARTCYLINDER
  // SmartCylinder *agent = dynamic_cast<SmartCylinder *>(getShapes()[0]);
  // std::vector<double> target{0.8,0.5};
  // auto state = agent->state( target );
  // std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;
  // std::cout << "[CUP2D] Computed state:" << std::endl;
  // for( auto s : state )
  //   std::cout << s << std::endl;
  // std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;


  // // For debugging reward function for windmills
  // Windmill *agent = dynamic_cast<Windmill *>(getShapes()[1]);
  // // J = 2.9e-6
  // double upper_bound_act = 1e-5;
  // std::uniform_real_distribution dist(0.0, upper_bound_act);
  // std::default_random_engine rd;
  // agent->act(dist(rd)); // put torque on the windmill
  // std::array<Real, 2> target{0.4,0.5};
  // std::vector<double> target_vel{0.0,0.0};
  // double C = 1e8;
  // double r = agent->reward(target, target_vel, C);

  // std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;
  // std::cout << "[CUP2D] Computed reward:" << std::endl;
  // std::cout << "Total reward: " << r << std::endl;
  // std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;

  for (size_t c=0; c<pipeline.size(); c++) {
    //if (pipeline.size()-2 == c) continue;
    if( sim.rank == 0 && sim.verbose )
      std::cout << "[CUP2D] running " << pipeline[c]->getName() << "...\n";
    (*pipeline[c])(dt);
    //sim.dumpAll( pipeline[c]->getName() );
  }

  sim.time += dt;
  sim.step++;


  // dump field
  if( bDump ) {
    if( sim.rank == 0 && sim.verbose )
      std::cout << "[CUP2D] dumping field...\n";
    sim.registerDump();
    sim.dumpAll("avemaria_"); 
  }

  const bool bOver = sim.bOver();

  if (bOver && sim.rank == 0 ){
    std::cout
  <<"=======================================================================\n";
    std::cout << "[CUP2D] Simulation Over... Profiling information:\n";
    sim.printResetProfiler();
    std::cout
  <<"=======================================================================\n";
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return bOver;
}

//
//  Sim_FSI_Gravity.cpp
//  CubismUP_2D
//
//  Created by Christian Conti on 1/26/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#include "Simulation.h"

#ifdef USE_VTK
#include <SerializerIO_ImageVTK.h>
#else
#define _USE_HDF_
#include <HDF5Dumper.h>
#endif
//#include <ZBinDumper.h>

#include "HelperOperators.h"
#include "CoordinatorIC.h"
#include "CoordinatorAdvection.h"
#include "CoordinatorMultistep.h"
#include "CoordinatorDiffusion.h"
#include "CoordinatorShape.h"
#include "CoordinatorPressure.h"
#include "CoordinatorGravity.h"
#include "FactoryFileLineParser.h"

#include "ShapesSimple.h"
#include "BlowFish.h"
#include "StefanFish.h"

#include <regex>
#include <algorithm>
#include <iterator>

static inline vector<string> splitter(string& content) {
    vector<string> split_content;
    const regex pattern(R"(\\n)");
    copy( sregex_token_iterator(content.begin(), content.end(), pattern, -1),
    sregex_token_iterator(), back_inserter(split_content) );
    return split_content;
}

void Simulation::dump(string fname) {
  stringstream ss;
  ss<<"avemaria_"<<fname<<std::setfill('0')<<std::setw(7)<<sim.step;
  #ifdef USE_VTK
    SerializerIO_ImageVTK<FluidGrid, FluidVTKStreamer> dumper;
    dumper.Write( *(sim.grid), sim.path4serialization + ss.str() + ".vti" );
  #else
    DumpHDF5<FluidGrid,StreamerVelocityVector>(*(sim.grid), sim.step, sim.time,
      StreamerVelocityVector::prefix() + ss.str(), sim.path4serialization);
    //if(sim.bVariableDensity)
      DumpHDF5<FluidGrid,StreamerPressure>(*(sim.grid), sim.step, sim.time,
      StreamerPressure::prefix() + ss.str(), sim.path4serialization);
    if(sim.bVariableDensity)
      DumpHDF5<FluidGrid,StreamerRho>(*(sim.grid), sim.step, sim.time,
      StreamerRho::prefix() + ss.str(), sim.path4serialization);
  #endif
  //DumpZBin<FluidGrid, StreamerSerialization>(*grid, serializedGrid.str(), path4serialization);
  //ReadZBin<FluidGrid, StreamerSerialization>(*grid, serializedGrid.str(), path4serialization);
}

Simulation::Simulation(int argc, char ** argv) : parser(argc,argv)
{
  cout << "=================================================================\n";
  cout << "\t\tFlow past a falling obstacle\n";
  cout << "=================================================================\n";
}

Simulation::~Simulation() {
  while( not pipeline.empty() ) {
    GenericCoordinator * g = pipeline.back();
    pipeline.pop_back();
    if(g not_eq nullptr) delete g;
  }
}

void Simulation::parseRuntime() {
  sim.bRestart = parser("-restart").asBool(false);
  cout << "bRestart is " << sim.bRestart << endl;

  // initialize grid
  parser.set_strict_mode();
  const int bpdx = parser("-bpdx").asInt();
  const int bpdy = parser("-bpdy").asInt();
  sim.grid = new FluidGrid(bpdx, bpdy, 1);
  assert( sim.grid not_eq nullptr );

  // simulation ending parameters
  parser.unset_strict_mode();
  sim.nsteps = parser("-nsteps").asInt(0);
  sim.endTime = parser("-tend").asDouble(0);

  // output parameters
  sim.dumpFreq = parser("-fdump").asDouble(0);
  sim.dumpTime = parser("-tdump").asDouble(0);

  sim.path2file = parser("-file").asString("./");
  sim.path4serialization = parser("-serialization").asString(sim.path2file);
  sim.bFreeSpace = parser("-bFreeSpace").asInt(1);

  // simulation settings
  sim.CFL = parser("-CFL").asDouble(.1);
  sim.lambda = parser("-lambda").asDouble(1e5);
  sim.dlm = parser("-dlm").asDouble(10.);
  sim.nu = parser("-nu").asDouble(1e-2);

  sim.verbose = parser("-verbose").asInt(1);
  sim.muteAll = parser("-muteAll").asInt(0);//stronger silence, not even files
  if(sim.muteAll) sim.verbose = 0;
}

void Simulation::createShapes() {
  parser.set_strict_mode();
  const Real axX = parser("-bpdx").asInt();
  const Real axY = parser("-bpdy").asInt();
  const Real ext = std::max(axX, axY);
  parser.unset_strict_mode();
  const string shapeArg = parser("-shapes").asString("");
  stringstream descriptors( shapeArg );
  string lines;
  unsigned k = 0;

  while (std::getline(descriptors, lines))
  {
    std::replace(lines.begin(), lines.end(), '_', ' ');
    // Two options! Either we have list of lines each containing a description
    // of an obstacle (like factory files or CUP3D factory-descriptor)
    // Or we have an argument list that looks like -shapes foo\nbar. Splitter
    // will create a vector of strings, the first containing foo and the second
    // bar so that they can be parsed separately. Reason being that in many
    // situations \n will not be read as line escape but as backslash n.
    const vector<string> vlines = splitter(lines);
    for (const string line: vlines)
    {
      istringstream line_stream(line);
      string objectName;
      cout << line << endl;
      line_stream >> objectName;
      // Comments and empty lines ignored:
      if(objectName.empty() or objectName[0]=='#') continue;
      FactoryFileLineParser ffparser(line_stream);
      Real center[2] = {
          (Real) ffparser("-xpos").asDouble(.5*axX/ext),
          (Real) ffparser("-ypos").asDouble(.5*axY/ext)
      };
      //ffparser.print_args();
      Shape* shape = nullptr;
      if (objectName=="disk")
        shape = new Disk(             sim, ffparser, center);
      else if (objectName=="halfDisk")
        shape = new HalfDisk(         sim, ffparser, center);
      else if (objectName=="ellipse")
        shape = new Ellipse(          sim, ffparser, center);
      else if (objectName=="diskVarDensity")
        shape = new DiskVarDensity(   sim, ffparser, center);
      else if (objectName=="ellipseVarDensity")
        shape = new EllipseVarDensity(sim, ffparser, center);
      else if (objectName=="blowfish")
        shape = new BlowFish(         sim, ffparser, center);
      else if (objectName=="stefanfish")
        shape = new StefanFish(       sim, ffparser, center);
      else {
        cout << "Error - this shape is not recognized! Aborting now\n";
        abort();
      }
      assert(shape not_eq nullptr);
      shape->obstacleID = k++;
      sim.shapes.push_back(shape);
    }
  }

  if( sim.shapes.size() ==  0) {
    std::cout << "Did not create any obstacles. Not supported. Aborting!\n";
    abort();
  }

  //now that shapes are created, we know whether we need variable rho solver:
  sim.checkVariableDensity();
}

void Simulation::init() {
  parseRuntime();
  createShapes();

  // setup initial conditions
  CoordinatorIC coordIC(sim);
  profiler.push_start(coordIC.getName());
  coordIC(0);
  profiler.pop_stop();

  pipeline.clear();

  pipeline.push_back( new CoordinatorComputeShape(sim) );
  pipeline.push_back( new CoordinatorVelocities(sim) );
  pipeline.push_back( new CoordinatorPenalization(sim) );

  #if 1 // in one sweep advect, diffuse, add hydrostatic
    pipeline.push_back( new CoordinatorMultistep<Lab>(sim) );
  #else
    pipeline.push_back( new CoordinatorAdvection<Lab>(sim) );
    pipeline.push_back( new CoordinatorDiffusion<Lab>(sim) );
    pipeline.push_back( new CoordinatorGravity(sim) );
  #endif

  pipeline.push_back( new CoordinatorPressure<Lab>(sim) );
  pipeline.push_back( new CoordinatorComputeForces(sim) );
  if(not sim.bFreeSpace)
    pipeline.push_back( new CoordinatorFadeOut(sim) );

  cout << "Coordinator/Operator ordering:\n";
  for (size_t c=0; c<pipeline.size(); c++)
    cout << "\t" << pipeline[c]->getName() << endl;
}

void Simulation::simulate() {
  while (1) {
    profiler.push_start("DT");
    const double dt = calcMaxTimestep();
    profiler.pop_stop();
    if (advance(dt)) break;
  }
}

double Simulation::calcMaxTimestep() {
  const Real maxU = findMaxUOMP( sim ); assert(maxU>=0);
  const double h = sim.getH();
  const double dtFourier = h*h/sim.nu, dtCFL = maxU<2.2e-16? 1 : h/maxU;
  sim.dt = sim.CFL * std::min(dtCFL, dtFourier);

  const double maxUb = sim.maxRelSpeed();
  const double dtBody = maxUb<2.2e-16? 1 : h/maxUb;
  sim.dt = std::min( sim.dt, sim.CFL*dtBody );

  if(sim.dlm > 1) sim.lambda = sim.dlm / sim.dt;
  if (sim.step < 100) {
    const double x = (sim.step+1.0)/100;
    const double rampCFL = std::exp(std::log(1e-3)*(1-x) + std::log(sim.CFL)*x);
    sim.dt = rampCFL*std::min({dtCFL, dtFourier, dtBody});
  }
  #ifndef RL_TRAIN
  if(sim.verbose)
    cout << "step, time, dt "// (Fourier, CFL, body): "
    <<sim.step<<" "<<sim.time<<" "<<sim.dt
    //<<" "<<dtFourier<<" "<<dtCFL<<" "<<dtBody
    //<<" "<<sim.uinfx<<" "<<sim.uinfy
    <<endl;
  #endif

  return sim.dt;
}

bool Simulation::advance(const double dt) {
  assert(dt>2.2e-16);
  const bool bDump = sim.bDump();

  for (size_t c=0; c<pipeline.size(); c++) {
    profiler.push_start(pipeline[c]->getName());
    (*pipeline[c])(sim.dt);
    profiler.pop_stop();
    // stringstream ss;
    // ss << path2file << "avemaria_" << pipeline[c]->getName() << "_";
    // ss << std::setfill('0') << std::setw(7) << step << ".vti";
    //      cout << ss.str() << endl;
    //      dump(ss);
  }

  sim.time += sim.dt;
  sim.step++;

  //dump some time steps every now and then
  profiler.push_start("Dump");
  if(bDump) {
    sim.registerDump();
    dump();
  }
  profiler.pop_stop();

  if (sim.step % 100 == 0 && sim.verbose) {
    profiler.printSummary();
    profiler.reset();
  }

  const bool bOver = sim.bOver();
  if(bOver) profiler.printSummary();
  return bOver;
}

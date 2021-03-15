#include "Windmill.h"
#include "ShapeLibrary.h"
#include "../Utils/BufferedLogger.h"

using namespace cubism;

#define NSENSORS 8


// smart cyclinder stuff
// Windmill::Windmill(SimulationData& s, cubism::ArgumentParser& p, double C[2]):
// Shape(s,p,C), radius(p("-radius").asDouble(0.1)) 
// {}

void Windmill::create(const std::vector<BlockInfo>& vInfo)
{
  // windmill stuff
  const Real h =  vInfo[0].h_gridpoint;
  for(auto & entry : obstacleBlocks) delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock*> (vInfo.size(), nullptr);

  #pragma omp parallel
  {
    // FillBlocks_Ellipse(const Real _e0, const Real _e1, const Real h, const double C[2], Real ang, Real rho)
    // e0 and e1 are the semi-axes, don't know which is smallest

    // Use orientation (part of Shape class) for angle

    // in the case of the windmill we have 3 ellipses that are not centered at 0
    // ellipse 1 

    // center of ellipse 1 wrt to center of windmill,at T=0
    double center_orig1[2] = {smajax/2, -(smajax/2)*std::tan(M_PI/6)};
    // center of ellipse 1 wrt to origin
    double center1[2] = {center[0] + std::cos(orientation) * center_orig1[0] - std::sin(orientation)* center_orig1[1], 
                         center[1] + std::sin(orientation) * center_orig1[0] + std::cos(orientation) * center_orig1[1]};

    FillBlocks_Ellipse kernel1(smajax, sminax, h, center1, (orientation + 2*M_PI / 3), rhoS);

    // ellipse 2 

    // center of ellipse 2 wrt to center of windmill,at T=0
    double center_orig2[2] = {0, smajax/(2*std::cos(M_PI/6))};
    // center of ellipse 1 wrt to origin
    double center2[2] = {center[0] + std::cos(orientation) * center_orig2[0] - std::sin(orientation)* center_orig2[1], 
                         center[1] + std::sin(orientation) * center_orig2[0] + std::cos(orientation) * center_orig2[1]};

    FillBlocks_Ellipse kernel2(smajax, sminax, h, center2, (orientation + M_PI / 3), rhoS);

    // ellipse 3 

    // center of ellipse 3 wrt to center of windmill,at T=0
    double center_orig3[2] = {-smajax/2, -(smajax/2)*std::tan(M_PI/6)};
    // center of ellipse 1 wrt to origin
    double center3[2] = {center[0] + std::cos(orientation) * center_orig3[0] - std::sin(orientation)* center_orig3[1], 
                         center[1] + std::sin(orientation) * center_orig3[0] + std::cos(orientation) * center_orig3[1]};

    FillBlocks_Ellipse kernel3(smajax, sminax, h, center3, orientation, rhoS);

    #pragma omp for schedule(dynamic, 1)
    for(size_t i=0; i<vInfo.size(); i++)
    {
      if(kernel1.is_touching(vInfo[i]))
      {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0
      }
      else if(kernel2.is_touching(vInfo[i]))
      {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0
      }
      else if(kernel3.is_touching(vInfo[i]))
      {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0
      }

      ScalarBlock& B = *(ScalarBlock*)vInfo[i].ptrBlock;
      if(obstacleBlocks[vInfo[i].blockID] == nullptr) continue;
      kernel1(vInfo[i], B, * obstacleBlocks[vInfo[i].blockID]);
      kernel2(vInfo[i], B, * obstacleBlocks[vInfo[i].blockID]);
      kernel3(vInfo[i], B, * obstacleBlocks[vInfo[i].blockID]);
    }

  }

}

void Windmill::updateVelocity(double dt)
{
  Shape::updateVelocity(dt);
}

void Windmill::updatePosition(double dt)
{
  Shape::updatePosition(dt);
}

// action: change torque on windmill
void Windmill::act( double action )
{
  // if(action.size() != 1){
  //   std::cout << "One action required, torque applied to the object.";
  //   fflush(0);
  //   abort();
  // }
  // appliedTorque = action[0];
  appliedTorque = action;
}

// reward : how close flow at one point is to wanted flow - energy used 
double Windmill::reward( std::array<Real, 2> target, std::vector<double> target_vel, double C)
{
  // first reward is opposite of energy given into the system : r_1 = -torque*angVel*dt
  double r_energy = -appliedTorque*omega*sim.dt; 
  // other reward is diff between target and average of area : r_2^t = C/t\sum_0^t (u(x,y,t)-u^*(x,y,t))^2
  const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo();
  std::vector<double>  avg = average(target, velInfo);
  diff_flow += std::pow(target_vel[0] - avg[0], 2) +std::pow(target_vel[1] - avg[1], 2);
  double r_flow = (C / sim.time) * diff_flow;
  return r_energy + r_flow;
}


std::vector<double>  Windmill::state()
{
  // state of a windmill is the angle and angular velocity

  // intitialize state vector
  std::vector<double> state;

  // angle
  state[0] = orientation;

  // ang vel
  state[1] = omega;

  return state;
}

/* helpers to compute sensor information */

// average function
std::vector<double> Windmill::average(std::array<Real, 2> pSens, const std::vector<cubism::BlockInfo>& velInfo) const
{
  // average velocity in a cube of 10 points per direction around the point pSens

  // get blockId
  const size_t blockId = holdingBlockID(pSens, velInfo);

  // get block
  const auto& sensBinfo = velInfo[blockId];

  // get origin of block
  const std::array<Real,2> oSens = sensBinfo.pos<Real>(0, 0);

  // get inverse gridspacing in block
  const Real invh = 1/velInfo[blockId].h_gridpoint;

  // get index for sensor
  const std::array<int,2> iSens = safeIdInBlock(pSens, oSens, invh);

  // stencil for averaging
  static constexpr int stenBeg[3] = {-5,-5, 0}, stenEnd[3] = { 6, 6, 1};

  VectorLab vellab; vellab.prepare(*(sim.vel), stenBeg, stenEnd, 1);
  vellab.load(velInfo[blockId], 0); VectorLab & __restrict__ V = vellab;

  double avgX=0;
  double avgY=0;

  for (ssize_t i(-5); i < 6; ++i)
    for (ssize_t j(-5); j < 6; ++j)
    {
      avgX += V(iSens[0] + i, iSens[1] + j).u[0];
      avgY += V(iSens[0] + i, iSens[1] + j).u[1];
    }
  
  return std::vector<double> {avgX, avgY};

}

// return flow velocity at point of flow sensor:
std::array<Real, 2> Windmill::sensVel(const std::array<Real,2> pSens, const std::vector<cubism::BlockInfo>& velInfo) const
{
  // get blockId
  const size_t blockId = holdingBlockID(pSens, velInfo);

  // get block
  const auto& sensBinfo = velInfo[blockId];

  // get origin of block
  const std::array<Real,2> oSens = sensBinfo.pos<Real>(0, 0);

  // get inverse gridspacing in block
  const Real invh = 1/velInfo[blockId].h_gridpoint;

  // get index for sensor
  const std::array<int,2> iSens = safeIdInBlock(pSens, oSens, invh);

  // get velocity field at point
  const VectorBlock& b = * (const VectorBlock*) sensBinfo.ptrBlock;

  return std::array<Real, 2>{{b(iSens[0], iSens[1]).u[0],
                              b(iSens[0], iSens[1]).u[1]}};
};

// function that finds block id of block containing pos (x,y)
size_t Windmill::holdingBlockID(const std::array<Real,2> pos, const std::vector<cubism::BlockInfo>& velInfo) const
{
  // std::cout << "pos=(" << pos[0] << ", " << pos[1] << ")\n"; 
  for(size_t i=0; i<velInfo.size(); ++i)
  {
    // get gridspacing in block
    const Real h = velInfo[i].h_gridpoint;

    // compute lower left corner of block
    std::array<Real,2> MIN = velInfo[i].pos<Real>(0, 0);
    for(int j=0; j<2; ++j)
      MIN[j] -= 0.5 * h; // pos returns cell centers

    // compute top right corner of block
    std::array<Real,2> MAX = velInfo[i].pos<Real>(VectorBlock::sizeX-1, VectorBlock::sizeY-1);
    for(int j=0; j<2; ++j)
      MAX[j] += 0.5 * h; // pos returns cell centers

    // check whether point is inside block
    // std::cout << "MIN=(" << MIN[0] << ", " << MIN[1] << "); MAX=(" << MAX[0] << ", " << MAX[1] << ")\n"; 
    if( pos[0] >= MIN[0] && pos[1] >= MIN[1] && pos[0] <= MAX[0] && pos[1] <= MAX[1] )
    {
      // select obstacle block
      if(obstacleBlocks[i] != nullptr ){
        return i;
      }
    }
  }
  printf("ABORT: coordinate (%g,%g) could not be associated to obstacle block\n", pos[0], pos[1]);
  fflush(0); abort();
  return 0;
};

// function that gives indice of point in block
std::array<int, 2> Windmill::safeIdInBlock(const std::array<Real,2> pos, const std::array<Real,2> org, const Real invh ) const
{
  const int indx = (int) std::round((pos[0] - org[0])*invh);
  const int indy = (int) std::round((pos[1] - org[1])*invh);
  const int ix = std::min( std::max(0, indx), VectorBlock::sizeX-1);
  const int iy = std::min( std::max(0, indy), VectorBlock::sizeY-1);
  return std::array<int, 2>{{ix, iy}};
};
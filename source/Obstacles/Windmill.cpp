#include "Windmill.h"
#include "ShapeLibrary.h"
#include "../Utils/BufferedLogger.h"
#include <cmath>

using namespace cubism;

void Windmill::create(const std::vector<BlockInfo>& vInfo)
{
  // windmill stuff
  const Real h =  vInfo[0].h;
  for(auto & entry : obstacleBlocks) delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock*> (vInfo.size(), nullptr);

  #pragma omp parallel
  {
    //// original ellipse
    /*
    // center of ellipse 1 wrt to center of windmill,at T=0
    double center_orig1[2] = {smajax/2, -(smajax/2)*std::tan(M_PI/6)};
    // center of ellipse 1 wrt to origin
    double center1[2] = {center[0] + std::cos(orientation) * center_orig1[0] - std::sin(orientation)* center_orig1[1], 
                         center[1] + std::sin(orientation) * center_orig1[0] + std::cos(orientation) * center_orig1[1]};

    FillBlocks_Ellipse kernel1(smajax, sminax, h, center1, (orientation + 2*M_PI / 3));

    // center of ellipse 2 wrt to center of windmill,at T=0
    double center_orig2[2] = {0, smajax/(2*std::cos(M_PI/6))};
    // center of ellipse 1 wrt to origin
    double center2[2] = {center[0] + std::cos(orientation) * center_orig2[0] - std::sin(orientation)* center_orig2[1], 
                         center[1] + std::sin(orientation) * center_orig2[0] + std::cos(orientation) * center_orig2[1]};

    FillBlocks_Ellipse kernel2(smajax, sminax, h, center2, (orientation + M_PI / 3));

    // center of ellipse 3 wrt to center of windmill,at T=0
    double center_orig3[2] = {-smajax/2, -(smajax/2)*std::tan(M_PI/6)};
    // center of ellipse 1 wrt to origin
    double center3[2] = {center[0] + std::cos(orientation) * center_orig3[0] - std::sin(orientation)* center_orig3[1], 
                         center[1] + std::sin(orientation) * center_orig3[0] + std::cos(orientation) * center_orig3[1]};

    FillBlocks_Ellipse kernel3(smajax, sminax, h, center3, orientation);
    */

    //// symmetrical ellipse

    Real frac = 0.55;
    Real d = smajax * (1.0 - 2.0*frac/3.0);

    // center of ellipse 1 wrt to center of windmill,at T=0, bottom one
    Real center_orig1[2] = {d * std::sin(M_PI/6), -d * std::cos(M_PI/6)};
    // center of ellipse 1 wrt to origin
    Real center1[2] = {center[0] + std::cos(orientation) * center_orig1[0] - std::sin(orientation)* center_orig1[1], 
                         center[1] + std::sin(orientation) * center_orig1[0] + std::cos(orientation) * center_orig1[1]};

    FillBlocks_Ellipse kernel1(smajax, sminax, h, center1, (orientation + 2*M_PI / 3));

    // center of ellipse 2 wrt to center of windmill,at T=0, top one
    Real center_orig2[2] = {d * std::sin(M_PI/6), +d * std::cos(M_PI/6)};
    // center of ellipse 1 wrt to origin
    Real center2[2] = {center[0] + std::cos(orientation) * center_orig2[0] - std::sin(orientation)* center_orig2[1], 
                         center[1] + std::sin(orientation) * center_orig2[0] + std::cos(orientation) * center_orig2[1]};

    FillBlocks_Ellipse kernel2(smajax, sminax, h, center2, (orientation + M_PI / 3));

    // center of ellipse 3 wrt to center of windmill,at T=0, horizontal one
    Real center_orig3[2] = {-d, 0};
    // center of ellipse 1 wrt to origin
    Real center3[2] = {center[0] + std::cos(orientation) * center_orig3[0] - std::sin(orientation)* center_orig3[1], 
                         center[1] + std::sin(orientation) * center_orig3[0] + std::cos(orientation) * center_orig3[1]};

    FillBlocks_Ellipse kernel3(smajax, sminax, h, center3, orientation);


    // fill blocks for the three ellipses
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

void Windmill::updateVelocity(Real dt)
{
  // simulation.cpp file was changed so that it would land on multiples of time_step
  // update velocity is called before update step so only need to define the prev_dt here

  if (std::floor((1/time_step) * (sim.time + sim.dt)) - std::floor((1/time_step) * (sim.time)) != 0)
  {
     prev_dt = sim.dt;
  }

  // omega = omega_over_time(sim.time); // for the creation of the baseline
  // omega = omega + action_ang_accel * dt; // for RL
  //omega = action_ang_vel_max *  (1-std::cos(2 * M_PI * action_freq * sim.time));
  omega = action_ang_vel_max * std::sin(2 * M_PI * action_freq * sim.time);

}

void Windmill::updatePosition(Real dt)
{
  if ( (std::floor((1/time_step) * (sim.time)) - std::floor((1/time_step) * (sim.time - prev_dt)) != 0)) // every .05 seconds print velocity profile
  {
    if (sim.rank == 0) print_vel_profile(avg_profile);

    // reset avg_profile to zero for next time step
    //std::fill(&avg_profile[0][0], &avg_profile[1][numberRegions-1], 0.0);
    avg_profile = std::vector<std::vector<Real>> (2, std::vector<Real>(numberRegions, 0.0));
    // temp_torque = torque_over_time(sim.time);
    // omega = omega + temp_torque * time_step / penalJ;
  }

  Shape::updatePosition(dt);

  update_avg_vel_profile(dt);
}

void Windmill::printRewards(Real r_flow)
{
  std::stringstream ssF;
  ssF<<sim.path2file<<"/rewards_"<<obstacleID<<".dat";
  std::stringstream & fout = logger.get_stream(ssF.str());

  fout<<sim.time<<" "<<r_flow<<std::endl;
  fout.flush();
}

void Windmill::printActions(double angvel, double freq)
{
  std::stringstream ssF;
  ssF<<sim.path2file<<"/action_"<<obstacleID<<".dat";
  std::stringstream & fout = logger.get_stream(ssF.str());

  fout<<sim.time<<" "<<angvel<<" "<<freq<<std::endl;
  fout.flush();
}

void Windmill::act(std::vector<double> action)
{
  // the action is the tuple (max angular velocity, frequency)
  // action_ang_accel = action;
  action_ang_vel_max = action[0];
  action_freq = action[1];
  if (sim.rank == 0) printActions(action_ang_vel_max, action_freq);
}

double Windmill::reward(std::vector<double> target_profile, std::vector<double> profile_t_1, std::vector<double> profile_t_, double norm_prof)
{
   std::cerr << "ERROR: no reward defined for Windmill::reward. Should you even be here?" << std::endl;
   abort();
   return -1; //to silence compilation warnings
}

void Windmill::update_avg_vel_profile(Real dt)
{
  std::vector<std::vector<Real>> vel = vel_profile();
  for (int k(0); k < numberRegions; ++k)
  {
    avg_profile[0][k] += vel[0][k] * dt / time_step;
    avg_profile[1][k] += vel[1][k] * dt / time_step;
  }
}

void Windmill::print_vel_profile(std::vector<std::vector<Real>> vel_profile)
{

  if(not sim.muteAll)
  {
    std::stringstream ssF;
    ssF<<sim.path2file<<"/x_velocity_profile_"<<obstacleID<<".dat";
    std::stringstream & fout = logger.get_stream(ssF.str());
    fout<<sim.time;

    for (int k = 0; k < numberRegions; ++k)
    {
      // need to normalize profile by the time step
      fout<<" "<<vel_profile[0][k];
    }
    fout<<std::endl;
    
    fout.flush();

    std::stringstream ssF2;
    ssF2<<sim.path2file<<"/y_velocity_profile_"<<obstacleID<<".dat";
    std::stringstream & fout2 = logger.get_stream(ssF2.str());
    fout2<<sim.time;

    for (int k = 0; k < numberRegions; ++k)
    {
      // need to normalize profile by the time step
      fout2<<" "<<vel_profile[1][k];
    }
    fout2<<std::endl;
    
    fout2.flush();
  }
}

std::vector<std::vector<Real>> Windmill::vel_profile()
{
  // We take a region of size 0.35 * 0.0875, which cuts 2 vertical blocks in half along a vertical line
  // we choose to split these 2 blocks in the vertical dimension into 16 intervals
  // each one of the 16 intervals has a height of 0.35/16 = 0.021875
  // we average the velocity in each of the 32 intervals

  std::vector<Real> vel_x_avg(numberRegions, 0.0);
  std::vector<Real> vel_y_avg(numberRegions, 0.0);
  std::vector<Real> region_area(numberRegions, 0.0);

  Real height = 0.021875;


  // the sim.vel blocks info is split over multiple mpi processes, need to work with mpi
  // to communicate the values of the blocks etc. 
  // depending on the rank, only certain values will be accessible, the other
  const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo();
  
  // loop over all the blocks in the current rank
  for(size_t t=0; t < velInfo.size(); ++t)
  {
    // get pointer on block
    const VectorBlock& b = * (const VectorBlock*) velInfo[t].ptrBlock;
    // loop over all the points
    Real da = velInfo[t].h * velInfo[t].h;

    for(size_t i=0; i < b.sizeX; ++i)
      {
        for(size_t j=0; j < b.sizeY; ++j)
        {
          const std::array<Real,2> oSens = velInfo[t].pos<Real>(i, j);
          int num = numRegion(oSens, height);
          if (num)
          {
            region_area[num-1] += da;
            vel_x_avg[num-1] += b(i, j).u[0] * da;
            vel_y_avg[num-1] += b(i,j).u[1] * da;
          }
        }
      }
  }


  // collects the sum over all the components of the velocity profile into the vector vel_avg
  MPI_Allreduce(MPI_IN_PLACE, &vel_x_avg[0], numberRegions, MPI_Real, MPI_SUM, sim.comm);
  MPI_Allreduce(MPI_IN_PLACE, &vel_y_avg[0], numberRegions, MPI_Real, MPI_SUM, sim.comm);

  // collects the sum over all the region areas in order to perform the averaging correctly
  MPI_Allreduce(MPI_IN_PLACE, &region_area[0], numberRegions, MPI_Real, MPI_SUM, sim.comm);


  // 2 x numberRegions vector
  std::vector<std::vector<Real>> vel_profile = std::vector<std::vector<Real>>(2, std::vector<Real> (numberRegions, 0.0));

  // divide each vel_avg by the corresponding area
  for (int k = 0; k < numberRegions; ++k)
  {
    vel_profile[0][k] = vel_x_avg[k]/region_area[k];
    vel_profile[1][k] = vel_y_avg[k]/region_area[k];
  }

  return vel_profile;
}

int Windmill::numRegion(const std::array<Real, 2> point, Real height) const
{
  // returns 0 if outside of the box
  std::array<Real, 2> lower_left = {x_start, y_start};
  std::array<Real, 2> upper_right = {x_end, y_end};
  Real rel_pos_height = point[1] - lower_left[1];
  //std::array<Real, 2> rel_pos = {point[0] - lower_left[0], point[1] - lower_left[1]};
  int num = 0;

  if(point[0] >= lower_left[0] && point[0] <= upper_right[0])
  {
    if(point[1] >= lower_left[1] && point[1] <= upper_right[1])
    {
      // point is inside the rectangle to compute velocity profile
      // now find out in what region of the rectangle we are in
      num = static_cast<int>(std::ceil(rel_pos_height/height));
      
      return num;
    }
  }

  return 0;
}

// set initial conditions of the agent
void Windmill::setInitialConditions(Real init_angle)
{
  // Intial fixed condition of angle and angular velocity
  
  printf("[Korali] Initial Conditions:\n");
  printf("[Korali] orientation: %f\n", (double)init_angle);

  setOrientation(init_angle);
}

Real Windmill::getAngularVelocity()
{
  return omega;
}

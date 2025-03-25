#include "Windmill.h"
#include "../Utils/BufferedLogger.h"
#include "ShapeLibrary.h"
#include <cmath>
using namespace cubism;
void Windmill::create(const std::vector<BlockInfo> &vInfo) {
  const Real h = vInfo[0].h;
  for (auto &entry : obstacleBlocks)
    delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock *>(vInfo.size(), nullptr);
#pragma omp parallel
  {
    Real frac = 0.55;
    Real d = smajax * (1.0 - 2.0 * frac / 3.0);
    Real center_orig1[2] = {d * std::sin(M_PI / 6), -d * std::cos(M_PI / 6)};
    Real center1[2] = {center[0] + std::cos(orientation) * center_orig1[0] -
                           std::sin(orientation) * center_orig1[1],
                       center[1] + std::sin(orientation) * center_orig1[0] +
                           std::cos(orientation) * center_orig1[1]};
    FillBlocks_Ellipse kernel1(smajax, sminax, h, center1,
                               (orientation + 2 * M_PI / 3));
    Real center_orig2[2] = {d * std::sin(M_PI / 6), +d * std::cos(M_PI / 6)};
    Real center2[2] = {center[0] + std::cos(orientation) * center_orig2[0] -
                           std::sin(orientation) * center_orig2[1],
                       center[1] + std::sin(orientation) * center_orig2[0] +
                           std::cos(orientation) * center_orig2[1]};
    FillBlocks_Ellipse kernel2(smajax, sminax, h, center2,
                               (orientation + M_PI / 3));
    Real center_orig3[2] = {-d, 0};
    Real center3[2] = {center[0] + std::cos(orientation) * center_orig3[0] -
                           std::sin(orientation) * center_orig3[1],
                       center[1] + std::sin(orientation) * center_orig3[0] +
                           std::cos(orientation) * center_orig3[1]};
    FillBlocks_Ellipse kernel3(smajax, sminax, h, center3, orientation);
#pragma omp for schedule(dynamic, 1)
    for (size_t i = 0; i < vInfo.size(); i++) {
      if (kernel1.is_touching(vInfo[i])) {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear();
      } else if (kernel2.is_touching(vInfo[i])) {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear();
      } else if (kernel3.is_touching(vInfo[i])) {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear();
      }
      ScalarBlock &B = *(ScalarBlock *)vInfo[i].ptrBlock;
      if (obstacleBlocks[vInfo[i].blockID] == nullptr)
        continue;
      kernel1(vInfo[i], B, *obstacleBlocks[vInfo[i].blockID]);
      kernel2(vInfo[i], B, *obstacleBlocks[vInfo[i].blockID]);
      kernel3(vInfo[i], B, *obstacleBlocks[vInfo[i].blockID]);
    }
  }
}
void Windmill::updateVelocity(Real dt) {
  if (std::floor((1 / time_step) * (sim.time + sim.dt)) -
          std::floor((1 / time_step) * (sim.time)) !=
      0) {
    prev_dt = sim.dt;
  }
  omega = action_ang_vel_max * std::sin(2 * M_PI * action_freq * sim.time);
}
void Windmill::updatePosition(Real dt) {
  if ((std::floor((1 / time_step) * (sim.time)) -
           std::floor((1 / time_step) * (sim.time - prev_dt)) !=
       0)) {
    if (sim.rank == 0)
      print_vel_profile(avg_profile);
    avg_profile = std::vector<std::vector<Real>>(
        2, std::vector<Real>(numberRegions, 0.0));
  }
  Shape::updatePosition(dt);
  update_avg_vel_profile(dt);
}
void Windmill::printRewards(Real r_flow) {
  std::stringstream ssF;
  ssF << sim.path2file << "/rewards_" << obstacleID << ".dat";
  std::stringstream &fout = logger.get_stream(ssF.str());
  fout << sim.time << " " << r_flow << std::endl;
  fout.flush();
}
void Windmill::printActions(double angvel, double freq) {
  std::stringstream ssF;
  ssF << sim.path2file << "/action_" << obstacleID << ".dat";
  std::stringstream &fout = logger.get_stream(ssF.str());
  fout << sim.time << " " << angvel << " " << freq << std::endl;
  fout.flush();
}
void Windmill::act(std::vector<double> action) {
  action_ang_vel_max = action[0];
  action_freq = action[1];
  if (sim.rank == 0)
    printActions(action_ang_vel_max, action_freq);
}
double Windmill::reward(std::vector<double> target_profile,
                        std::vector<double> profile_t_1,
                        std::vector<double> profile_t_, double norm_prof) {
  std::cerr << "ERROR: no reward defined for Windmill::reward. Should you even "
               "be here?"
            << std::endl;
  abort();
  return -1;
}
void Windmill::update_avg_vel_profile(Real dt) {
  std::vector<std::vector<Real>> vel = vel_profile();
  for (int k(0); k < numberRegions; ++k) {
    avg_profile[0][k] += vel[0][k] * dt / time_step;
    avg_profile[1][k] += vel[1][k] * dt / time_step;
  }
}
void Windmill::print_vel_profile(std::vector<std::vector<Real>> vel_profile) {
  if (not sim.muteAll) {
    std::stringstream ssF;
    ssF << sim.path2file << "/x_velocity_profile_" << obstacleID << ".dat";
    std::stringstream &fout = logger.get_stream(ssF.str());
    fout << sim.time;
    for (int k = 0; k < numberRegions; ++k) {
      fout << " " << vel_profile[0][k];
    }
    fout << std::endl;
    fout.flush();
    std::stringstream ssF2;
    ssF2 << sim.path2file << "/y_velocity_profile_" << obstacleID << ".dat";
    std::stringstream &fout2 = logger.get_stream(ssF2.str());
    fout2 << sim.time;
    for (int k = 0; k < numberRegions; ++k) {
      fout2 << " " << vel_profile[1][k];
    }
    fout2 << std::endl;
    fout2.flush();
  }
}
std::vector<std::vector<Real>> Windmill::vel_profile() {
  std::vector<Real> vel_x_avg(numberRegions, 0.0);
  std::vector<Real> vel_y_avg(numberRegions, 0.0);
  std::vector<Real> region_area(numberRegions, 0.0);
  Real height = 0.021875;
  const std::vector<cubism::BlockInfo> &velInfo = sim.vel->getBlocksInfo();
  for (size_t t = 0; t < velInfo.size(); ++t) {
    const VectorBlock &b = *(const VectorBlock *)velInfo[t].ptrBlock;
    Real da = velInfo[t].h * velInfo[t].h;
    for (size_t i = 0; i < b.sizeX; ++i) {
      for (size_t j = 0; j < b.sizeY; ++j) {
        const std::array<Real, 2> oSens = velInfo[t].pos<Real>(i, j);
        int num = numRegion(oSens, height);
        if (num) {
          region_area[num - 1] += da;
          vel_x_avg[num - 1] += b(i, j).u[0] * da;
          vel_y_avg[num - 1] += b(i, j).u[1] * da;
        }
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &vel_x_avg[0], numberRegions, MPI_Real, MPI_SUM,
                sim.comm);
  MPI_Allreduce(MPI_IN_PLACE, &vel_y_avg[0], numberRegions, MPI_Real, MPI_SUM,
                sim.comm);
  MPI_Allreduce(MPI_IN_PLACE, &region_area[0], numberRegions, MPI_Real, MPI_SUM,
                sim.comm);
  std::vector<std::vector<Real>> vel_profile =
      std::vector<std::vector<Real>>(2, std::vector<Real>(numberRegions, 0.0));
  for (int k = 0; k < numberRegions; ++k) {
    vel_profile[0][k] = vel_x_avg[k] / region_area[k];
    vel_profile[1][k] = vel_y_avg[k] / region_area[k];
  }
  return vel_profile;
}
int Windmill::numRegion(const std::array<Real, 2> point, Real height) const {
  std::array<Real, 2> lower_left = {x_start, y_start};
  std::array<Real, 2> upper_right = {x_end, y_end};
  Real rel_pos_height = point[1] - lower_left[1];
  int num = 0;
  if (point[0] >= lower_left[0] && point[0] <= upper_right[0]) {
    if (point[1] >= lower_left[1] && point[1] <= upper_right[1]) {
      num = static_cast<int>(std::ceil(rel_pos_height / height));
      return num;
    }
  }
  return 0;
}
void Windmill::setInitialConditions(Real init_angle) {
  setOrientation(init_angle);
}
Real Windmill::getAngularVelocity() { return omega; }

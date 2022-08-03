#include "../Shape.h"

class Windmill : public Shape
{
  // windmill stuff
  // 3 branches of same length
  const Real semiAxis[2]; 
  const Real smajax = std::max(semiAxis[0], semiAxis[1]);
  const Real sminax = std::min(semiAxis[0], semiAxis[1]);
  //const Real windscale = std::sqrt(forcedu*forcedu+forcedv*forcedv);
  const Real lengthscale = getCharLength();

  // keeps track of the of the average veloctiy profile between two rl time steps
  // weighted by the time step of the sim
  std::vector<std::vector<Real>> avg_profile = std::vector<std::vector<Real>>(2, std::vector<Real>(numberRegions, 0.0));


  Real time_step = 0.05;
  Real prev_dt = 0;
  double action_ang_vel_max = 0.;
  double action_freq = 0.;


  // domain for velocity profile
  Real x_start = 0.35;
  Real x_end = x_start + 0.0875;
  Real y_start = 0.175;
  Real y_end = 0.525;

  int numberRegions = 16;

 public:

  Windmill(SimulationData& s, cubism::ArgumentParser& p, Real C[2]):
  Shape(s,p,C), semiAxis{(Real) p("-semiAxisX").asDouble(), (Real) p("-semiAxisY").asDouble()}
  {
    action_ang_vel_max = forcedomega;
    omega = 0;
    action_freq = 0.5;
    setInitialConditions(0);
  }

  void resetAll() override
  {
    // reset all other variables
    Shape::resetAll();
  }

  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
  void updateVelocity(Real dt) override;
  void updatePosition(Real dt) override;

  void printRewards(Real r_flow);
  void printActions(double angvel, double freq);
  
  void act( std::vector<double> action);
  double reward(std::vector<double> target_profile, std::vector<double> profile_t_1, std::vector<double> profile_t_, double norm_prof);

  void update_avg_vel_profile(Real dt);
  void print_vel_profile(std::vector<std::vector<Real>> vel_profile);

  std::vector<std::vector<Real>> vel_profile();
  int numRegion(const std::array<Real, 2> point, Real height) const;
  void setInitialConditions(Real init_angle);
  Real getAngularVelocity();

  Real getCharLength() const override
  {
    return semiAxis[0] >= semiAxis[1] ? 2*semiAxis[0] : 2*semiAxis[1];
  }
};
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
  std::array<Real, 2> target = {0.7, 0.6};
  std::array<Real, 2> dimensions = {0.1, 0.1};
  Real energy = 0;

  // keeps track of the of the average veloctiy profile between two rl time steps
  // weighted by the time step of the sim
  std::vector<Real> avg_profile = vector<Real>(32, 0.0);


  Real time_step = 0.05;
  Real ang_accel;
  Real action_ang_accel = 0.;
  Real temp_torque = 0;
  Real prev_dt = 0;


  // domain for velocity profile
  Real x_start = 0.35;
  Real x_end = x_start + 0.0875;
  Real y_start = 0.35;
  Real y_end = 1.05;

 public:

  Windmill(SimulationData& s, cubism::ArgumentParser& p, Real C[2]):
  Shape(s,p,C), semiAxis{(Real) p("-semiAxisX").asDouble(), (Real) p("-semiAxisY").asDouble()}
  {
    ang_accel = forcedomega;
    omega = 0;
    // set a random orientation
    setInitialConditions(0);
  }

  void resetAll() override
  {
    // reset all other variables
    Shape::resetAll();
  }

  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
  void updateVelocity(Real dt) override;
  Real omega_over_time(Real time);
  void updatePosition(Real dt) override;

  void printRewards(Real r_flow);
  void printActions(Real value);
  
  void act( Real action );
  Real reward(std::vector<Real> target_profile, std::vector<Real> profile_t_1, std::vector<Real> profile_t_);

  void update_avg_vel_profile(Real dt);
  void print_vel_profile(std::vector<Real> vel_profile);

  std::vector<Real> vel_profile();
  int numRegion(const std::array<Real, 2> point, Real height) const;
  void setInitialConditions(Real init_angle);
  Real getAngularVelocity();

  Real getCharLength() const override
  {
    return semiAxis[0] >= semiAxis[1] ? 2*semiAxis[0] : 2*semiAxis[1];
  }
};
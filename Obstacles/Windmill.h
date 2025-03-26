#include "../Shape.h"
class Windmill : public Shape {
  const Real semiAxis[2];
  const Real smajax = std::max(semiAxis[0], semiAxis[1]);
  const Real sminax = std::min(semiAxis[0], semiAxis[1]);
  const Real lengthscale = getCharLength();
  Real time_step = 0.05;
  Real prev_dt = 0;
  double action_ang_vel_max = 0.;
  double action_freq = 0.;
  Real x_start = 0.35;
  Real x_end = x_start + 0.0875;
  Real y_start = 0.175;
  Real y_end = 0.525;
  int numberRegions = 16;

public:
  Windmill(SimulationData &s, cubism::ArgumentParser &p, Real C[2])
      : Shape(s, p, C), semiAxis{(Real)p("-semiAxisX").asDouble(),
                                 (Real)p("-semiAxisY").asDouble()},
        action_ang_vel_max(p("-angvelmax").asDouble()),
        action_freq(p("-freq").asDouble()) {
    omega = 0;
    setInitialConditions(0);
  }
  void resetAll() override { Shape::resetAll(); }
  void create(const std::vector<cubism::BlockInfo> &vInfo) override;
  void updateVelocity(Real dt) override;
  void updatePosition(Real dt) override;
  void setInitialConditions(Real init_angle);
  Real getAngularVelocity();
  Real getCharLength() const override {
    return semiAxis[0] >= semiAxis[1] ? 2 * semiAxis[0] : 2 * semiAxis[1];
  }
};

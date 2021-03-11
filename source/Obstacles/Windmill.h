#include "../Shape.h"

class Windmill : public Shape
{
  // windmill stuff
  // 3 branches of same length
  const Real semiAxis[2]; 
  const Real smajax = std::max(semiAxis[0], semiAxis[1]);
  const Real sminax = std::min(semiAxis[0], semiAxis[1]);

  Real diff_flow = 0;

 public:


 // Smart cyclinder stuff
  Real dist, oldDist;

  Windmill(SimulationData& s, cubism::ArgumentParser& p, double C[2]):
  Shape(s,p,C), semiAxis{(Real) p("-semiAxisX").asDouble(), (Real) p("-semiAxisY").asDouble()}
  {}

  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
  void updateVelocity(double dt) override;
  void updatePosition(double dt) override;

  Real getCharLength() const override
  {
    return 2 * smajax;
  }
  
  void act( std::vector<double> action );
  double reward( std::vector<double> target, std::vector<double> target_vel, const std::vector<cubism::BlockInfo>& velInfo, double C);
  std::vector<double> state( std::vector<double> target );

  // Helpers for State function
  std::array<Real, 2> Windmill::average(const std::array<Real,2> pSens, const std::vector<cubism::BlockInfo>& velInfo) const;

  std::array<Real, 2> sensVel(const std::array<Real,2> pSens, const std::vector<cubism::BlockInfo>& velInfo) const;
  
  size_t holdingBlockID(const std::array<Real,2> pos, const std::vector<cubism::BlockInfo>& velInfo) const;

  std::array<int, 2> safeIdInBlock(const std::array<Real,2> pos, const std::array<Real,2> org, const Real invh ) const;

};
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

  Windmill(SimulationData& s, cubism::ArgumentParser& p, double C[2]):
  Shape(s,p,C), semiAxis{(Real) p("-semiAxisX").asDouble(), (Real) p("-semiAxisY").asDouble()}
  {}

  void resetAll() override
  {
    diff_flow = 0;
    // reset all other variables
    Shape::resetAll();
  }

  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
  void updateVelocity(double dt) override;
  void updatePosition(double dt) override;
  
  void act( double action );
  double reward( std::array<Real,2> target, std::array<Real,2> target_vel, double C = 10);
  std::array<Real,2> state();

  // Helpers for State function
  std::array<Real,2> average(const std::array<Real,2> pSens, const std::vector<cubism::BlockInfo>& velInfo) const;

  std::array<Real, 2> sensVel(const std::array<Real,2> pSens, const std::vector<cubism::BlockInfo>& velInfo) const;
  
  size_t holdingBlockID(const std::array<Real,2> pos, const std::vector<cubism::BlockInfo>& velInfo) const;

  std::array<int, 2> safeIdInBlock(const std::array<Real,2> pos, const std::array<Real,2> org, const Real invh ) const;

  Real getCharLength() const // have to have this method otherwise class is virtual
  {
    return 0;
  }

};
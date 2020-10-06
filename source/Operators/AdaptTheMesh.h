#pragma once

#include "../Operator.h"
#include "Cubism/AMR_MeshAdaptation.h"
class AdaptTheMesh : public Operator
{
 public:
  AdaptTheMesh(SimulationData& s) : Operator(s) {}

  void operator()(const double dt);

  std::string getName()
  {
    return "AdaptTheMesh";
  }
};
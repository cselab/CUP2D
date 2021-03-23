//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "CarlingFish.h"
#include "FishData.h"
#include "FishUtilities.h"

using namespace cubism;

CarlingFish::CarlingFish(SimulationData&s, ArgumentParser&p, double C[2])
  : Fish(s,p,C) {
  const Real ampFac = p("-amplitudeFactor").asDouble(1.0);
  myFish = new AmplitudeFish(length, Tperiod, phaseShift, sim.getH(), ampFac);
  printf("AmplitudeFish %d %f %f %f\n",myFish->Nm, length, Tperiod, phaseShift);
}

void CarlingFish::create(const std::vector<BlockInfo>& vInfo) {
  Fish::create(vInfo);
}

void CarlingFish::resetAll() {
  Fish::resetAll();
}

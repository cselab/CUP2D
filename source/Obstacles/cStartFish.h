//
//  CubismUP_2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Ioannis Mandralis (ioannima@ethz.ch).
//

#pragma once
#include "Fish.h"

//#define STEFANS_SENSORS_STATE

class StefanFish: public Fish
{
    const bool bCorrectTrajectory;
    const bool bCorrectPosition;
public:
    void act(const Real lTact, const std::vector<double>& a) const;
    double getLearnTPeriod() const;
    double getPhase(const double t) const;

    void resetAll() override;
    StefanFish(SimulationData&s, cubism::ArgumentParser&p, double C[2]);
    void create(const std::vector<cubism::BlockInfo>& vInfo) override;

    // member functions for state/reward
    std::vector<double> state(Shape*const p) const;
    std::vector<double> getCStartState() const;
    double reward() const;

};

//
//  CubismUP_2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Ioannis Mandralis (ioannima@ethz.ch).
//

#pragma once
#include "Fish.h"

class NeuroKinematicFish: public Fish
{
public:
    // Core functions
    NeuroKinematicFish(SimulationData&s, cubism::ArgumentParser&p, double C[2]);
    void resetAll() override;
    void create(const std::vector<cubism::BlockInfo>& vInfo) override;
    void act(const Real lTact, const std::vector<double>& a) const;
    // Member functions for state/reward
    std::vector<double> state() const;
    // Helper functions
    void getTarget(double outTarget[2]) const;
    void setTarget(double inTarget[2]) const;
    double getRadialDisplacement() const;
    double getDistanceFromTarget() const;
    double getTimeNextAct() const;

};

//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "Fish.h"

class NeuroKinematicFish: public Fish
{
public:
    // Core functions
    NeuroKinematicFish(SimulationData&s, cubism::ArgumentParser&p, Real C[2]);
    void resetAll() override;
    void create(const std::vector<cubism::BlockInfo>& vInfo) override;
    void act(const Real lTact, const std::vector<Real>& a) const;
    // Member functions for state/reward
    std::vector<Real> state() const;
    // Helper functions
    void getTarget(Real outTarget[2]) const;
    void setTarget(Real inTarget[2]) const;
    Real getRadialDisplacement() const;
    Real getPolarAngle() const;
    Real getDistanceFromTarget() const;
    Real getTimeNextAct() const;

};

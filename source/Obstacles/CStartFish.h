//
//  CubismUP_2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Ioannis Mandralis (ioannima@ethz.ch).
//

#pragma once
#include "Fish.h"

class CStartFish: public Fish
{
public:
    void act(const Real lTact, const std::vector<double>& a) const;
    void actTurn(const Real lTact, const std::vector<double>& a) const;
//    void actSimple(const Real lTact, const std::vector<double>& a) const;
//    void actModulate(const Real lTact, const std::vector<double>& a) const;
    double getPrep() const;

    void resetAll() override;
    CStartFish(SimulationData&s, cubism::ArgumentParser&p, double C[2]);
    void create(const std::vector<cubism::BlockInfo>& vInfo) override;

    // member functions for state/reward
    std::vector<double> state() const;
    double getRadialDisplacement() const;

};

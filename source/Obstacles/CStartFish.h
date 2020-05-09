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
    void actCStart(const Real lTact, const std::vector<double>& a) const;
    void actTurn(const Real lTact, const std::vector<double>& a) const;
    void setTarget(double desiredTarget[2]) const;
    void getTarget(double outTarget[2]) const;
//    void actSimple(const Real lTact, const std::vector<double>& a) const;
//    void actModulate(const Real lTact, const std::vector<double>& a) const;
    double getPrep() const;

    void resetAll() override;
    CStartFish(SimulationData&s, cubism::ArgumentParser&p, double C[2]);
    void create(const std::vector<cubism::BlockInfo>& vInfo) override;

    // member functions for state/reward
    std::vector<double> stateEscape() const;
    std::vector<double> stateSequentialEscape() const;
    std::vector<double> stateEscapeTradeoff() const;
    std::vector<double> stateEscapeVariableEnergy() const;
    std::vector<double> stateTarget() const;
    std::vector<double> stateCStart() const;
    double getRadialDisplacement() const;
    double getDistanceFromTarget() const;
    double getTimeNextAct() const;
    void setEnergyExpended(const double setEnergyExpended);
    void setDistanceTprop(const double distanceTprop);
    double getDistanceTprop() const;
    void setVirtualOrigin(const double vo[2]);
    void setEnergyBudget(const double baselineEnergy);
    double getEnergyBudget() const;
    double getEnergyExpended() const;
    double getPolarAngle() const;
};

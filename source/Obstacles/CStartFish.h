//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once
#include "Fish.h"

class CStartFish: public Fish
{
public:
    void act(const Real lTact, const std::vector<Real>& a) const;
    void actCStart(const Real lTact, const std::vector<Real>& a) const;
    void actTurn(const Real lTact, const std::vector<Real>& a) const;
    void setTarget(Real desiredTarget[2]) const;
    void getTarget(Real outTarget[2]) const;
//    void actSimple(const Real lTact, const std::vector<Real>& a) const;
//    void actModulate(const Real lTact, const std::vector<Real>& a) const;
    Real getPrep() const;

    void resetAll() override;
    CStartFish(SimulationData&s, cubism::ArgumentParser&p, Real C[2]);
    void create(const std::vector<cubism::BlockInfo>& vInfo) override;

    // member functions for state/reward
    std::vector<Real> stateEscape() const;
    std::vector<Real> stateSequentialEscape() const;
    std::vector<Real> stateEscapeTradeoff() const;
    std::vector<Real> stateEscapeVariableEnergy() const;
    std::vector<Real> stateTarget() const;
    std::vector<Real> stateCStart() const;
    Real getRadialDisplacement() const;
    Real getDistanceFromTarget() const;
    Real getTimeNextAct() const;
    void setEnergyExpended(const Real setEnergyExpended);
    void setDistanceTprop(const Real distanceTprop);
    Real getDistanceTprop() const;
    void setVirtualOrigin(const Real vo[2]);
    void setEnergyBudget(const Real baselineEnergy);
    Real getEnergyBudget() const;
    Real getEnergyExpended() const;
    Real getPolarAngle() const;
};

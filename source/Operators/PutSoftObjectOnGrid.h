#pragma once

#include "../Operator.h"

class Shape;
class PutSoftObjectsOnGrid : public Operator
{
    protected:
    const std::vector<cubism::BlockInfo>& velInfo   = sim.vel->getBlocksInfo();
    const std::vector<cubism::BlockInfo>& tmpInfo   = sim.tmp->getBlocksInfo();
    const std::vector<cubism::BlockInfo>& chiInfo   = sim.chi->getBlocksInfo();
    const std::vector<cubism::BlockInfo>& EchiInfo   = sim.Echi->getBlocksInfo();
    const std::vector<cubism::BlockInfo>& invmInfo  = sim.invm->getBlockInfo();
    public:
    using Operator::Operator;
    void operator()(Real dt) override;
    std::string getName() override
  {
    return "PutSoftObjectsOnGrid";
  }
}
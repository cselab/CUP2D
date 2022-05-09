#pragma once
#include "../Operator.h"
class advInvm : public Operator
{
	const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo();
	const std::vector<cubism::BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();
	const std::vector<cubism::BlockInfo>& vOldInfo = sim.vOld->getBlocksInfo();
	const std::vector<cubism::BlockInfo>& invmInfo = sim.invm->getBlocksInfo();
	const std::vector<cubism::BlockInfo>& uDefInfo = sim.uDef->getBlocksInfo();
	void advect(double dt);
public:
	advInvm(SimulationData& s) : Operator(s) { }
	void operator()(const double dt);
	std::string getName()
	{
		return "advInvm";
	}
};


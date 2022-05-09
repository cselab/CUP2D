#pragma once
#include "../Operator.h"
class advInvm : public Operator
{
	const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo();
	const std::vector<cubism::BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();
	const std::vector<cubism::BlockInfo>& tmpV1Info = sim.tmpV1->getBlocksInfo();
	const std::vector<cubism::BlockInfo>& tmpV2Info = sim.tmpV2->getBlocksInfo();
	const std::vector<cubism::BlockInfo>& invmInfo = sim.invm->getBlocksInfo();
	void advect(double dt);
public:
	advInvm(SimulationData& s) : Operator(s) { }
	void operator()(const double dt);
	std::string getName()
	{
		return "advInvm";
	}
};


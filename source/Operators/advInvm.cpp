#include "advInvm.h"

using namespace cubism;

#ifdef CUP2D_PRESERVE_SYMMETRY
#define CUP2D_DISABLE_OPTIMIZATIONS __attribute__((optimize("-O1")))
#else
#define CUP2D_DISABLE_OPTIMIZATIONS
#endif

CUP2D_DISABLE_OPTIMIZATIONS
static inline Real weno5_plus(const Real um2, const Real um1, const Real u, const Real up1, const Real up2)
{
	const Real exponent = 2;
	const Real e = 1e-6;
	const Real b1 = 13.0 / 12.0*pow((um2 + u) - 2 * um1, 2) + 0.25*pow((um2 + 3 * u) - 4 * um1, 2);
	const Real b2 = 13.0 / 12.0*pow((um1 + up1) - 2 * u, 2) + 0.25*pow(um1 - up1, 2);
	const Real b3 = 13.0 / 12.0*pow((u + up2) - 2 * up1, 2) + 0.25*pow((3 * u + up2) - 4 * up1, 2);
	const Real g1 = 0.1;
	const Real g2 = 0.6;
	const Real g3 = 0.3;
	const Real what1 = g1 / pow(b1 + e, exponent);
	const Real what2 = g2 / pow(b2 + e, exponent);
	const Real what3 = g3 / pow(b3 + e, exponent);
	const Real aux = 1.0 / ((what1 + what3) + what2);
	const Real w1 = what1 * aux;
	const Real w2 = what2 * aux;
	const Real w3 = what3 * aux;
	const Real f1 = (11.0 / 6.0)*u + ((1.0 / 3.0)*um2 - (7.0 / 6.0)*um1);
	const Real f2 = (5.0 / 6.0)*u + ((-1.0 / 6.0)*um1 + (1.0 / 3.0)*up1);
	const Real f3 = (1.0 / 3.0)*u + ((+5.0 / 6.0)*up1 - (1.0 / 6.0)*up2);
	return (w1*f1 + w3 * f3) + w2 * f2;
}

CUP2D_DISABLE_OPTIMIZATIONS
static inline Real weno5_minus(const Real um2, const Real um1, const Real u, const Real up1, const Real up2)
{
	const Real exponent = 2;
	const Real e = 1e-6;
	const Real b1 = 13.0 / 12.0*pow((um2 + u) - 2 * um1, 2) + 0.25*pow((um2 + 3 * u) - 4 * um1, 2);
	const Real b2 = 13.0 / 12.0*pow((um1 + up1) - 2 * u, 2) + 0.25*pow(um1 - up1, 2);
	const Real b3 = 13.0 / 12.0*pow((u + up2) - 2 * up1, 2) + 0.25*pow((3 * u + up2) - 4 * up1, 2);
	const Real g1 = 0.3;
	const Real g2 = 0.6;
	const Real g3 = 0.1;
	const Real what1 = g1 / pow(b1 + e, exponent);
	const Real what2 = g2 / pow(b2 + e, exponent);
	const Real what3 = g3 / pow(b3 + e, exponent);
	const Real aux = 1.0 / ((what1 + what3) + what2);
	const Real w1 = what1 * aux;
	const Real w2 = what2 * aux;
	const Real w3 = what3 * aux;
	const Real f1 = (1.0 / 3.0)*u + ((-1.0 / 6.0)*um2 + (5.0 / 6.0)*um1);
	const Real f2 = (5.0 / 6.0)*u + ((1.0 / 3.0)*um1 - (1.0 / 6.0)*up1);
	const Real f3 = (11.0 / 6.0)*u + ((-7.0 / 6.0)*up1 + (1.0 / 3.0)*up2);
	return (w1*f1 + w3 * f3) + w2 * f2;
}

static inline Real derivative(const Real U, const Real um3, const Real um2, const Real um1,
	const Real u,
	const Real up1, const Real up2, const Real up3)
{
	Real fp = 0.0;
	Real fm = 0.0;
	if (U > 0)
	{
		fp = weno5_plus(um2, um1, u, up1, up2);
		fm = weno5_plus(um3, um2, um1, u, up1);
	}
	else
	{
		fp = weno5_minus(um1, u, up1, up2, up3);
		fm = weno5_minus(um2, um1, u, up1, up2);
	}
	return (fp - fm);
}

static inline Real dINVMX_adv(const VectorLab&INVM, const Real V[2], const Real uinf[2], const Real advF, const int ix, const int iy)
{
	const Real u = V[0];
	const Real v = V[1];
	const Real UU = u + uinf[0];
	const Real VV = v + uinf[1];

	const Real invm_v = INVM(ix, iy).u[0];
	const Real invm_v_p1x = INVM(ix + 1, iy).u[0];
	const Real invm_v_p2x = INVM(ix + 2, iy).u[0];
	const Real invm_v_p3x = INVM(ix + 3, iy).u[0];
	const Real invm_v_m1x = INVM(ix - 1, iy).u[0];
	const Real invm_v_m2x = INVM(ix - 2, iy).u[0];
	const Real invm_v_m3x = INVM(ix - 3, iy).u[0];

	const Real invm_v_p1y = INVM(ix, iy + 1).u[0];
	const Real invm_v_p2y = INVM(ix, iy + 2).u[0];
	const Real invm_v_p3y = INVM(ix, iy + 3).u[0];
	const Real invm_v_m1y = INVM(ix, iy - 1).u[0];
	const Real invm_v_m2y = INVM(ix, iy - 2).u[0];
	const Real invm_v_m3y = INVM(ix, iy - 3).u[0];


	const Real dinvmdx = derivative(UU, invm_v_m3x, invm_v_m2x, invm_v_m1x, invm_v, invm_v_p1x, invm_v_p2x, invm_v_p3x);
	const Real dinvmdy = derivative(VV, invm_v_m3y, invm_v_m2y, invm_v_m1y, invm_v, invm_v_p1y, invm_v_p2y, invm_v_p3y);

	return advF * (UU*dinvmdx + VV * dinvmdy);
}

static inline Real dINVMY_adv(const VectorLab&INVM, const Real V[2], const Real uinf[2], const Real advF, const int ix, const int iy)
{
	const Real u = V[0];
	const Real v = V[1];
	const Real UU = u + uinf[0];
	const Real VV = v + uinf[1];

	const Real invm_v = INVM(ix, iy).u[1];
	const Real invm_v_p1x = INVM(ix + 1, iy).u[1];
	const Real invm_v_p2x = INVM(ix + 2, iy).u[1];
	const Real invm_v_p3x = INVM(ix + 3, iy).u[1];
	const Real invm_v_m1x = INVM(ix - 1, iy).u[1];
	const Real invm_v_m2x = INVM(ix - 2, iy).u[1];
	const Real invm_v_m3x = INVM(ix - 3, iy).u[1];

	const Real invm_v_p1y = INVM(ix, iy + 1).u[1];
	const Real invm_v_p2y = INVM(ix, iy + 2).u[1];
	const Real invm_v_p3y = INVM(ix, iy + 3).u[1];
	const Real invm_v_m1y = INVM(ix, iy - 1).u[1];
	const Real invm_v_m2y = INVM(ix, iy - 2).u[1];
	const Real invm_v_m3y = INVM(ix, iy - 3).u[1];


	const Real dinvmdx = derivative(UU, invm_v_m3x, invm_v_m2x, invm_v_m1x, invm_v, invm_v_p1x, invm_v_p2x, invm_v_p3x);
	const Real dinvmdy = derivative(VV, invm_v_m3y, invm_v_m2y, invm_v_m1y, invm_v, invm_v_p1y, invm_v_p2y, invm_v_p3y);

	return advF * (UU*dinvmdx + VV * dinvmdy);
}


struct KernelAdvect
{
	KernelAdvect(const SimulationData & s, const Real c, const Real uinfx, const Real uinfy) : sim(s), coef(c)
	{
		uinf[0] = uinfx;
		uinf[1] = uinfy;
	}
	const SimulationData & sim;
	const Real coef;
	Real uinf[2];
	const StencilInfo stencil{ -3, -3, 0, 4, 4, 1, true, {0,1} };
	const std::vector<cubism::BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();
	const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo();
	void operator()(VectorLab& lab, const BlockInfo& info) const
	{
		const Real h = info.h;
		const Real afac = -sim.dt*h;
		VectorBlock & __restrict__ TMP = *(VectorBlock*)tmpVInfo[info.blockID].ptrBlock;
		VectorBlock & __restrict__ V = *(VectorBlock*)velInfo[info.blockID].ptrBlock;
		for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
			for (int ix = 0; ix < VectorBlock::sizeX; ++ix)
			{
				const Real V0[2] = { V(ix,iy).u[0],V(ix,iy).u[1] };
				TMP(ix, iy).u[0] = coef * dINVMX_adv(lab, V0, uinf, afac, ix, iy);
				TMP(ix, iy).u[1] = coef * dINVMY_adv(lab, V0, uinf, afac, ix, iy);
			}
	}
};


void advInvm::operator()(const Real dt)
{
	sim.startProfiler("advDiff");
	const size_t Nblocks = velInfo.size();
	const Real UINF[2] = { sim.uinfx, sim.uinfy };
	/********************************************************************/
	//invm1=invm0+dt*RHS(invm0)
	KernelAdvect Step1(sim,1, UINF[0], UINF[1]);
	cubism::compute<VectorLab>(Step1, sim.invm, sim.tmpV);
	// compute invm1 and save it in vold
	#pragma omp parallel for
	for (size_t i = 0; i < Nblocks; i++)
	{
		const VectorBlock & __restrict__ INVM0 = *(VectorBlock*)invmInfo[i].ptrBlock;
		VectorBlock & __restrict__ INVM1 = *(VectorBlock*)tmpV1Info[i].ptrBlock;
		const VectorBlock & __restrict__ tmpV = *(VectorBlock*)tmpVInfo[i].ptrBlock;
		const Real ih2 = 1.0 / (velInfo[i].h*velInfo[i].h);
		for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
			for (int ix = 0; ix < VectorBlock::sizeX; ++ix)
			{
				INVM1(ix, iy).u[0] = INVM0(ix, iy).u[0] + tmpV(ix, iy).u[0] * ih2;
				INVM1(ix, iy).u[1] = INVM0(ix, iy).u[1] + tmpV(ix, iy).u[1] * ih2;
			}
	}
																					/********************************************************************/
	//invm2=0.75*invm0+0.25*invm1+0.25*dt*RHS(invm1)
	KernelAdvect Step2(sim, 0.25, UINF[0], UINF[1]);
	cubism::compute<VectorLab>(Step2, sim.tmpV1, sim.tmpV);
	//compute invm2 and save it in uDef
	#pragma omp parallel for
	for (size_t i = 0; i < Nblocks; i++)
	{
		VectorBlock & __restrict__ INVM0 = *(VectorBlock*)invmInfo[i].ptrBlock;
		const VectorBlock & __restrict__ INVM1 = *(VectorBlock*)tmpV1Info[i].ptrBlock;
		VectorBlock & __restrict__ INVM2 = *(VectorBlock*)tmpV2Info[i].ptrBlock;
		const VectorBlock & __restrict__ tmpV = *(VectorBlock*)tmpVInfo[i].ptrBlock;
		const Real ih2 = 1.0 / (velInfo[i].h*velInfo[i].h);
		for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
			for (int ix = 0; ix < VectorBlock::sizeX;++ix)															{
				INVM2(ix, iy).u[0] = 0.75*INVM0(ix, iy).u[0] + 0.25*INVM1(ix, iy).u[0]+ tmpV(ix, iy).u[0] * ih2;
				INVM2(ix, iy).u[1] = 0.75*INVM0(ix, iy).u[1] + 0.25*INVM1(ix, iy).u[1] + tmpV(ix, iy).u[1] * ih2;
			}
	}
	/********************************************************************/
	//invm3=1/3*invm0+2/3*invm2+2/3*dt*RHS(invm2)
	KernelAdvect Step3(sim, 2.0/3.0, UINF[0], UINF[1]);
	cubism::compute<VectorLab>(Step3, sim.tmpV2, sim.tmpV);
	//compute invm2 and save it in invm
	#pragma omp parallel for
	for (size_t i = 0; i < Nblocks; i++)
	{
		VectorBlock & __restrict__ INVM = *(VectorBlock*)invmInfo[i].ptrBlock;
		VectorBlock & __restrict__ INVM2 = *(VectorBlock*)tmpV2Info[i].ptrBlock;
		const VectorBlock & __restrict__ tmpV = *(VectorBlock*)tmpVInfo[i].ptrBlock;
		const Real ih2 = 1.0 / (velInfo[i].h*velInfo[i].h);
		for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
			for (int ix = 0; ix < VectorBlock::sizeX; ++ix)
			{
				INVM(ix, iy).u[0] = 1.0/3.0*INVM(ix, iy).u[0] + 2.0/3.0*INVM2(ix, iy).u[0] + tmpV(ix, iy).u[0] * ih2;
				INVM(ix, iy).u[1] = 1.0/3.0*INVM(ix, iy).u[1] + 2.0 / 3.0*INVM2(ix, iy).u[1]+tmpV(ix,iy).u[1]*ih2;								}
	}																				/********************************************************************/
	sim.stopProfiler();
}

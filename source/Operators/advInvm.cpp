#include "advInvm.h"
#include <eigen3/Eigen/Dense>
#include "../Shape.h"
using namespace cubism;

#ifdef CUP2D_PRESERVE_SYMMETRY
#define CUP2D_DISABLE_OPTIMIZATIONS __attribute__((optimize("-O1")))
#else
#define CUP2D_DISABLE_OPTIMIZATIONS
#endif

CUP2D_DISABLE_OPTIMIZATIONS

namespace cubism {

template <typename Lab1,typename Lab2, typename Kernel, typename TGrid1, typename TGrid2>
void compute2(Kernel &&kernel, TGrid1 *g, TGrid2 *f)
{
   	cubism::SynchronizerMPI_AMR<typename TGrid1::Real,TGrid1>& Synch1 = *(g->sync(kernel));
	cubism::SynchronizerMPI_AMR<typename TGrid2::Real,TGrid2>& Synch2 = *(f->sync(kernel));
    std::vector<cubism::BlockInfo*> *inner1 = &Synch1.avail_inner();
	std::vector<cubism::BlockInfo*> *inner2 = &Synch2.avail_inner();
    std::vector<cubism::BlockInfo*> *halo1 = &Synch1.avail_halo();
	std::vector<cubism::BlockInfo*> *halo2 = &Synch2.avail_halo();
    #pragma omp parallel
    {
        Lab1 lab1;
        lab1.prepare(*g, Synch1);

		Lab2 lab2;
		lab2.prepare(*f,Synch2);

        #pragma omp for nowait
		for(size_t i=0;i<inner1->size();i++){
			const cubism::BlockInfo *I1=inner1->at(i);
			const cubism::BlockInfo *I2=inner2->at(i);
			lab1.load(*I1,0);
			lab2.load(*I2,0);
			kernel(lab1,*I1,lab2,*I2);
		}

        #pragma omp barrier

        #pragma omp for nowait
        for (size_t i=0;i<halo1->size();i++)
        {
          const cubism::BlockInfo *I1=halo1->at(i);
		  const cubism::BlockInfo *I2=halo2->at(i);
		  lab1.load(*I1, 0);
		  lab2.load(*I2, 0);
          kernel(lab1, *I1,lab2,*I2);
        }
    }
}
}
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
	KernelAdvect(const SimulationData & s, const Real c, const Real uinfx, const Real uinfy,const int shapeid_) : sim(s), coef(c),shapeid(shapeid_)
	{
		uinf[0] = uinfx;
		uinf[1] = uinfy;
	}
	const SimulationData & sim;
	const Real coef;
	Real uinf[2];
	const int shapeid;
	const StencilInfo stencil{ -3, -3, 0, 4, 4, 1, true, {0,1} };
	const std::vector<cubism::BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();
	const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo();
	//const std::vector<cubism::BlockInfo>& chiInfo = sim.chi->getBlocksInfo();
	void operator()(VectorLab& lab, const BlockInfo& info) const
	{
		std::vector<ObstacleBlock*>& OBLOCK = sim.Eshapes[shapeid]->obstacleBlocks;
		if(OBLOCK[info.blockID]==nullptr) return;
		const Real h = info.h;
		const Real afac = -sim.dt*h;
		VectorBlock & __restrict__ TMP = *(VectorBlock*)tmpVInfo[info.blockID].ptrBlock;
		VectorBlock & __restrict__ V = *(VectorBlock*)velInfo[info.blockID].ptrBlock;
		//ScalarBlock & __restrict__ CHI = *(ScalarBlock*)chiInfo[info.blockID].ptrBlock;
		for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
			for (int ix = 0; ix < VectorBlock::sizeX; ++ix)
			{
				const Real V0[2]={V(ix,iy).u[0],V(ix,iy).u[1]};
				TMP(ix, iy).u[0] = coef * dINVMX_adv(lab, V0, uinf, afac, ix, iy);
				TMP(ix, iy).u[1] = coef * dINVMY_adv(lab, V0, uinf, afac, ix, iy);
			}
	}
};
struct KernelMark
{
	KernelMark(const SimulationData & s,const int shapeid_):sim(s),shapeid(shapeid_)
	{}
	const SimulationData & sim;
	const StencilInfo stencil{-1,-1,0,2,2,1,true,{0,1}};
	const int shapeid;
	const std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
	void operator()(ScalarLab& lab, const BlockInfo& info) const
	{
		std::vector<ObstacleBlock*>& OBLOCK = sim.Eshapes[shapeid]->obstacleBlocks;
		if(OBLOCK[info.blockID]==nullptr) return;
		ScalarBlock & __restrict__ TMP = *(ScalarBlock*)tmpInfo[info.blockID].ptrBlock;
		for (int iy=0;iy<VectorBlock::sizeY;++iy)
		for (int ix=0;ix<VectorBlock::sizeX;++ix){
			if(lab(ix,iy).s==shapeid+1) continue;
			if(sim.step==87&&info.level!=sim.levelMax-1&&(ix==0||iy==0))
			std::cout<<lab(ix-1,iy).s<<","<<lab(ix,iy-1).s<<std::endl;
			if(lab(ix-1,iy).s==shapeid+1
			 ||lab(ix+1,iy).s==shapeid+1
			 ||lab(ix,iy-1).s==shapeid+1
			 ||lab(ix,iy+1).s==shapeid+1) 
			 TMP(ix,iy).s=shapeid+200000;
		}
	}
};
struct Kernelextrapolate
{
	Kernelextrapolate(const SimulationData & s,const int r,const int shapeid_) : sim(s), radius(r),shapeid(shapeid_)
	{}
	const SimulationData & sim;
	const StencilInfo stencil{-4, -4, 0, 5, 5, 1, true, {0,1}};
	const int radius,shapeid;
	const std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
	const std::vector<cubism::BlockInfo>& invmInfo = sim.invms[shapeid]->getBlocksInfo();
	const std::vector<cubism::BlockInfo>& SinvmInfo = sim.invm->getBlocksInfo();
	void operator()(VectorLab& lab,const BlockInfo& info,ScalarLab& TagLab,const BlockInfo& info2) const
	{
		std::vector<ObstacleBlock*>& OBLOCK = sim.Eshapes[shapeid]->obstacleBlocks;
		if(OBLOCK[info.blockID]==nullptr) return;
		ScalarBlock & __restrict__ TMP = *(ScalarBlock*)tmpInfo[info.blockID].ptrBlock;
		VectorBlock & __restrict__ INVM = *(VectorBlock*)invmInfo[info.blockID].ptrBlock;
		VectorBlock & __restrict__ SINVM = *(VectorBlock*)SinvmInfo[info.blockID].ptrBlock;
		for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
		for (int ix = 0; ix < VectorBlock::sizeX; ++ix)
		{
			//decide point (ix,iy) is a point adjacent to the valid inverse map
			if (TagLab(ix,iy).s!=shapeid+200000) continue;
			//collect points with known inverse map value within the radius 
			std::vector<Real> bufA((2*radius+1)*(2*radius+1)*3);
			std::vector<Real> bufb((2 * radius + 1)*(2 * radius + 1)*2);
			int num_ngbs = 0;
			for (int i = -radius; i <= radius; i++) 
			for (int j = -radius; j <= radius; j++) {
				if (TagLab(ix + i, iy + j).s ==shapeid+1) {
					bufA[num_ngbs * 3] = Real(i);
					bufA[num_ngbs * 3 + 1] = Real(j);
					bufA[num_ngbs * 3 + 2] = 1.0;
					bufb[num_ngbs*2] = lab(ix + i, iy + j).u[0];
					bufb[num_ngbs * 2+1] = lab(ix + i, iy + j).u[1];
					num_ngbs ++;
				}
			}
			//assemble least square matrix Ax=b
			Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A(bufA.data(), num_ngbs, 3);
			Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> b(bufb.data(), num_ngbs, 2);
			//solve a,b,c in inv_m(x,y)=ax+by+c
			Eigen::Matrix<Real, 3, 2> x =  (A.transpose() * A).ldlt().solve(A.transpose() * b);
			//update the inverse map at (ix,iy)
			INVM(ix, iy).u[0] = x(2, 0);
			INVM(ix, iy).u[1] = x(2, 1);
			SINVM(ix,iy).u[0] = max(SINVM(ix,iy).u[0],x(2,0));
			SINVM(ix,iy).u[1] = max(SINVM(ix,iy).u[1],x(2,1));
			TMP(ix,iy).s=shapeid+1;
			}
	}
};

void advInvm::advect(const Real dt)
{
	const size_t Nblocks = velInfo.size();
	const Real UINF[2] = { sim.uinfx, sim.uinfy };
	for (size_t shapeid=0;shapeid<sim.Eshapes.size();shapeid++) {
		/********************************************************************/
		//put invm of the shape to the global grid
		std::vector<ObstacleBlock*>& OBLOCK = sim.Eshapes[shapeid]->obstacleBlocks;
		const std::vector<BlockInfo>& localinvmInfo = sim.invms[shapeid]->getBlocksInfo();
		/*#pragma omp parallel for
		for (size_t i = 0; i < Nblocks; i++)
		{
			if(OBLOCK[invmInfo[i].blockID]=nullptr) continue;

			const VectorBlock & __restrict__ LOCALINVM = *(VectorBlock*)localinvmInfo[i].ptrBlock;
			VectorBlock & __restrict__ INVM = *(VectorBlock*)invmInfo[i].ptrBlock;
			for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
			for (int ix = 0; ix < VectorBlock::sizeX; ++ix)
			{
				if(LOCALINVM(ix,iy).u[0]!=0.0||LOCALINVM(ix,iy).u[1]!=0.0){
					INVM(ix, iy).u[0] = LOCALINVM(ix,iy).u[0];
					INVM(ix, iy).u[1] = LOCALINVM(ix,iy).u[1];
				}
			}
		}*/
		/********************************************************************/
		//invm1=invm0+dt*RHS(invm0)
		KernelAdvect Step1(sim,1, UINF[0], UINF[1],shapeid);
		cubism::compute<VectorLab>(Step1, sim.invms[shapeid].get(), sim.tmpV);
		// compute invm1 and save it in tmpV1
		#pragma omp parallel for
		for (size_t i = 0; i < Nblocks; i++)
		{
			const VectorBlock & __restrict__ INVM0 = *(VectorBlock*)localinvmInfo[i].ptrBlock;
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
		KernelAdvect Step2(sim, 0.25, UINF[0], UINF[1],shapeid);
		cubism::compute<VectorLab>(Step2, sim.tmpV1, sim.tmpV);
		//compute invm2 and save it in tmpV2
		#pragma omp parallel for
		for (size_t i = 0; i < Nblocks; i++)
		{
			VectorBlock & __restrict__ INVM0 = *(VectorBlock*)localinvmInfo[i].ptrBlock;
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
		KernelAdvect Step3(sim, 2.0/3.0, UINF[0], UINF[1],shapeid);
		cubism::compute<VectorLab>(Step3, sim.tmpV2, sim.tmpV);
		//compute invm2 and save it in invm
		#pragma omp parallel for
		for (size_t i = 0; i < Nblocks; i++)
		{
			//( (ScalarBlock*)  tmpInfo[i].ptrBlock )->set(-1);//clear tmp for next step;
			if(OBLOCK[localinvmInfo[i].blockID]==nullptr) continue;
			//const CHI_MAT & __restrict__ X = OBLOCK[localinvmInfo[i].blockID]->chi;
			VectorBlock & __restrict__ SINVM = *(VectorBlock*)invmInfo[i].ptrBlock;
			VectorBlock & __restrict__ INVM = *(VectorBlock*)localinvmInfo[i].ptrBlock;
			VectorBlock & __restrict__ INVM2 = *(VectorBlock*)tmpV2Info[i].ptrBlock;
			const VectorBlock & __restrict__ tmpV = *(VectorBlock*)tmpVInfo[i].ptrBlock;
			const Real ih2 = 1.0 / (velInfo[i].h*velInfo[i].h);
			for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
			for (int ix = 0; ix < VectorBlock::sizeX; ++ix)
			{
				//only update invm of points inside sim.shapes[shapeid]
				if(sim.Eshapes[shapeid]->isinside(INVM(ix,iy).u[0],INVM(ix,iy).u[1])){
					INVM(ix, iy).u[0] = 1.0/3.0*INVM(ix, iy).u[0] + 2.0/3.0*INVM2(ix, iy).u[0] + tmpV(ix, iy).u[0] * ih2;
					INVM(ix, iy).u[1] = 1.0/3.0*INVM(ix, iy).u[1] + 2.0 / 3.0*INVM2(ix, iy).u[1]+tmpV(ix,iy).u[1]*ih2;
					SINVM(ix,iy).u[0] = max(SINVM(ix,iy).u[0],INVM(ix,iy).u[0]);
					SINVM(ix,iy).u[1] = max(SINVM(ix,iy).u[1],INVM(ix,iy).u[1]);
				}	
				else	{INVM(ix, iy).u[0]=0.0;INVM(ix, iy).u[1]=0.0;}
			}
		}
	}
}
void advInvm::extrapolate(const int layers)
{
	const size_t Nblocks = velInfo.size();


	/*******************************************************************/
	//record points belongs to shape[i] with i on tmp
	/*for (size_t shapeid=0;shapeid<sim.Eshapes.size();shapeid++) {
		auto shape=sim.Eshapes[shapeid];
		const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
		#pragma omp parallel for
		for (size_t i = 0; i < Nblocks; i++){
			VectorBlock & __restrict__ INVM = *(VectorBlock*)invmInfo[i].ptrBlock;
        	VectorBlock & __restrict__ TMPV1 = *(VectorBlock*)tmpV1Info[i].ptrBlock;TMPV1.clear();
			ScalarBlock & __restrict__ TMP = *(ScalarBlock*)tmpInfo[i].ptrBlock;
			if(OBLOCK[invmInfo[i].blockID]==nullptr) continue;
			ObstacleBlock& o = * OBLOCK[invmInfo[i].blockID];
      		UDEFMAT & __restrict__ Oinvm = o.invm;
			const CHIMAT & __restrict__ X = o.chi;
			for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
        	for (int ix = 0; ix < VectorBlock::sizeX; ++ix){
				if(X[iy][ix]==1.0)
				{
					TMP(ix,iy).s=shapeid;
					//clear Oinvm for each shape for extrapolation
					Oinvm[iy][ix][0]=INVM(ix,iy).u[0];
					Oinvm[iy][ix][0]=INVM(ix,iy).u[1];
				}
				TMPV1(ix,iy).u[0]=INVM(ix,iy).u[0];
				TMPV1(ix,iy).u[1]=INVM(ix,iy).u[1];
			}
		}
	}*/
	/********************************************************************/
	//extrapolate fixed layers and store in invm of each shape
	for(size_t id=0;id<sim.Eshapes.size();id++){
		// 1) mark points of shape id with tag:id+1
		auto shape=sim.Eshapes[id];
		const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
		const std::vector<BlockInfo>& localinvmInfo = sim.invms[id]->getBlocksInfo();
		#pragma omp parallel for
		for (size_t i = 0; i < Nblocks; i++){
			if(OBLOCK[invmInfo[i].blockID]==nullptr) continue;
			ScalarBlock & __restrict__ TMP = *(ScalarBlock*)tmpInfo[i].ptrBlock;TMP.clear();
			VectorBlock & __restrict__ INVM =*(VectorBlock*)localinvmInfo[i].ptrBlock;
			for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
        	for (int ix = 0; ix < VectorBlock::sizeX; ++ix)
				if(INVM(ix,iy).u[0]!=0||INVM(ix,iy).u[1]!=0) TMP(ix,iy).s=id+1;
		}
		if (sim.step==87)
		sim.dumptmpDebug("markfirst");

		// 2) extrapolate 
		KernelMark Mark(sim,id);
		Kernelextrapolate extrapolate(sim,4,id);
		for(int t=0;t<layers;t++){
			cubism::compute<ScalarLab>(Mark,sim.tmp);
			compute2<VectorLab,ScalarLab>(extrapolate,sim.invms[id].get(),sim.tmp);
		}
	}
}
void advInvm::operator()(const Real dt)
{
	sim.startProfiler("advInvm");
	const size_t Nblocks = velInfo.size();
	#pragma omp parallel for
  	for (size_t i=0; i < Nblocks; i++)
    	( (VectorBlock*)  invmInfo[i].ptrBlock )->clear();
	advect(dt);
	extrapolate(5);
	sim.stopProfiler();
}

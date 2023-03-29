//
//  CubismUP_2D
//  Copyright (c) 2023 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "SmartNaca.h"

using namespace cubism;

struct GradScalarOnTmpV
{
  GradScalarOnTmpV(const SimulationData & s) : sim(s) {}
  const SimulationData & sim;
  const StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0}};
  const std::vector<cubism::BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();
  void operator()(ScalarLab & lab, const BlockInfo& info) const
  {
    auto & __restrict__ TMPV = *(VectorBlock*) tmpVInfo[info.blockID].ptrBlock;
    const Real ih = 0.5/info.h;
    for(int y=0; y<ScalarBlock::sizeY; ++y)
    for(int x=0; x<ScalarBlock::sizeX; ++x)
    {
      TMPV(x,y).u[0] = ih * (lab(x+1,y).s-lab(x-1,y).s);
      TMPV(x,y).u[1] = ih * (lab(x,y+1).s-lab(x,y-1).s);
    }
  }
};

static Real M6(const Real x)
{
  if      ( x < 0.5 ) return 0.5*(x+1.5)*(x+1.5)-1.5*(x+0.5)*(x+0.5);
  else if ( x < 1.5 ) return 0.5*(1.5-x)*(1.5-x);
  return 0.;
}

SmartNaca::SmartNaca(SimulationData&s, ArgumentParser&p, Real C[2]): Naca(s,p,C), Nactuators ( p("-Nactuators").asInt(2)),actuator_ds( p("-actuatords").asDouble(0.05)),thickness(p("-tRatio").asDouble(0.12))
{
  actuators.resize(Nactuators,0.);
  actuatorSchedulers.resize(Nactuators);
  actuators_prev_value.resize(Nactuators);
  actuators_next_value.resize(Nactuators);
}

void SmartNaca::finalize()
{
  //dummy actuator values for testing
  #if 0
  static bool visited = false;
  if (sim.time > 2.0 && visited == false)
  {
    visited = true;
    std::vector<Real> q(actuators.size());
    for (int i = 0 ; i < (int)actuators.size(); i ++) q[i] = 0.25*(2*(i+1)%2-1);
    q[0] =  0.5;
    q[1] = -0.25;
    act(q,0);
  }
  #endif

  //transition from one actuator value to the next
  const Real transition_duration = 1.0;
  Real tot = 0.0;
  for (size_t idx = 0 ; idx < actuators.size(); idx++)
  {
    Real dummy;
    actuatorSchedulers[idx].transition (sim.time,t_change,t_change+transition_duration,actuators_prev_value[idx],actuators_next_value[idx]);
    actuatorSchedulers[idx].gimmeValues(sim.time,actuators[idx],dummy);
    tot += std::fabs(actuators[idx]);
  }

  //used for reward function
  const Real cd = forcex / (0.5*u*u*thickness);
  fx_integral += -cd*sim.dt; //-std::fabs(cd)*sim.dt;

  //if actuators are zero don't do anything
  if (tot < 1e-21) return;

  //Compute gradient of chi and of signed-distance-function here.
  //Used later for the actuators
  const std::vector<cubism::BlockInfo>& tmpInfo  = sim.tmp ->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();
  const size_t Nblocks = tmpVInfo.size();
  const int Ny = ScalarBlock::sizeY;
  const int Nx = ScalarBlock::sizeX;

  //store grad(chi) in a vector and grad(SDF) in tmpV
  cubism::compute<ScalarLab>(GradScalarOnTmpV(sim),sim.chi);
  std::vector<double> gradChi(ScalarBlock::sizeY*ScalarBlock::sizeX*Nblocks*2);
  #pragma omp parallel for
  for (size_t i = 0 ; i < Nblocks; i++)
  {
    auto & __restrict__ TMP  = *(ScalarBlock*) tmpInfo [i].ptrBlock;
    auto & __restrict__ TMPV = *(VectorBlock*) tmpVInfo[i].ptrBlock;
    for(int iy=0; iy<Ny; iy++)
    for(int ix=0; ix<Nx; ix++)
    {
      const size_t idx = i*Ny*Nx + iy*Nx + ix;
      gradChi[2*idx+0] = TMPV(ix,iy).u[0];
      gradChi[2*idx+1] = TMPV(ix,iy).u[1];
      TMP(ix,iy).s = 0;
    }
    if(obstacleBlocks[i] == nullptr) continue; //obst not in block
    ObstacleBlock& o = * obstacleBlocks[i];
    const auto & __restrict__ SDF  = o.dist;
    for(int iy=0; iy<ScalarBlock::sizeY; iy++)
    for(int ix=0; ix<ScalarBlock::sizeX; ix++)
    {
      TMP(ix,iy).s = SDF[iy][ix];
    }
  }
  cubism::compute<ScalarLab>(GradScalarOnTmpV(sim),sim.tmp);

  const Real * const rS = myFish->rS;
  const Real * const rX = myFish->rX;
  const Real * const rY = myFish->rY;
  const Real * const norX = myFish->norX;
  const Real * const norY = myFish->norY;
  const Real * const width = myFish->width;
  std::vector<int>       ix_store;
  std::vector<int>       iy_store;
  std::vector<long long> id_store;
  std::vector<Real>      nx_store;
  std::vector<Real>      ny_store;
  std::vector<Real>      cc_store;
  std::vector<int>      idx_store;
  Real surface   = 0.0;
  Real surface_c = 0.0;
  Real mass_flux = 0.0;
  #pragma omp parallel for reduction(+: surface,surface_c,mass_flux)
  for (const auto & info : sim.vel->getBlocksInfo())
  {
    if(obstacleBlocks[info.blockID] == nullptr) continue; //obst not in block
    ObstacleBlock& o = * obstacleBlocks[info.blockID];
    auto & __restrict__ UDEF = o.udef;
    const auto & __restrict__ SDF  = o.dist;
    const Real h2 = info.h*info.h;
    auto & __restrict__ TMPV = *(VectorBlock*) tmpVInfo[info.blockID].ptrBlock;

    for(int iy=0; iy<ScalarBlock::sizeY; iy++)
    for(int ix=0; ix<ScalarBlock::sizeX; ix++)
    {
      if ( SDF[iy][ix] > info.h || SDF[iy][ix] < -info.h) continue;
      UDEF[iy][ix][0] = 0.0;
      UDEF[iy][ix][1] = 0.0;
      Real p[2];
      info.pos(p, ix, iy);

      //find closest surface point to analytical expression
      int  ss_min = 0;
      int  sign_min = 0;
      Real dist_min = 1e10;
      for (int ss = 0 ; ss < myFish->Nm; ss++)
      {
        Real Pp [2] = {rX[ss]+width[ss]*norX[ss],rY[ss]+width[ss]*norY[ss]};
        Real Pm [2] = {rX[ss]-width[ss]*norX[ss],rY[ss]-width[ss]*norY[ss]};
        const Real dp = pow(Pp[0]-p[0],2)+pow(Pp[1]-p[1],2);
        const Real dm = pow(Pm[0]-p[0],2)+pow(Pm[1]-p[1],2);
        if (dp < dist_min)
        {
          sign_min = 1;
          dist_min = dp;
          ss_min = ss;
        }
        if (dm < dist_min)
        {
          sign_min = -1;
          dist_min = dm;
          ss_min = ss;
        }
      }

      const Real smax = rS[myFish->Nm-1]-rS[0];
      const Real ds   = 2*smax/Nactuators;
      const Real current_s = rS[ss_min];
      if (current_s < 0.01*length || current_s > 0.99*length) continue;

      int idx = (current_s / ds); //this is the closest actuator
      const Real s0 = 0.5*ds + idx * ds;
      if (sign_min == -1) idx += Nactuators/2;

      if (std::fabs( current_s - s0 ) < 0.5*actuator_ds*length)
      {
        const size_t index = 2*(info.blockID*Ny*Nx+iy*Nx+ix);
        const Real dchidx = gradChi[index  ];
        const Real dchidy = gradChi[index+1];
        Real nx = TMPV(ix,iy).u[0];
        Real ny = TMPV(ix,iy).u[1];
        const Real nn = pow(nx*nx+ny*ny+1e-21,-0.5);
        nx *= nn;
        ny *= nn;
        const Real c0 = std::fabs(current_s - s0)/ (0.5*actuator_ds*length);
        const Real c = 1.0 - c0*c0;
        UDEF[iy][ix][0] = c*actuators[idx]*nx;
        UDEF[iy][ix][1] = c*actuators[idx]*ny;
        #pragma omp critical
        {
          ix_store.push_back(ix);
          iy_store.push_back(iy);
          id_store.push_back(info.blockID);
          nx_store.push_back(nx);
          ny_store.push_back(ny);
          cc_store.push_back(c);
          idx_store.push_back(idx);
        }
        const Real fac = (dchidx*nx+dchidy*ny)*h2;
        mass_flux += fac*(c*actuators[idx]);
        surface   += fac;
        surface_c += fac*c;
      }
    }
  }

  Real Qtot [3] = {mass_flux,surface,surface_c};
  MPI_Allreduce(MPI_IN_PLACE,Qtot,3,MPI_Real,MPI_SUM,sim.comm);
  //const Real uMean = Qtot[0]/Qtot[1];
  const Real q = Qtot[0]/Qtot[2];

  //Substract total mass flux (divided by surface) from actuator velocities
  #pragma omp parallel for
  for (size_t idx = 0 ; idx < id_store.size(); idx++)
  {
    const long long blockID = id_store[idx];
    const int ix            = ix_store[idx];
    const int iy            = iy_store[idx];
    const int idx_st        =idx_store[idx];
    const Real nx           = nx_store[idx];
    const Real ny           = ny_store[idx];
    const Real c            = cc_store[idx];
    ObstacleBlock& o = * obstacleBlocks[blockID];
    auto & __restrict__ UDEF = o.udef;
    UDEF[iy][ix][0] = c*(actuators[idx_st]-q)*nx;
    UDEF[iy][ix][1] = c*(actuators[idx_st]-q)*ny;
  }
}

void SmartNaca::act( std::vector<Real> action, const int agentID)
{
  t_change = sim.time;
  if(action.size() != actuators.size())
  {
    std::cerr << "action size needs to be equal to actuators\n";
    fflush(0);
    abort();
  }
  for (size_t i = 0 ; i < action.size() ; i ++)
  {
    actuators_prev_value[i] = actuators[i];
    actuators_next_value[i] = action   [i];
  }
}

Real SmartNaca::reward(const int agentID)
{
  Real retval = fx_integral; // divided by dt=1.0, the time between actions
  fx_integral = 0;
  Real regularizer = 0.0;
  for (size_t idx = 0 ; idx < actuators.size(); idx++)
  {
    regularizer += actuators[idx]*actuators[idx];
  }
  regularizer = pow(regularizer,0.5)/actuators.size();
  return retval - 0.1*regularizer;
}

std::vector<Real> SmartNaca::state(const int agentID)
{
   const int nx = 16;
   const int ny = 8;
   const double cx = centerOfMass[0];
   const double cy = centerOfMass[1];
   const double ex = length*2;
   const double ey = length;
   const double hx = ex / nx;
   const double hy = ey / ny;

   std::vector<double> ux (nx*ny);
   std::vector<double> uy (nx*ny);
   std::vector<double> pr (nx*ny);
   std::vector<double> vol(nx*ny);

   const auto & vInfo = sim.vel->getBlocksInfo();
   const auto & pInfo = sim.pres->getBlocksInfo();
   for(size_t i=0; i<vInfo.size(); i++)
   {
     const VectorBlock & V  = *(VectorBlock*) vInfo[i].ptrBlock;
     const ScalarBlock & p  = *(ScalarBlock*) pInfo[i].ptrBlock;
     for(int iy=0; iy<ScalarBlock::sizeY; iy++)
     for(int ix=0; ix<ScalarBlock::sizeX; ix++)
     {
       double pp[2];
       vInfo[i].pos(pp, ix, iy);
       pp[0] -= cx;
       pp[1] -= cy;
       for (int jy = 0 ; jy < ny; jy ++)
       for (int jx = 0 ; jx < nx; jx ++)
       {
            const double x = -0.5*ex + 0.5*hx + jx*hx;
            const double y = -0.5*ey + 0.5*hy + jy*hy;
            const double dx = std::fabs(x-pp[0])/hx;
            const double dy = std::fabs(y-pp[1])/hy;
            const double coef = M6(dx)*M6(dy);
            const int idx = jx + jy*nx;
            ux [idx]+=coef*V(ix,iy).u[0];
            uy [idx]+=coef*V(ix,iy).u[1];
            pr [idx]+=coef*p(ix,iy).s;
            vol[idx]+=coef;
       }
     }
   }
   MPI_Allreduce(MPI_IN_PLACE,ux.data(),ux.size(),MPI_DOUBLE,MPI_SUM,sim.comm);
   MPI_Allreduce(MPI_IN_PLACE,uy.data(),uy.size(),MPI_DOUBLE,MPI_SUM,sim.comm);
   MPI_Allreduce(MPI_IN_PLACE,pr.data(),pr.size(),MPI_DOUBLE,MPI_SUM,sim.comm);
   MPI_Allreduce(MPI_IN_PLACE,vol.data(),vol.size(),MPI_DOUBLE,MPI_SUM,sim.comm);

   std::vector<double> S;
   for (int idx = 0 ; idx < nx*ny; idx ++)
   {
      S.push_back(ux[idx]/(vol[idx]+1e-15));
      S.push_back(uy[idx]/(vol[idx]+1e-15));
      S.push_back(pr[idx]/(vol[idx]+1e-15));
   }
   return S;

#if 0
  std::vector<Real> S;
  const int bins = 64;

  const Real dtheta = 2.*M_PI / bins;
  std::vector<int>   n_s   (bins,0.0);
  std::vector<Real>  p_s   (bins,0.0);
  std::vector<Real> fX_s   (bins,0.0);
  std::vector<Real> fY_s   (bins,0.0);
  for(auto & block : obstacleBlocks) if(block not_eq nullptr)
  {
    for(size_t i=0; i<block->n_surfPoints; i++)
    {
      const Real x     = block->x_s[i] - origC[0];
      const Real y     = block->y_s[i] - origC[1];
      const Real ang   = atan2(y,x);
      const Real theta = ang < 0 ? ang + 2*M_PI : ang;
      const Real p     = block->p_s[i];
      const Real fx    = block->fX_s[i];
      const Real fy    = block->fY_s[i];
      const int idx = theta / dtheta;
      n_s [idx] ++;
      p_s [idx] += p;
      fX_s[idx] += fx;
      fY_s[idx] += fy;
    }
  }

  MPI_Allreduce(MPI_IN_PLACE,n_s.data(),n_s.size(),MPI_INT ,MPI_SUM,sim.comm);
  for (int idx = 0 ; idx < bins; idx++)
  {
    p_s [idx] /= n_s[idx];
    fX_s[idx] /= n_s[idx];
    fY_s[idx] /= n_s[idx];
  }

  for (int idx = 0 ; idx < bins; idx++) S.push_back( p_s[idx]);
  for (int idx = 0 ; idx < bins; idx++) S.push_back(fX_s[idx]);
  for (int idx = 0 ; idx < bins; idx++) S.push_back(fY_s[idx]);
  MPI_Allreduce(MPI_IN_PLACE,  S.data(),  S.size(),MPI_Real,MPI_SUM,sim.comm);
  S.push_back(forcex);
  S.push_back(forcey);
  S.push_back(torque);

  if (sim.rank ==0 )
    for (size_t i = 0 ; i < S.size() ; i++) std::cout << S[i] << " ";

  return S;
#endif
}

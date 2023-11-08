//
//  CubismUP_2D
//  Copyright (c) 2023 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "SmartNaca.h"

using namespace cubism;

//This function will compute the gradient of a scalar and save the result to the tmpV grid object.
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

//static Real M6(const Real x)
//{
//  if      ( x < 0.5 ) return 0.5*(x+1.5)*(x+1.5)-1.5*(x+0.5)*(x+0.5);
//  else if ( x < 1.5 ) return 0.5*(1.5-x)*(1.5-x);
//  return 0.;
//}

//Class constructor with default values for some parameters.
SmartNaca::SmartNaca(SimulationData& s, ArgumentParser& p, Real C[2]): 
	Naca(s,p,C), 
	Nactuators ( p("-Nactuators").asInt(2)),
	actuator_ds( p("-actuatords").asDouble(0.05)),
	thickness(p("-tRatio").asDouble(0.12)), 
	regularizer( p("-regularizer").asDouble(0.0)),
	value1(p("-value1").asDouble(0.0)), 
	value2(p("-value2").asDouble(0.0)), 
	value3(p("-value3").asDouble(0.0)), 
	value4(p("-value4").asDouble(0.0))
{
  actuators.resize(Nactuators,0.);
  actuatorSchedulers.resize(Nactuators);
  actuators_prev_value.resize(Nactuators);
  actuators_next_value.resize(Nactuators);
  if (Nactuators == 4){
	  act({value1,value2,value3,value4},0);
  }
}

//
//Called at every timestep, to impose the actuation velocities.
//For each grid point, we do the following:
// (A) Check if that grid point lies on the airfoil surface.
// (B) If it does:
// 	(i): Check if it lies inside an actuator.
// 	(ii): It it does:
// 		(a): Compute the local normal vector.
// 		(b): Compute the actuation velocity magnitude
// 		(c): Impose actuation velocity, based on normal vector direction and actuation velocity magnitude.
//
//
//The computation of the normal vector uses the signed distance function (SDF).
//The SDF is the distance of a point from the airfoil's surface, with the addition of a +- sign if the point
//lies inside or outside of the airfoil.
//The normal vector is equal to the gradient of the SDF: n = grad(SDF) and it is computed with the help of 'GradScalarOnTmpV'.
//
//The imposed actuation velocities are required to have a total mass flux of zero, so that no fluid is added/removed
//from the simulation domain. To achieve this, we compute their total mass flux (i.e. surface integral of normal velocity)
//and substract it from the current velocities.
//The computation of the surface integral uses the SDF and the 'chi' scalar function.
//The chi field is equal to 0 for grid points with a fluid and is equal to 1 for grid points with
//an object (the airfoil); grid point on the airfoil surface have a chi value between 0 and 1.
//
//It can be shown that the surface integral of the normal velocity is equal to:
//
// Q = integral {normal_velocity dS } = integral { normal_velocity * (nx* dchi/dx + ny*dchi/dy) dA }
//
//where (nx,ny) is the normal vector and dA means we are taking the volume integral over the simulation domain.
//

void SmartNaca::finalize()
{
  //Smooth transition from one actuator value to the next
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
  fx_integral += -cd*sim.dt;

  //if actuators are zero don't do anything
  if (tot < 1e-21) return;

  //Compute gradient of chi and of signed distance function here.
  //Used later for the actuators
  //store grad(chi) in a vector and grad(SDF) in tmpV
  const std::vector<cubism::BlockInfo>& tmpInfo  = sim.tmp ->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();
  cubism::compute<ScalarLab>(GradScalarOnTmpV(sim),sim.chi); //Compute grad(chi) and put it on 'tmpV'
  const size_t Nblocks = tmpVInfo.size();
  const int Ny = ScalarBlock::sizeY;
  const int Nx = ScalarBlock::sizeX;
  std::vector<double> gradChi(ScalarBlock::sizeY*ScalarBlock::sizeX*Nblocks*2);

  //Loop over the blocks of the grid
  #pragma omp parallel for
  for (size_t i = 0 ; i < Nblocks; i++)
  {
    auto & __restrict__ TMP  = *(ScalarBlock*) tmpInfo [i].ptrBlock;
    auto & __restrict__ TMPV = *(VectorBlock*) tmpVInfo[i].ptrBlock;

    //Loop over the grid points of each block and store grad(chi) from tmpV to the gradChi vector
    for(int iy=0; iy<Ny; iy++)
    for(int ix=0; ix<Nx; ix++)
    {
      const size_t idx = i*Ny*Nx + iy*Nx + ix;
      gradChi[2*idx+0] = TMPV(ix,iy).u[0];
      gradChi[2*idx+1] = TMPV(ix,iy).u[1];
      TMP(ix,iy).s = 0;
    }

    //Check if the current block contains any part of the airfoil. If it does, place the (already known) SDF
    //to the tmp grid
    if(obstacleBlocks[i] == nullptr) continue; //obst not in block
    ObstacleBlock& o = * obstacleBlocks[i];
    const auto & __restrict__ SDF  = o.dist;
    for(int iy=0; iy<ScalarBlock::sizeY; iy++)
    for(int ix=0; ix<ScalarBlock::sizeX; ix++)
    {
      TMP(ix,iy).s = SDF[iy][ix];
    }
  }

  cubism::compute<ScalarLab>(GradScalarOnTmpV(sim),sim.tmp); //compute grad(SDF) and place it on tmpV

  //The top part of the airfoil surface is defined as:
  //
  // Sx (s) = rX(s) + width(s) * norX(s) 
  // Sy (s) = rY(s) + width(s) * norY(s) 
  //
  //The bottom part of the surface is defined as:
  //
  // Sx (s) = rX(s) - width(s) * norX(s) 
  // Sy (s) = rY(s) - width(s) * norY(s) 
  //
  //where Sx and Sy are the x- and y-coordinates of each surface point and 's' is a scalar 
  //that parametrizes the surface.
  //The following arrays define the surface of the airfoil (which is disretized into 'Nm' points).
  const Real * const rS = myFish->rS; // s parameter discrete values
  const Real * const rX = myFish->rX; // rX parameter discrete values
  const Real * const rY = myFish->rY; // rY parameter discrete values
  const Real * const norX = myFish->norX; // norX parameter discrete values
  const Real * const norY = myFish->norY; // norY parameter discrete values
  const Real * const width = myFish->width; // width parameter discrete values

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

  //Loop over blocks of the grid.
  #pragma omp parallel for reduction(+: surface,surface_c,mass_flux)
  for (const auto & info : sim.vel->getBlocksInfo())
  {

    //If the current block does not contain part of the airfoil, do nothing
    if(obstacleBlocks[info.blockID] == nullptr) continue;

    //Get the SDF and the imposed velocity arrays for the current block
    ObstacleBlock& o = * obstacleBlocks[info.blockID];
    auto & __restrict__ UDEF = o.udef;
    const auto & __restrict__ SDF  = o.dist;

    //Volume element dA used for integrals (info.h is the grid spacing of the current block)
    const Real h2 = info.h*info.h;

    //Get the tmpV array for the current block (which currently contains the gradient of the SDF)
    auto & __restrict__ TMPV = *(VectorBlock*) tmpVInfo[info.blockID].ptrBlock;

    //Loop over grid points of current block.
    for(int iy=0; iy<ScalarBlock::sizeY; iy++)
    for(int ix=0; ix<ScalarBlock::sizeX; ix++)
    {
      //If the SDF is greater than h, this means we are too far from the airfoil surface (defined from SDF=0),
      //so we do nothing.
      if ( SDF[iy][ix] > info.h || SDF[iy][ix] < -info.h) continue;

      //Set imposed velocities to zero.
      UDEF[iy][ix][0] = 0.0;
      UDEF[iy][ix][1] = 0.0;

      //Get (x,y) coordinates of current grid point (ix,iy) and put them to p[2].
      Real p[2];
      info.pos(p, ix, iy);

      //Find the point of the airfoil surface that is closese to this grid point (use analutical expression)
      //Do so by looping over all 'Nm' discrete surface points.
      int  ss_min = 0;
      int  sign_min = 0;
      Real dist_min = 1e10;
      for (int ss = 0 ; ss < myFish->Nm; ss++)
      {
        Real Pp [2] = {rX[ss]+width[ss]*norX[ss],rY[ss]+width[ss]*norY[ss]}; //top surface of airfoil
        Real Pm [2] = {rX[ss]-width[ss]*norX[ss],rY[ss]-width[ss]*norY[ss]}; //bottom surface of airfoil
        const Real dp = pow(Pp[0]-p[0],2)+pow(Pp[1]-p[1],2); //distance from top point
        const Real dm = pow(Pm[0]-p[0],2)+pow(Pm[1]-p[1],2); //distance from bottom point
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

      //Now that closest surface point is found, find the actuator index (idx) of the actuator that is closest to that point
      const Real smax = rS[myFish->Nm-1]-rS[0];
      const Real ds   = 2*smax/Nactuators;
      const Real current_s = rS[ss_min];
      if (current_s < 0.01*length || current_s > 0.99*length) continue;
      int idx = (current_s / ds); //this is the closest actuator
      const Real s0 = 0.5*ds + idx * ds;
      if (sign_min == -1) idx += Nactuators/2;

      //If grid point is inside the closest actuator, impose actuation velocity
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

  //Substract total mass flux (divided by surface) from actuator velocities, in order to get zero total mass flux.
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

//Set new values for the actuation velocities
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

//Compute the RL reward.
Real SmartNaca::reward(const int agentID)
{
  Real retval = fx_integral; // divided by dt=1.0, the time between actions
  fx_integral = 0;
  Real regularizer_sum = 0.0;
  for (size_t idx = 0 ; idx < actuators.size(); idx++)
  {
    regularizer_sum += actuators[idx]*actuators[idx];
  }
  regularizer_sum = pow(regularizer_sum,0.5)/actuators.size();
  return retval - regularizer*regularizer_sum;
}

//Compute the RL state.
std::vector<Real> SmartNaca::state(const int agentID)
{
  #if 0
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
  #else

    const Real * const rS = myFish->rS;
    const Real * const rX = myFish->rX;
    const Real * const rY = myFish->rY;
    const Real * const norX = myFish->norX;
    const Real * const norY = myFish->norY;
    const Real * const width = myFish->width;

    std::vector<Real> S;
    const int bins = 16;
    const Real bin_ds = 0.05;
    std::vector<int>   n_s   (bins,0.0);
    std::vector<Real>  p_s   (bins,0.0);
    std::vector<Real> fX_s   (bins,0.0);
    std::vector<Real> fY_s   (bins,0.0);
    for(auto & block : obstacleBlocks) if(block not_eq nullptr)
    {
      for(size_t i=0; i<block->n_surfPoints; i++)
      {
        const Real x = block->x_s[i];
        const Real y = block->y_s[i];

        //find closest surface point to analytical expression
        int  ss_min = 0;
        int  sign_min = 0;
        Real dist_min = 1e10;
        for (int ss = 0 ; ss < myFish->Nm; ss++)
        {
          Real Pp [2] = {rX[ss]+width[ss]*norX[ss],rY[ss]+width[ss]*norY[ss]};
          Real Pm [2] = {rX[ss]-width[ss]*norX[ss],rY[ss]-width[ss]*norY[ss]};
          const Real dp = pow(Pp[0]-x,2)+pow(Pp[1]-y,2);
          const Real dm = pow(Pm[0]-x,2)+pow(Pm[1]-y,2);
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
        const Real ds   = 2*smax/bins;
        const Real current_s = rS[ss_min];
        if (current_s < 0.01*length || current_s > 0.99*length) continue;

        int idx = (current_s / ds); //this is the closest actuator
        const Real s0 = 0.5*ds + idx * ds;
        if (sign_min == -1) idx += bins/2;
        if (std::fabs( current_s - s0 ) < 0.5*bin_ds*length)
        {
          const Real p  = block->p_s[i];
          const Real fx = block->fX_s[i];
          const Real fy = block->fY_s[i];
          n_s [idx] ++;
          p_s [idx] += p;
          fX_s[idx] += fx;
          fY_s[idx] += fy;
        }
      }
    }
    MPI_Allreduce(MPI_IN_PLACE,n_s.data(),n_s.size(),MPI_INT ,MPI_SUM,sim.comm);
    for (int idx = 0 ; idx < bins; idx++)
    {
      if (n_s[idx] == 0) continue;
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
    if (sim.rank ==0 )
      for (size_t i = 0 ; i < S.size() ; i++) std::cout << S[i] << " ";
    return S;
  #endif
}

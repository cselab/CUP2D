//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "Shape.h"
//#include "OperatorComputeForces.h"
#include "Utils/BufferedLogger.h"
#include <gsl/gsl_linalg.h>
#include <iomanip>
using namespace cubism;

//#define EXPL_INTEGRATE_MOM

static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
Real Shape::getMinRhoS() const { return rhoS; }
Real Shape::getCharMass() const { return 0; }
Real Shape::getMaxVel() const { return std::sqrt(u*u + v*v); }

void Shape::updateVelocity(Real dt)
{
  #ifdef EXPL_INTEGRATE_MOM
  if(not bForcedx  || sim.time > timeForced) 
    u = ( fluidMomX + dt * appliedForceX ) / penalM;
  if(not bForcedy  || sim.time > timeForced)
    v = ( fluidMomY + dt * appliedForceY ) / penalM;
  if(not bBlockang || sim.time > timeForced)
    omega = ( fluidAngMom + dt * appliedTorque ) / penalJ;
  #else  
  //A and b need to be declared as double (not Real)
  double A[3][3] = {
    { (double)  penalM, (double)      0, (double) -penalDY },
    { (double)       0, (double) penalM, (double)  penalDX },
    { (double)-penalDY, (double)penalDX, (double)  penalJ  }
  };
  double b[3] = {
    (double) (fluidMomX   + dt * appliedForceX),
    (double) (fluidMomY   + dt * appliedForceY),
    (double) (fluidAngMom + dt * appliedTorque)
  };
  
  if(bForcedx && sim.time < timeForced) {
                 A[0][1] = 0; A[0][2] = 0; b[0] = penalM * forcedu;
  }
  if(bForcedy && sim.time < timeForced) {
    A[1][0] = 0;              A[1][2] = 0; b[1] = penalM * forcedv;
  }
  if(bBlockang && sim.time < timeForced) {
    A[2][0] = 0; A[2][1] = 0;              b[2] = penalJ * forcedomega;
  }

  gsl_matrix_view Agsl = gsl_matrix_view_array (&A[0][0], 3, 3);
  gsl_vector_view bgsl = gsl_vector_view_array (b, 3);
  gsl_vector * xgsl = gsl_vector_alloc (3);
  int sgsl;
  gsl_permutation * permgsl = gsl_permutation_alloc (3);
  gsl_linalg_LU_decomp (& Agsl.matrix, permgsl, & sgsl);
  gsl_linalg_LU_solve (& Agsl.matrix, permgsl, & bgsl.vector, xgsl);

  if(not bForcedx  || sim.time > timeForced)  u     = gsl_vector_get(xgsl, 0);
  if(not bForcedy  || sim.time > timeForced)  v     = gsl_vector_get(xgsl, 1);
  if(not bBlockang || sim.time > timeForced)  omega = gsl_vector_get(xgsl, 2);

  if( bBreakSymmetry )
  {
    //if( sim.time > 1.0 && sim.time < 1.1 )
    //  omega =  u/getCharLength();
    //else if( sim.time > 1.1 && sim.time < 1.2 )
    //  omega = -u/getCharLength();
    //else
    //  omega = 0;

    if( sim.time > 1.0 && sim.time < 2.0 )
      v = 0.1*getCharLength()*std::sin(M_PI*(sim.time-2.0));
    else
      v = forcedv;
  }

  gsl_permutation_free(permgsl);
  gsl_vector_free(xgsl);
  #endif
}

void Shape::updateLabVelocity( int nSum[2], Real uSum[2] )
{
  if(bFixedx) { (nSum[0])++; uSum[0] -= u; }
  if(bFixedy) { (nSum[1])++; uSum[1] -= v; }
}

void Shape::updatePosition(Real dt)
{
  // Remember, uinf is -ubox, therefore we sum it to u body to get
  // velocity of shapre relative to the sim box
  centerOfMass[0] += dt * ( u + sim.uinfx );
  centerOfMass[1] += dt * ( v + sim.uinfy );
  labCenterOfMass[0] += dt * u;
  labCenterOfMass[1] += dt * v;

  orientation += dt*omega;
  orientation = orientation> M_PI ? orientation-2*M_PI : orientation;
  orientation = orientation<-M_PI ? orientation+2*M_PI : orientation;

  const Real cosang = std::cos(orientation), sinang = std::sin(orientation);

  center[0] = centerOfMass[0] + cosang*d_gm[0] - sinang*d_gm[1];
  center[1] = centerOfMass[1] + sinang*d_gm[0] + cosang*d_gm[1];

  const Real CX = labCenterOfMass[0], CY = labCenterOfMass[1], t = sim.time;
  const Real cx = centerOfMass[0], cy = centerOfMass[1], angle = orientation;

  // do not print/write for initial PutObjectOnGrid
  if( dt <= 0 ) return;

  if(not sim.muteAll && sim.rank == 0)
  {
    printf("CM:[%.02f %.02f] C:[%.02f %.02f] ang:%.02f u:%.05f v:%.05f av:%.03f"
        " M:%.02e J:%.02e\n", (double)cx, (double)cy, (double)center[0], (double)center[1], (double)angle, (double)u, (double)v, (double)omega, (double)M, (double)J);
    std::stringstream ssF;
    ssF<<sim.path2file<<"/velocity_"<<obstacleID<<".dat";
    std::stringstream & fout = logger.get_stream(ssF.str());
    if(sim.step==0)
     fout<<"t dt CXsim CYsim CXlab CYlab angle u v omega M J accx accy accw\n";

    fout<<t<<" "<<dt<<" "<<cx<<" "<<cy<<" "<<CX<<" "<<CY<<" "<<angle<<" "
        <<u<<" "<<v<<" "<<omega<<" "<<M<<" "<<J<<" "<<fluidMomX/penalM<<" "
        <<fluidMomY/penalM<<" "<<fluidAngMom/penalJ<<"\n";
  }
}

Shape::Integrals Shape::integrateObstBlock(const std::vector<BlockInfo>& vInfo)
{
  Real _x=0, _y=0, _m=0, _j=0, _u=0, _v=0, _a=0;
  #pragma omp parallel for schedule(dynamic,1) reduction(+:_x,_y,_m,_j,_u,_v,_a)
  for(size_t i=0; i<vInfo.size(); i++)
  {
    const Real hsq = std::pow(vInfo[i].h, 2);
    const auto pos = obstacleBlocks[vInfo[i].blockID];
    if(pos == nullptr) continue;
    const CHI_MAT & __restrict__ CHI = pos->chi;
    const CHI_MAT & __restrict__ RHO = pos->rho;
    const UDEFMAT & __restrict__ UDEF = pos->udef;
    for(int iy=0; iy<ObstacleBlock::sizeY; ++iy)
    for(int ix=0; ix<ObstacleBlock::sizeX; ++ix)
    {
      if (CHI[iy][ix] <= 0) continue;
      Real p[2];
      vInfo[i].pos(p, ix, iy);
      const Real rhochi = CHI[iy][ix] * RHO[iy][ix] * hsq;
      p[0] -= centerOfMass[0];
      p[1] -= centerOfMass[1];
      _x += rhochi*p[0];
      _y += rhochi*p[1];
      _m += rhochi;
      _j += rhochi*(p[0]*p[0] + p[1]*p[1]);
      _u += rhochi*UDEF[iy][ix][0];
      _v += rhochi*UDEF[iy][ix][1];
      _a += rhochi*(p[0]*UDEF[iy][ix][1] - p[1]*UDEF[iy][ix][0]);
    }
  }
  Real quantities[7] = {_x,_y,_m,_j,_u,_v,_a};
  MPI_Allreduce(MPI_IN_PLACE, quantities, 7, MPI_Real, MPI_SUM, sim.chi->getCartComm());
  _x = quantities[0];
  _y = quantities[1];
  _m = quantities[2];
  _j = quantities[3];
  _u = quantities[4];
  _v = quantities[5];
  _a = quantities[6];
  _u /= _m;
  _v /= _m;
  _a /= _j;
  return Integrals(_x, _y, _m, _j, _u, _v, _a);
}

void Shape::removeMoments(const std::vector<BlockInfo>& vInfo)
{
  Shape::Integrals I = integrateObstBlock(vInfo);
  M = I.m; J = I.j;

  //with current center put shape on grid, with current shape on grid we updated
  //the center of mass, now recompute the distance betweeen the two:
  const Real dCx = center[0]-centerOfMass[0];
  const Real dCy = center[1]-centerOfMass[1];
  d_gm[0] =  dCx*std::cos(orientation) +dCy*std::sin(orientation);
  d_gm[1] = -dCx*std::sin(orientation) +dCy*std::cos(orientation);


  #pragma omp parallel for schedule(dynamic)
  for(size_t i=0; i<vInfo.size(); i++)
  {
    const auto pos = obstacleBlocks[vInfo[i].blockID];
    if(pos == nullptr) continue;

    for(int iy=0; iy<ObstacleBlock::sizeY; ++iy)
    for(int ix=0; ix<ObstacleBlock::sizeX; ++ix) {
        Real p[2];
        vInfo[i].pos(p, ix, iy);
        p[0] -= centerOfMass[0];
        p[1] -= centerOfMass[1];
        pos->udef[iy][ix][0] -= I.u -I.a*p[1];
        pos->udef[iy][ix][1] -= I.v +I.a*p[0];
    }
  }
};

void Shape::diagnostics()
{
  /*
  const std::vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
  const Real hsq = std::pow(vInfo[0].h, 2);
  Real _a=0, _m=0, _x=0, _y=0, _t=0;
  #pragma omp parallel for schedule(dynamic) reduction(+:_a,_m,_x,_y,_t)
  for(size_t i=0; i<vInfo.size(); i++) {
      const auto pos = obstacleBlocks[vInfo[i].blockID];
      if(pos == nullptr) continue;
      FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;

      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        if (pos->chi[iy][ix] <= 0) continue;
        const Real Xs = pos->chi[iy][ix] * hsq;
        Real p[2];
        vInfo[i].pos(p, ix, iy);
        p[0] -= centerOfMass[0];
        p[1] -= centerOfMass[1];
        const Real*const udef = pos->udef[iy][ix];
        const Real uDiff = b(ix,iy).u - (u -omega*p[1] +udef[0]);
        const Real vDiff = b(ix,iy).v - (v +omega*p[0] +udef[1]);
        _a += Xs;
        _m += Xs;
        _x += uDiff*Xs;
        _y += vDiff*Xs;
        _t += (p[0]*vDiff-p[1]*uDiff)*Xs;
      }
  }
  area_penal   = _a;
  mass_penal   = _m;
  forcex_penal = _x * sim.lambda;
  forcey_penal = _y * sim.lambda;
  torque_penal = _t * sim.lambda;
  */
}

void Shape::computeForces()
{
  //additive quantities:
  perimeter = 0; forcex = 0; forcey = 0; forcex_P = 0;
  forcey_P = 0; forcex_V = 0; forcey_V = 0; torque = 0;
  torque_P = 0; torque_V = 0; drag = 0; thrust = 0; lift= 0; 
  Pout = 0; PoutNew = 0; PoutBnd = 0; defPower = 0; defPowerBnd = 0; circulation = 0;

  for (auto & block : obstacleBlocks) if(block not_eq nullptr)
  {
    circulation += block->circulation;
    perimeter   += block->perimeter;  torque   += block->torque;
    forcex      += block->forcex;     forcey   += block->forcey;
    forcex_P    += block->forcex_P;   forcey_P += block->forcey_P;
    forcex_V    += block->forcex_V;   forcey_V += block->forcey_V;
    torque_P    += block->torque_P;   torque_V += block->torque_V;
    drag        += block->drag;       thrust   += block->thrust;
    lift        += block->lift;
    Pout        += block->Pout;        PoutNew += block->PoutNew;
    defPowerBnd += block->defPowerBnd;
    PoutBnd += block->PoutBnd;        defPower += block->defPower;
  }
  Real quantities[19];
  quantities[ 0] = circulation;
  quantities[ 1] = perimeter  ;
  quantities[ 2] = forcex     ;
  quantities[ 3] = forcex_P   ;
  quantities[ 4] = forcex_V   ;
  quantities[ 5] = torque_P   ;
  quantities[ 6] = drag       ;
  quantities[ 7] = lift       ;
  quantities[ 8] = Pout       ;
  quantities[ 9] = PoutNew    ;
  quantities[10] = PoutBnd    ;
  quantities[11] = torque     ;
  quantities[12] = forcey     ;
  quantities[13] = forcey_P   ;
  quantities[14] = forcey_V   ;
  quantities[15] = torque_V   ;
  quantities[16] = thrust     ;
  quantities[17] = defPowerBnd;
  quantities[18] = defPower   ;
  MPI_Allreduce(MPI_IN_PLACE, quantities, 19, MPI_Real, MPI_SUM, sim.chi->getCartComm());
  circulation = quantities[ 0];
  perimeter   = quantities[ 1];
  forcex      = quantities[ 2];
  forcex_P    = quantities[ 3];
  forcex_V    = quantities[ 4];
  torque_P    = quantities[ 5];
  drag        = quantities[ 6];
  lift        = quantities[ 7];
  Pout        = quantities[ 8];
  PoutNew     = quantities[ 9];
  PoutBnd     = quantities[10];
  torque      = quantities[11];
  forcey      = quantities[12];
  forcey_P    = quantities[13];
  forcey_V    = quantities[14];
  torque_V    = quantities[15];
  thrust      = quantities[16];
  defPowerBnd = quantities[17];
  defPower    = quantities[18];

  //derived quantities:
  Pthrust    = thrust * std::sqrt(u*u + v*v);
  Pdrag      =   drag * std::sqrt(u*u + v*v);
  const Real denUnb = Pthrust- std::min(defPower, (Real)0);
  const Real demBnd = Pthrust-          defPowerBnd;
  EffPDef    = Pthrust/std::max(denUnb, EPS);
  EffPDefBnd = Pthrust/std::max(demBnd, EPS);

  if(sim.dt <= 0) return;

  if (not sim.muteAll && sim._bDump && bDumpSurface)
  {
    std::stringstream s;
    if (sim.rank == 0)
      s << "x,y,p,u,v,nx,ny,omega,uDef,vDef,fX,fY,fXv,fYv\n"; 
    for(auto & block : obstacleBlocks) if(block not_eq nullptr)
      block->fill_stringstream(s);
    std::string st    = s.str();
    MPI_Offset offset = 0;
    MPI_Offset len    = st.size() * sizeof(char);
    MPI_File surface_file;
    std::stringstream ssF;
    ssF<<sim.path2file<<"/surface_"<<obstacleID <<"_"<<std::setfill('0')<<std::setw(7)<<sim.step<<".csv";
    MPI_File_delete(ssF.str().c_str(), MPI_INFO_NULL); // delete the file if it exists
    MPI_File_open(sim.chi->getCartComm(), ssF.str().c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &surface_file);
    MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, sim.chi->getCartComm());
    MPI_File_write_at_all(surface_file, offset, st.data(), st.size(), MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&surface_file);
  }

  int tot_blocks = 0;
  int nb = (int)sim.chi->getBlocksInfo().size();
  MPI_Reduce(&nb, &tot_blocks, 1, MPI_INT, MPI_SUM, 0, sim.chi->getCartComm());
  if(not sim.muteAll && sim.rank == 0)
  {
    std::stringstream ssF, ssP;
    ssF<<sim.path2file<<"/forceValues_"<<obstacleID<<".dat";
    ssP<<sim.path2file<<"/powerValues_"<<obstacleID<<".dat";

    std::stringstream &fileForce = logger.get_stream(ssF.str());
    if(sim.step==0)
      fileForce<<"time Fx Fy FxPres FyPres FxVisc FyVisc tau tauPres tauVisc drag thrust lift perimeter circulation blocks\n";

    fileForce<<sim.time<<" "<<forcex<<" "<<forcey<<" "<<forcex_P<<" "<<forcey_P
             <<" "<<forcex_V <<" "<<forcey_V<<" "<<torque <<" "<<torque_P<<" "
             <<torque_V<<" "<<drag<<" "<<thrust<<" "<<lift<<" "<<perimeter<<" "
             <<circulation<<" "<<tot_blocks<<"\n";

    std::stringstream &filePower = logger.get_stream(ssP.str());
    if(sim.step==0)
      filePower<<"time Pthrust Pdrag PoutBnd Pout PoutNew defPowerBnd defPower EffPDefBnd EffPDef\n";
    filePower<<sim.time<<" "<<Pthrust<<" "<<Pdrag<<" "<<PoutBnd<<" "<<Pout<<" "<<PoutNew<<" "<<defPowerBnd<<" "<<defPower<<" "<<EffPDefBnd<<" "<<EffPDef<<"\n";
  }
}

Shape::Shape( SimulationData& s, ArgumentParser& p, Real C[2] ) :
  sim(s), origC{C[0],C[1]}, origAng( p("-angle").asDouble(0)*M_PI/180 ),
  center{C[0],C[1]}, centerOfMass{C[0],C[1]}, orientation(origAng),
  rhoS(      p("-rhoS").asDouble(1) ),
  bFixed(    p("-bFixed").asBool(false) ),
  bFixedx(   p("-bFixedx" ).asBool(bFixed) ),
  bFixedy(   p("-bFixedy" ).asBool(bFixed) ),
  bForced(   p("-bForced").asBool(false) ),
  bForcedx(  p("-bForcedx").asBool(bForced)),
  bForcedy(  p("-bForcedy").asBool(bForced)),
  bBlockang( p("-bBlockAng").asBool(bForcedx || bForcedy) ),
  forcedu(  -p("-xvel").asDouble(0) ),
  forcedv(  -p("-yvel").asDouble(0) ),
  forcedomega(-p("-angvel").asDouble(0)),
  bDumpSurface(p("-dumpSurf").asInt(0)),
  timeForced(p("-timeForced").asDouble(std::numeric_limits<Real>::max())),
  bBreakSymmetry(p("-bBreakSymmetry").asBool(false))
  {}

Shape::~Shape()
{
  for(auto & entry : obstacleBlocks) delete entry;
  obstacleBlocks.clear();
}

//functions needed for restarting the simulation
void Shape::saveRestart( FILE * f ) {
  assert(f != NULL);
  fprintf(f, "x:     %20.20e\n", centerOfMass[0]   );
  fprintf(f, "y:     %20.20e\n", centerOfMass[1]   );
  fprintf(f, "xlab:  %20.20e\n", labCenterOfMass[0]);
  fprintf(f, "ylab:  %20.20e\n", labCenterOfMass[1]);
  fprintf(f, "u:     %20.20e\n", u                 );
  fprintf(f, "v:     %20.20e\n", v                 );
  fprintf(f, "omega: %20.20e\n", omega             );
  fprintf(f, "orientation: %20.20e\n", orientation );
  fprintf(f, "d_gm0: %20.20e\n", d_gm[0] );
  fprintf(f, "d_gm1: %20.20e\n", d_gm[1] );
  fprintf(f, "center0: %20.20e\n", center[0] );
  fprintf(f, "center1: %20.20e\n", center[1] );
  //maybe center0,center1,d_gm0,d_gm1 are not all needed, but it's only four numbers so we might
  //as well dump them
}

void Shape::loadRestart( FILE * f ) {
  assert(f != NULL);
  bool ret = true;
  ret = ret && 1==fscanf(f, "x:     %le\n", &centerOfMass[0]   );
  ret = ret && 1==fscanf(f, "y:     %le\n", &centerOfMass[1]   );
  ret = ret && 1==fscanf(f, "xlab:  %le\n", &labCenterOfMass[0]);
  ret = ret && 1==fscanf(f, "ylab:  %le\n", &labCenterOfMass[1]);
  ret = ret && 1==fscanf(f, "u:     %le\n", &u                 );
  ret = ret && 1==fscanf(f, "v:     %le\n", &v                 );
  ret = ret && 1==fscanf(f, "omega: %le\n", &omega             );
  ret = ret && 1==fscanf(f, "orientation: %le\n", &orientation );
  ret = ret && 1==fscanf(f, "d_gm0: %le\n", &d_gm[0] );
  ret = ret && 1==fscanf(f, "d_gm1: %le\n", &d_gm[1] );
  ret = ret && 1==fscanf(f, "center0: %le\n", &center[0] );
  ret = ret && 1==fscanf(f, "center1: %le\n", &center[1] );
  if( (not ret) ) {
    printf("Error reading restart file. Aborting...\n");
    fflush(0); abort();
  }
  if (sim.rank == 0)
    printf("Restarting Object.. x: %le, y: %le, xlab: %le, ylab: %le, u: %le, v: %le, omega: %le\n", centerOfMass[0], centerOfMass[1], labCenterOfMass[0], labCenterOfMass[1], u, v, omega);
}

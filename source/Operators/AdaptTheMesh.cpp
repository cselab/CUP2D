#include "AdaptTheMesh.h"

using namespace cubism;

void AdaptTheMesh::operator()(const double dt)
{
  count ++;

  if ((count-1) % 10 != 0 && count > 10) return;

  sim.startProfiler("AdaptTheMesh");

  //if (count % 1000 == 0) {auto K = computeDivergence(sim); K.run();}

  findOmega.run();//store vorticity in tmp

  double Rtol = sim.Rtol;
  double Ctol = sim.Ctol;

  bool verbose = sim.verbose;
  ScalarAMR tmp_amr   ( *sim.tmp    ,Rtol, Ctol, verbose);//refine according to tmp (which is vorticity magnitude)

  verbose = false;
  ScalarAMR chi_amr   ( *sim.chi    ,0.01, 0.0 , verbose);
  VectorAMR vel_amr   ( *sim.vel    ,Rtol, Ctol, verbose);
  ScalarAMR pres_amr  ( *sim.pres   ,Rtol, Ctol, verbose);
  ScalarAMR invRho_amr( *sim.invRho ,Rtol, Ctol, verbose);
  VectorAMR vFluid_amr( *sim.vFluid ,Rtol, Ctol, verbose);
  VectorAMR uDef_amr  ( *sim.uDef   ,Rtol, Ctol, verbose);  
  MeshAdaptation_basic<DumpGrid> dump_amr( *sim.dump);  
  
  ScalarAMR pRHS_amr  ( *sim.pRHS   ,Rtol, Ctol, verbose);
  VectorAMR tmpV_amr  ( *sim.tmpV   ,Rtol, Ctol, verbose);

  //ScalarAMR pOld_amr  ( *sim.pOld   ,Rtol, Ctol, verbose); 
  MeshAdaptation_basic<ScalarGrid> pOld_amr(*sim.pOld);  


  tmp_amr   .AdaptTheMesh();
  chi_amr   .AdaptLikeOther1<ScalarGrid>(*sim.tmp);
  vel_amr   .AdaptLikeOther1<ScalarGrid>(*sim.tmp);
  pres_amr  .AdaptLikeOther1<ScalarGrid>(*sim.tmp);
  invRho_amr.AdaptLikeOther1<ScalarGrid>(*sim.tmp);
  vFluid_amr.AdaptLikeOther1<ScalarGrid>(*sim.tmp);
  uDef_amr  .AdaptLikeOther1<ScalarGrid>(*sim.tmp);
  dump_amr  .AdaptLikeOther <ScalarGrid>(*sim.tmp);
  #ifdef PRECOND
  MeshAdaptation_basic<ScalarGrid> z_cg_amr( *sim.z_cg);  
  z_cg_amr  .AdaptLikeOther<ScalarGrid>(*sim.tmp);
  #endif
  pRHS_amr  .AdaptLikeOther1<ScalarGrid>(*sim.tmp);
  tmpV_amr  .AdaptLikeOther1<ScalarGrid>(*sim.tmp);

  //pOld_amr  .AdaptLikeOther1<ScalarGrid>(*sim.tmp);
  pOld_amr  .AdaptLikeOther<ScalarGrid>(*sim.tmp);

  chi_amr   .AdaptTheMesh();
  tmp_amr   .AdaptLikeOther1<ScalarGrid>(*sim.chi);
  vel_amr   .AdaptLikeOther1<ScalarGrid>(*sim.chi);
  pres_amr  .AdaptLikeOther1<ScalarGrid>(*sim.chi);
  invRho_amr.AdaptLikeOther1<ScalarGrid>(*sim.chi);
  vFluid_amr.AdaptLikeOther1<ScalarGrid>(*sim.chi);
  uDef_amr  .AdaptLikeOther1<ScalarGrid>(*sim.chi);
  dump_amr  .AdaptLikeOther <ScalarGrid>(*sim.chi);
  #ifdef PRECOND
  z_cg_amr  .AdaptLikeOther<ScalarGrid>(*sim.chi);
  #endif

  pRHS_amr  .AdaptLikeOther1<ScalarGrid>(*sim.chi);
  tmpV_amr  .AdaptLikeOther1<ScalarGrid>(*sim.chi);

  //pOld_amr  .AdaptLikeOther1<ScalarGrid>(*sim.chi);
  pOld_amr  .AdaptLikeOther<ScalarGrid>(*sim.chi);

  sim.chi   ->SortBlocks();
  sim.vel   ->SortBlocks();
  sim.pres  ->SortBlocks();
  sim.pOld  ->SortBlocks();
  sim.pRHS  ->SortBlocks();
  sim.invRho->SortBlocks();
  sim.tmpV  ->SortBlocks();
  sim.vFluid->SortBlocks();
  sim.tmp   ->SortBlocks();
  sim.uDef  ->SortBlocks();
  sim.dump  ->SortBlocks();
  #ifdef PRECOND
  sim.z_cg  ->SortBlocks();
  #endif
  sim.stopProfiler();
}
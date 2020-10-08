#include "AdaptTheMesh.h"

using namespace cubism;


void AdaptTheMesh::operator()(const double dt)
{
  sim.startProfiler("AdaptTheMesh");

  ScalarGrid * chi   = sim.chi    ;
  VectorGrid * vel   = sim.vel    ;
  ScalarGrid * pres  = sim.pres   ;
  ScalarGrid * pOld  = sim.pOld   ;
  ScalarGrid * pRHS  = sim.pRHS   ;
  ScalarGrid * invRho= sim.invRho ;
  VectorGrid * tmpV  = sim.tmpV   ;
  VectorGrid * vFluid= sim.vFluid ;
  ScalarGrid * tmp   = sim.tmp    ;
  VectorGrid * uDef  = sim.uDef   ;
  VectorGrid * vOld  = sim.vOld   ;
  DumpGrid   * dump  = sim.dump   ;

  #ifdef PRECOND
  ScalarGrid * z_cg   = sim.z_cg    ;
  #endif

  double Rtol = sim.Rtol;
  double Ctol = sim.Ctol;

  bool verbose = true;
  ScalarAMR chi_amr   ( *sim.chi,Rtol,Ctol, verbose);
  verbose = false;
  VectorAMR vel_amr   ( *vel    ,Rtol, Ctol, verbose);
  ScalarAMR pres_amr  ( *pres   ,Rtol, Ctol, verbose);
  ScalarAMR pOld_amr  ( *pOld   ,Rtol, Ctol, verbose); 
  ScalarAMR pRHS_amr  ( *pRHS   ,Rtol, Ctol, verbose);
  ScalarAMR invRho_amr( *invRho ,Rtol, Ctol, verbose);
  VectorAMR tmpV_amr  ( *tmpV   ,Rtol, Ctol, verbose);
  VectorAMR vFluid_amr( *vFluid ,Rtol, Ctol, verbose);
  ScalarAMR tmp_amr   ( *tmp    ,Rtol, Ctol, verbose);   
  VectorAMR uDef_amr  ( *uDef   ,Rtol, Ctol, verbose);  
  VectorAMR vOld_amr  ( *vOld   ,Rtol, Ctol, verbose);  
  MeshAdaptation_basic<DumpGrid> dump_amr  ( *dump);  

  chi_amr.AdaptTheMesh();

  #ifdef PRECOND
  ScalarAMR z_cg_amr   ( *z_cg    ,Rtol, Ctol, verbose);   
  z_cg_amr   .AdaptLikeOther1<ScalarGrid>(*chi);
  #endif
  vel_amr   .AdaptLikeOther1<ScalarGrid>(*chi);
  pres_amr  .AdaptLikeOther1<ScalarGrid>(*chi);
  pOld_amr  .AdaptLikeOther1<ScalarGrid>(*chi);
  pRHS_amr  .AdaptLikeOther1<ScalarGrid>(*chi);
  invRho_amr.AdaptLikeOther1<ScalarGrid>(*chi);
  tmpV_amr  .AdaptLikeOther1<ScalarGrid>(*chi);
  vFluid_amr.AdaptLikeOther1<ScalarGrid>(*chi);
  tmp_amr   .AdaptLikeOther1<ScalarGrid>(*chi);
  uDef_amr  .AdaptLikeOther1<ScalarGrid>(*chi);
  vOld_amr  .AdaptLikeOther1<ScalarGrid>(*chi);
  dump_amr  .AdaptLikeOther <ScalarGrid>(*chi);

  chi    -> SortBlocks();
  vel    -> SortBlocks();
  pres   -> SortBlocks();
  pOld   -> SortBlocks();
  pRHS   -> SortBlocks();
  invRho -> SortBlocks();
  tmpV   -> SortBlocks();
  vFluid -> SortBlocks();
  tmp    -> SortBlocks();
  uDef   -> SortBlocks();
  vOld   -> SortBlocks();
  dump   -> SortBlocks();
  #ifdef PRECOND
  z_cg   -> SortBlocks();
  #endif
  sim.stopProfiler();
}

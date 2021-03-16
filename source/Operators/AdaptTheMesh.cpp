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
  ScalarAMR chi_amr   ( *sim.chi    ,0.05, 0.01 , verbose);
  VectorAMR vel_amr   ( *sim.vel    ,Rtol, Ctol, verbose);
  ScalarAMR pres_amr  ( *sim.pres   ,Rtol, Ctol, verbose);
  VectorAMR uDef_amr  ( *sim.uDef   ,Rtol, Ctol, verbose);  
  MeshAdaptation_basic<DumpGrid> dump_amr( *sim.dump);  
  
  VectorAMR tmpV_amr  ( *sim.tmpV   ,Rtol, Ctol, verbose);

  //ScalarAMR pOld_amr  ( *sim.pOld   ,Rtol, Ctol, verbose); 
  MeshAdaptation_basic<ScalarGrid> pOld_amr(*sim.pOld);  

  tmp_amr .AdaptTheMesh();
  chi_amr .AdaptLikeOther1<ScalarGrid>(*sim.tmp);
  vel_amr .AdaptLikeOther1<ScalarGrid>(*sim.tmp);
  pres_amr.AdaptLikeOther1<ScalarGrid>(*sim.tmp);
  uDef_amr.AdaptLikeOther1<ScalarGrid>(*sim.tmp);
  dump_amr.AdaptLikeOther <ScalarGrid>(*sim.tmp);
  tmpV_amr.AdaptLikeOther1<ScalarGrid>(*sim.tmp);
  //pOld_amr.AdaptLikeOther1<ScalarGrid>(*sim.tmp);
  pOld_amr.AdaptLikeOther <ScalarGrid>(*sim.tmp);

  chi_amr .AdaptTheMesh();
  tmp_amr .AdaptLikeOther1<ScalarGrid>(*sim.chi);
  vel_amr .AdaptLikeOther1<ScalarGrid>(*sim.chi);
  pres_amr.AdaptLikeOther1<ScalarGrid>(*sim.chi);
  uDef_amr.AdaptLikeOther1<ScalarGrid>(*sim.chi);
  dump_amr.AdaptLikeOther <ScalarGrid>(*sim.chi);
  tmpV_amr.AdaptLikeOther1<ScalarGrid>(*sim.chi);
  //pOld_amr.AdaptLikeOther1<ScalarGrid>(*sim.chi);
  pOld_amr.AdaptLikeOther <ScalarGrid>(*sim.chi);

  sim.chi ->SortBlocks();
  sim.vel ->SortBlocks();
  sim.pres->SortBlocks();
  sim.pOld->SortBlocks();
  sim.tmpV->SortBlocks();
  sim.tmp ->SortBlocks();
  sim.uDef->SortBlocks();
  sim.dump->SortBlocks();
  sim.stopProfiler();
}

/*

Basic example of usage of CubismAMR, to solve

du/dt + du/dx + du/dy = 0

with a 1st order upwind scheme and Euler timestepping.

*/

//'DIMENSION' needs to be defined first (spatial dimensions)
#define DIMENSION 2

//Include the necessary headers
#include <mpi.h>
#include "Definitions.h"
#include "HDF5Dumper.h"
#include "AMR_MeshAdaptation.h"
#include "BlockInfo.h"
#include "BlockLabMPI.h"
#include "GridMPI.h"
#include "Grid.h"
#include "StencilInfo.h"

//define shorthand aliases for easier use
#define blocksize 8 //this is the size of each 2D GridBlock (8x8)
using element   = cubism::ScalarElement<double>;
using block     = cubism::GridBlock<blocksize,DIMENSION,element>;
using grid      = cubism::GridMPI<cubism::Grid<block>>;
using lab       = cubism::BlockLabMPI<cubism::BlockLab<grid>>;
using amr       = cubism::MeshAdaptation<lab>;

//struct where we write down the uniform grid computation of our numerical scheme
struct KernelAdvection
{
  KernelAdvection(grid & _output) :output(&_output){}
  grid * output;

  //stencil of +-1 points in x and y directions and no points in z direction
  const cubism::StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0}};

  //compute the 1st order finite difference for derivatives
  //the result is stored in a second grid ('output')
  //careful: 'mylab' is a copy of the input grid and the input grid should
  //not be overwritten while stencil computations with Blocklabs are still
  //being performed.
  void operator()(lab & mylab, const cubism::BlockInfo& info) const
  {
    const double ih = 1.0/info.h;
    block & out = (*output)(info.blockID);
    for(int iy=0; iy<block::sizeY; ++iy)
    for(int ix=0; ix<block::sizeX; ++ix)
    {
      const element dudx = ih*(mylab(ix,iy)-mylab(ix-1,iy));
      const element dudy = ih*(mylab(ix,iy)-mylab(ix,iy-1));
      out(ix,iy) = dudx + dudy;
    }
  }
};

void performSimulation()
{
  //some initial hard-coded parameters
  constexpr int blocksX = 2; //number of blocks in X, at coarsest level
  constexpr int blocksY = 2; //number of blocks in Y, at coarsest level
  constexpr int blocksZ = 1; //number of blocks in Z, at coarsest level (does not really apply to 2D)
  constexpr double domain_size = 1.0; //square domain of size 1.0 x 1.0
  constexpr int levelStart = 2; //initial level at t=0
  constexpr int levelMax = 3; //maximum level allowed
  constexpr bool xperiodic = true; //assume periodic domain
  constexpr bool yperiodic = true;
  constexpr bool zperiodic = true;
  grid g  (blocksX,blocksY,blocksZ,domain_size,levelStart,levelMax,MPI_COMM_WORLD,xperiodic,yperiodic,zperiodic); 
  grid tmp(blocksX,blocksY,blocksZ,domain_size,levelStart,levelMax,MPI_COMM_WORLD,xperiodic,yperiodic,zperiodic); 
  amr   g_amr(g  ,0.5,0.05); //refine when u > 0.5, compress when u < 0.05
  amr tmp_amr(tmp,0.5,0.05);

  //set an initial condition: iterate over all BlockInfos
  for (cubism::BlockInfo & info: g.getBlocksInfo())
  {
      //access the corresponding GridBlock
      block & b = g(info.blockID);

      //iterate over its elements
      for (int iy = 0; iy < blocksize; iy++)
      for (int ix = 0; ix < blocksize; ix++)
      {
         double p[2]; //these are the (x,y) coordinates of a point
         info.pos(p,ix,iy);
         const double r2 = pow(p[0]-0.5,2) + pow(p[1]-0.5,2);
         b(ix,iy).s = r2 < 0.01 ? 1.0 : 0.0;
      }
  }

  //perform iterations 
  KernelAdvection K(tmp);
  double time = 0;
  double dt = 0.001; //set a small value for the timestep
  int step = 0;
  while (time < 1.0)
  {
     step ++;
     std::cout << "Time=" << time << std::endl;
     time += dt;

     //compute rhs for Euler timestep
     cubism::compute<lab>(K,&g);
     for (cubism::BlockInfo & info: g.getBlocksInfo())
     {
         block & b = g  (info.blockID);
         block & t = tmp(info.blockID);
         for (int iy = 0; iy < blocksize; iy++)
         for (int ix = 0; ix < blocksize; ix++)
         {
             b(ix,iy) -= dt * t(ix,iy);
         }
     }

     //tag the grid
     g_amr.Tag();
     tmp_amr.TagLike(g.getBlocksInfo());

     //refine the grid
     g_amr.Adapt(time,true,false);
     tmp_amr.Adapt(time,false,false);

     //save output to file every 10 timesteps
     if (step % 10 == 0) cubism::DumpHDF5_MPI<cubism::StreamerScalar,double>(g, time, "result_" + std::to_string(step),"./");
  }
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  performSimulation();  

  MPI_Finalize();
  return 0;
}

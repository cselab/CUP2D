//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <omp.h>
#define OMPI_SKIP_MPICXX 1
#ifdef _FLOAT_PRECISION_
using Real = float;
#define MPI_Real MPI_FLOAT
#endif
#ifdef _DOUBLE_PRECISION_
using Real = double;
#define MPI_Real MPI_DOUBLE
#endif
#ifdef _LONG_DOUBLE_PRECISION_
using Real = long double;
#define MPI_Real MPI_LONG_DOUBLE
#endif

#include <Cubism/ArgumentParser.h>
#include <Cubism/Grid.h>
#include <Cubism/GridMPI.h>
#include <Cubism/BlockInfo.h>
#include <Cubism/BlockLab.h>
#include <Cubism/BlockLabMPI.h>
#include <Cubism/StencilInfo.h>
#include <Cubism/AMR_MeshAdaptation.h>
#include <Cubism/AMR_MeshAdaptationMPI.h>
#include <Cubism/Definitions.h>

#ifndef _DIM_
#define _DIM_ 2
#endif//_DIM_

enum BCflag {freespace, periodic, wall};
inline BCflag string2BCflag(const std::string &strFlag)
{
  if      (strFlag == "periodic" )
  {
    //printf("[CUP2D] Using periodic boundary conditions\n");
    return periodic;
  }
  else if (strFlag == "freespace")
  {
    //printf("[CUP2D] Using freespace boundary conditions\n");
    return freespace;
  }
  else
  {
     fprintf(stderr,"BC not recognized %s\n",strFlag.c_str());
     fflush(0);abort();
     return periodic; // dummy
  }
}
//CAREFUL THESE ARE GLOBAL VARIABLES!
extern BCflag cubismBCX;
extern BCflag cubismBCY;

template<typename BlockType, template<typename X> class allocator = std::allocator>
class BlockLabDirichlet: public cubism::BlockLab<BlockType, allocator>
{
public:
  using ElementType = typename BlockType::ElementType;
  static constexpr int sizeX = BlockType::sizeX;
  static constexpr int sizeY = BlockType::sizeY;
  static constexpr int sizeZ = BlockType::sizeZ;

  virtual bool is_xperiodic() override{ return cubismBCX == periodic; }
  virtual bool is_yperiodic() override{ return cubismBCY == periodic; }
  virtual bool is_zperiodic() override{ return false; }

  // Apply bc on face of direction dir and side side (0 or 1):
  template<int dir, int side> void applyBCface(bool coarse=false)
  {

    const int A = 1 - dir;
    if (!coarse)
    {
      auto * const cb = this->m_cacheBlock;
      int s[3] = {0,0,0}, e[3] = {0,0,0};
      const int* const stenBeg = this->m_stencilStart;
      const int* const stenEnd = this->m_stencilEnd;
      s[0] =  dir==0 ? (side==0 ? stenBeg[0] : sizeX ) : stenBeg[0];
      s[1] =  dir==1 ? (side==0 ? stenBeg[1] : sizeY ) : stenBeg[1];
      e[0] =  dir==0 ? (side==0 ? 0 : sizeX + stenEnd[0]-1 ) : sizeX +  stenEnd[0]-1;
      e[1] =  dir==1 ? (side==0 ? 0 : sizeY + stenEnd[1]-1 ) : sizeY +  stenEnd[1]-1;

      for(int iy=s[1]; iy<e[1]; iy++)
      for(int ix=s[0]; ix<e[0]; ix++)
      {
        const int x = ( dir==0? (side==0? 0: sizeX-1):ix ) - stenBeg[0];
        const int y = ( dir==1? (side==0? 0: sizeY-1):iy ) - stenBeg[1];
        //cb->Access(ix-stenBeg[0], iy-stenBeg[1], 0) = (-1.0)*cb->Access(x,y,0);
        cb->Access(ix-stenBeg[0], iy-stenBeg[1], 0).member(1-A) = (-1.0)*cb->Access(x,y,0).member(1-A);
        cb->Access(ix-stenBeg[0], iy-stenBeg[1], 0).member(A) = cb->Access(x,y,0).member(A);
      }
    }
    else
    {
      auto * const cb = this->m_CoarsenedBlock;

      const int eI[3] = {(this->m_stencilEnd[0])/2 + 1 + this->m_InterpStencilEnd[0] -1,
                         (this->m_stencilEnd[1])/2 + 1 + this->m_InterpStencilEnd[1] -1,
                         (this->m_stencilEnd[2])/2 + 1 + this->m_InterpStencilEnd[2] -1};
      const int sI[3] = {(this->m_stencilStart[0]-1)/2+  this->m_InterpStencilStart[0],
                         (this->m_stencilStart[1]-1)/2+  this->m_InterpStencilStart[1],
                         (this->m_stencilStart[2]-1)/2+  this->m_InterpStencilStart[2]};

      const int* const stenBeg = sI;
      const int* const stenEnd = eI;

      int s[3] = {0,0,0}, e[3] = {0,0,0};

      s[0] =  dir==0 ? (side==0 ? stenBeg[0] : sizeX/2 ) : stenBeg[0];
      s[1] =  dir==1 ? (side==0 ? stenBeg[1] : sizeY/2 ) : stenBeg[1];

      e[0] =  dir==0 ? (side==0 ? 0 : sizeX/2 + stenEnd[0]-1 ) : sizeX/2 +  stenEnd[0]-1;
      e[1] =  dir==1 ? (side==0 ? 0 : sizeY/2 + stenEnd[1]-1 ) : sizeY/2 +  stenEnd[1]-1;

      for(int iy=s[1]; iy<e[1]; iy++)
      for(int ix=s[0]; ix<e[0]; ix++)
      {
        const int x = ( dir==0? (side==0? 0: sizeX/2-1):ix ) - stenBeg[0];
        const int y = ( dir==1? (side==0? 0: sizeY/2-1):iy ) - stenBeg[1];
        //cb->Access(ix-stenBeg[0], iy-stenBeg[1], 0) = (-1.0)*cb->Access(x,y,0);
        cb->Access(ix-stenBeg[0], iy-stenBeg[1], 0).member(1-A) = (-1.0)*cb->Access(x,y,0).member(1-A);
        cb->Access(ix-stenBeg[0], iy-stenBeg[1], 0).member(A) = cb->Access(x,y,0).member(A);
      }
    }
  }

  // Called by Cubism:
  void _apply_bc(const cubism::BlockInfo& info, const Real t = 0, const bool coarse = false) override
  {
    if (!coarse)
    {
      if (is_xperiodic() == false)
      {
        if( info.index[0]==0 )           this->template applyBCface<0,0>();
        if( info.index[0]==this->NX-1 )  this->template applyBCface<0,1>();
      }
      if (is_yperiodic() == false)
      {
        if( info.index[1]==0 )           this->template applyBCface<1,0>();
        if( info.index[1]==this->NY-1 )  this->template applyBCface<1,1>();
      }
    }
    else
    {
      if (is_xperiodic() == false)
      {
        if( info.index[0]==0 )           this->template applyBCface<0,0>(coarse);
        if( info.index[0]==this->NX-1 )  this->template applyBCface<0,1>(coarse);
      }
      if (is_yperiodic() == false)
      {
        if( info.index[1]==0 )           this->template applyBCface<1,0>(coarse);
        if( info.index[1]==this->NY-1 )  this->template applyBCface<1,1>(coarse);
      }
    }
  }

  BlockLabDirichlet(): cubism::BlockLab<BlockType,allocator>(){}
  BlockLabDirichlet(const BlockLabDirichlet&) = delete;
  BlockLabDirichlet& operator=(const BlockLabDirichlet&) = delete;
};



template<typename BlockType, template<typename X> class allocator = std::allocator>
class BlockLabNeumann: public cubism::BlockLabNeumann<BlockType, 2, allocator>
{
public:
  using cubismLab = cubism::BlockLabNeumann<BlockType, 2, allocator>;
  virtual bool is_xperiodic() override{ return cubismBCX == periodic; }
  virtual bool is_yperiodic() override{ return cubismBCY == periodic; }
  virtual bool is_zperiodic() override{ return false; }

  // Called by Cubism:
  void _apply_bc(const cubism::BlockInfo& info, const Real t = 0, const bool coarse = false) override
  {
      if (is_xperiodic() == false)
      {
       if(info.index[0]==0 )          cubismLab::template Neumann2D<0,0>(coarse);
       if(info.index[0]==this->NX-1 ) cubismLab::template Neumann2D<0,1>(coarse);
      }
      if (is_yperiodic() == false)
      {
       if(info.index[1]==0 )          cubismLab::template Neumann2D<1,0>(coarse);
       if(info.index[1]==this->NY-1 ) cubismLab::template Neumann2D<1,1>(coarse);
      }
  }
};

using ScalarElement = cubism::ScalarElement<Real>;
using VectorElement = cubism::VectorElement<2,Real>;
using ScalarBlock   = cubism::GridBlock<_BS_,2,ScalarElement>;
using VectorBlock   = cubism::GridBlock<_BS_,2,VectorElement>;
using ScalarGrid    = cubism::GridMPI<cubism::Grid<ScalarBlock, std::allocator>>;
using VectorGrid    = cubism::GridMPI<cubism::Grid<VectorBlock, std::allocator>>;

using VectorLab = cubism::BlockLabMPI<BlockLabDirichlet<VectorBlock, std::allocator>,VectorGrid>;
using ScalarLab = cubism::BlockLabMPI<BlockLabNeumann  <ScalarBlock, std::allocator>,ScalarGrid>;
using ScalarAMR = cubism::MeshAdaptationMPI<ScalarGrid,ScalarLab,ScalarGrid>;
using VectorAMR = cubism::MeshAdaptationMPI<VectorGrid,VectorLab,ScalarGrid>;

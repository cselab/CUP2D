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

#ifndef _FLOAT_PRECISION_
using Real = double;
#else // _FLOAT_PRECISION_
using Real = float;
#endif // _FLOAT_PRECISION_

#include <Cubism/ArgumentParser.h>
#include <Cubism/Grid.h>
#include <Cubism/BlockInfo.h>
#include <Cubism/BlockLab.h>
#include <Cubism/StencilInfo.h>

#include <Cubism/AMR_MeshAdaptation.h>

#ifndef _BS_
#define _BS_ 32
#endif//_BS_

#ifndef _DIM_
#define _DIM_ 2
#endif//_BS_


struct ScalarElement
{
  using RealType = Real;
  Real s = 0;
  inline void clear() { s = 0; }
  inline void set(const Real v) { s = v; }
  inline void copy(const ScalarElement& c) { s = c.s; }
  //ScalarElement(const ScalarElement& c) = delete;
  // ScalarElement& operator=(const ScalarElement& c) { s = c.s; return *this; }
  ScalarElement& operator=(const ScalarElement& c) = default;

  ScalarElement &operator*=(const Real a)
  {
    this->s*=a;
    return *this;
  }
  ScalarElement &operator+=(const ScalarElement &rhs)
  {
    this->s+=rhs.s;
    return *this;
  }
  ScalarElement &operator-=(const ScalarElement &rhs)
  {
    this->s-=rhs.s;
    return *this;
  }
  ScalarElement &operator/=(const ScalarElement &rhs)
  {
    this->s/=rhs.s;
    return *this;
  }
  friend ScalarElement operator*(const Real a, ScalarElement el)
  {
      return (el *= a);
  }
  friend ScalarElement operator+(ScalarElement lhs, const ScalarElement &rhs)
  {
      return (lhs += rhs);
  }
  friend ScalarElement operator-(ScalarElement lhs, const ScalarElement &rhs)
  {
      return (lhs -= rhs);
  }
  friend ScalarElement operator/(ScalarElement lhs, const ScalarElement &rhs)
  {
      return (lhs /= rhs);
  }
  bool operator<(const ScalarElement &other) const
  {
     return (s < other.s);
  }
  bool operator>(const ScalarElement &other) const
  {
     return (s > other.s);
  }
  bool operator<=(const ScalarElement &other) const
  {
     return (s <= other.s);
  }
  bool operator>=(const ScalarElement &other) const
  {
     return (s >= other.s);
  }
  double magnitude()
  {
    return s;
  }

  Real & member(int i)
  {
    return s;
  }
  static constexpr int DIM = 1;

};

struct VectorElement
{
  using RealType = Real;
  static constexpr int DIM = _DIM_;
  Real u[DIM];

  VectorElement() { clear(); }

  inline void clear() { for(int i=0; i<DIM; ++i) u[i] = 0; }
  inline void set(const Real v) { for(int i=0; i<DIM; ++i) u[i] = v; }
  inline void copy(const VectorElement& c) {
    for(int i=0; i<DIM; ++i) u[i] = c.u[i];
  }
  //VectorElement(const VectorElement& c) = delete;
  // VectorElement& operator=(const VectorElement& c) {
  //   for(int i=0; i<DIM; ++i) u[i] = c.u[i];
  //   return *this;
  // }
  VectorElement& operator=(const VectorElement& c) = default;
  
  VectorElement &operator*=(const Real a)
  {
    for(int i=0; i<DIM; ++i)
      this->u[i]*=a;
    return *this;
  }
  VectorElement &operator+=(const VectorElement &rhs)
  {
    for(int i=0; i<DIM; ++i)
      this->u[i]+=rhs.u[i];
    return *this;
  }
  VectorElement &operator-=(const VectorElement &rhs)
  {
    for(int i=0; i<DIM; ++i)
      this->u[i]-=rhs.u[i];
    return *this;
  }
  VectorElement &operator/=(const VectorElement &rhs)
  {
    for(int i=0; i<DIM; ++i)
      this->u[i]/=rhs.u[i];
    return *this;
  }
  friend VectorElement operator*(const Real a, VectorElement el)
  {
      return (el *= a);
  }
  friend VectorElement operator+(VectorElement lhs, const VectorElement &rhs)
  {
      return (lhs += rhs);
  }
  friend VectorElement operator-(VectorElement lhs, const VectorElement &rhs)
  {
      return (lhs -= rhs);
  }
  friend VectorElement operator/(VectorElement lhs, const VectorElement &rhs)
  {
      return (lhs /= rhs);
  }
  bool operator<(const VectorElement &other) const
  {
    double s1 = 0.0;
    double s2 = 0.0;
    for(int i=0; i<DIM; ++i)
    {
      s1 +=u[i]*u[i];
      s2 +=other.u[i]*other.u[i];
    }

    return (s1 < s2);
  }
  bool operator>(const VectorElement &other) const
  {
    double s1 = 0.0;
    double s2 = 0.0;
    for(int i=0; i<DIM; ++i)
    {
      s1 +=u[i]*u[i];
      s2 +=other.u[i]*other.u[i];
    }

    return (s1 > s2);
  }
  bool operator<=(const VectorElement &other) const
  {
    double s1 = 0.0;
    double s2 = 0.0;
    for(int i=0; i<DIM; ++i)
    {
      s1 +=u[i]*u[i];
      s2 +=other.u[i]*other.u[i];
    }

    return (s1 <= s2);
  }
  bool operator>=(const VectorElement &other) const
  {
    double s1 = 0.0;
    double s2 = 0.0;
    for(int i=0; i<DIM; ++i)
    {
      s1 +=u[i]*u[i];
      s2 +=other.u[i]*other.u[i];
    }

    return (s1 >= s2);
  }

  double magnitude()
  {
    double s1 = 0.0;
    for(int i=0; i<DIM; ++i)
    {
      s1 +=u[i]*u[i];
    }
    return sqrt(s1);
  }
  Real & member(int i)
  {
    return u[i];
  }
};

template <typename TElement>
struct GridBlock
{
  //these identifiers are required by cubism!
  static constexpr int BS = _BS_;
  static constexpr int sizeX = _BS_;
  static constexpr int sizeY = _BS_;
  static constexpr int sizeZ = _DIM_ > 2 ? _BS_ : 1;
  static constexpr std::array<int, 3> sizeArray = {sizeX, sizeY, sizeZ};
  using ElementType = TElement;
  using element_type = TElement;
  using RealType = Real;

  ElementType data[sizeZ][sizeY][sizeX];

  inline void clear() {
      ElementType * const entry = &data[0][0][0];
      for(int i=0; i<sizeX*sizeY*sizeZ; ++i) entry[i].clear();
  }
  inline void set(const Real v) {
      ElementType * const entry = &data[0][0][0];
      for(int i=0; i<sizeX*sizeY*sizeZ; ++i) entry[i].set(v);
  }
  inline void copy(const GridBlock<ElementType>& c) {
      ElementType * const entry = &data[0][0][0];
      const ElementType * const source = &c.data[0][0][0];
      for(int i=0; i<sizeX*sizeY*sizeZ; ++i) entry[i].copy(source[i]);
  }

  const ElementType& operator()(int ix, int iy=0, int iz=0) const {
      assert(ix>=0 && iy>=0 && iz>=0 && ix<sizeX && iy<sizeY && iz<sizeZ);
      return data[iz][iy][ix];
  }
  ElementType& operator()(int ix, int iy=0, int iz=0) {
      assert(ix>=0 && iy>=0 && iz>=0 && ix<sizeX && iy<sizeY && iz<sizeZ);
      return data[iz][iy][ix];
  }
  GridBlock(const GridBlock&) = delete;
  GridBlock& operator=(const GridBlock&) = delete;
};

template<typename BlockType,
         template<typename X> class allocator = std::allocator>
class BlockLabOpen: public cubism::BlockLab<BlockType, allocator>
{
public:
  using ElementType = typename BlockType::ElementType;
  static constexpr int sizeX = BlockType::sizeX;
  static constexpr int sizeY = BlockType::sizeY;
  static constexpr int sizeZ = BlockType::sizeZ;

  virtual bool is_xperiodic() override{ return false; }
  virtual bool is_yperiodic() override{ return false; }
  virtual bool is_zperiodic() override{ return false; }

  // Apply bc on face of direction dir and side side (0 or 1):
  template<int dir, int side> void applyBCface(bool coarse=false)
  {
    if (!coarse)
    {
      auto * const cb = this->m_cacheBlock;
      int s[3] = {0,0,0}, e[3] = {0,0,0};
      const int* const stenBeg = this->m_stencilStart;
      const int* const stenEnd = this->m_stencilEnd;
      s[0] =  dir==0 ? (side==0 ? stenBeg[0] : sizeX ) : stenBeg[0];
      s[1] =  dir==1 ? (side==0 ? stenBeg[1] : sizeY ) : stenBeg[1];

      e[0] =  dir==0 ? (side==0 ? 0 : sizeX + stenEnd[0]-1 )
                     : sizeX +  stenEnd[0]-1;
      e[1] =  dir==1 ? (side==0 ? 0 : sizeY + stenEnd[1]-1 )
                     : sizeY +  stenEnd[1]-1;

      for(int iy=s[1]; iy<e[1]; iy++)
      for(int ix=s[0]; ix<e[0]; ix++)
        cb->Access(ix-stenBeg[0], iy-stenBeg[1], 0) =
            cb->Access(
            ( dir==0? (side==0? 0: sizeX-1):ix ) - stenBeg[0],
            ( dir==1? (side==0? 0: sizeY-1):iy ) - stenBeg[1], 0 );
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

      e[0] =  dir==0 ? (side==0 ? 0 : sizeX/2 + stenEnd[0]-1 )
                     : sizeX/2 +  stenEnd[0]-1;
      e[1] =  dir==1 ? (side==0 ? 0 : sizeY/2 + stenEnd[1]-1 )
                     : sizeY/2 +  stenEnd[1]-1;

      for(int iy=s[1]; iy<e[1]; iy++)
      for(int ix=s[0]; ix<e[0]; ix++)
        cb->Access(ix-stenBeg[0], iy-stenBeg[1], 0) =
            cb->Access(
            ( dir==0? (side==0? 0: sizeX/2-1):ix ) - stenBeg[0],
            ( dir==1? (side==0? 0: sizeY/2-1):iy ) - stenBeg[1], 0 );
    }
  }

  // Called by Cubism:
  void _apply_bc(const cubism::BlockInfo& info, const Real t = 0, const bool coarse = false)
  {
    if (!coarse)
    {
      if( info.index[0]==0 )           this->template applyBCface<0,0>();
      if( info.index[0]==this->NX-1 )  this->template applyBCface<0,1>();
      if( info.index[1]==0 )           this->template applyBCface<1,0>();
      if( info.index[1]==this->NY-1 )  this->template applyBCface<1,1>();
    }
    else
    {
      if( info.index[0]==0 )           this->template applyBCface<0,0>(coarse);
      if( info.index[0]==this->NX-1 )  this->template applyBCface<0,1>(coarse);
      if( info.index[1]==0 )           this->template applyBCface<1,0>(coarse);
      if( info.index[1]==this->NY-1 )  this->template applyBCface<1,1>(coarse);
    }


  }

  BlockLabOpen(): cubism::BlockLab<BlockType,allocator>(){}
  BlockLabOpen(const BlockLabOpen&) = delete;
  BlockLabOpen& operator=(const BlockLabOpen&) = delete;

  const ElementType& operator()(const int ix, const int iy) const {
    return this->read(ix,iy,0);
  }
  ElementType& operator()(const int ix, const int iy) {
    return this->m_cacheBlock->Access(ix - this->m_stencilStart[0],
                                      iy - this->m_stencilStart[1], 0);
  }
};

template<typename BlockType,
         template<typename X> class allocator = std::allocator>
class BlockLabDirichlet: public cubism::BlockLab<BlockType, allocator>
{
public:
  using ElementType = typename BlockType::ElementType;
  static constexpr int sizeX = BlockType::sizeX;
  static constexpr int sizeY = BlockType::sizeY;
  static constexpr int sizeZ = BlockType::sizeZ;

  virtual bool is_xperiodic() override{ return false; }
  virtual bool is_yperiodic() override{ return false; }
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
  void _apply_bc(const cubism::BlockInfo& info, const Real t = 0, const bool coarse = false)
  {
    if (!coarse)
    {
      if( info.index[0]==0 )           this->template applyBCface<0,0>();
      if( info.index[0]==this->NX-1 )  this->template applyBCface<0,1>();
      if( info.index[1]==0 )           this->template applyBCface<1,0>();
      if( info.index[1]==this->NY-1 )  this->template applyBCface<1,1>();
    }
    else
    {
      if( info.index[0]==0 )           this->template applyBCface<0,0>(coarse);
      if( info.index[0]==this->NX-1 )  this->template applyBCface<0,1>(coarse);
      if( info.index[1]==0 )           this->template applyBCface<1,0>(coarse);
      if( info.index[1]==this->NY-1 )  this->template applyBCface<1,1>(coarse);
    }


  }

  BlockLabDirichlet(): cubism::BlockLab<BlockType,allocator>(){}
  BlockLabDirichlet(const BlockLabDirichlet&) = delete;
  BlockLabDirichlet& operator=(const BlockLabDirichlet&) = delete;

  const ElementType& operator()(const int ix, const int iy) const {
    return this->read(ix,iy,0);
  }
  ElementType& operator()(const int ix, const int iy) {
    return this->m_cacheBlock->Access(ix - this->m_stencilStart[0],
                                      iy - this->m_stencilStart[1], 0);
  }
};

struct StreamerScalar
{
  static constexpr int NCHANNELS = 1;
  template <typename TBlock, typename T>
  static inline void operate(const TBlock& b,
    const int ix, const int iy, const int iz, T output[NCHANNELS]) {
    output[0] = b(ix,iy,iz).s;
  }
  static std::string prefix() { return std::string(""); }
  static const char * getAttributeName() { return "Scalar"; }
};

struct StreamerVector
{
  static constexpr int NCHANNELS = 3;

  template <typename TBlock, typename T>
  static void operate(const TBlock& b, const int ix, const int iy, const int iz, T output[NCHANNELS]) {
      for (int i = 0; i < _DIM_; i++) output[i] = b(ix,iy,iz).u[i];
  }

  template <typename TBlock, typename T>
  static void operate(TBlock& b, const T input[NCHANNELS], const int ix, const int iy, const int iz) {
      for (int i = 0; i < _DIM_; i++) b(ix,iy,iz).u[i] = input[i];
  }
  static std::string prefix() { return std::string(""); }
  static const char * getAttributeName() { return "Vector"; }
};

using ScalarBlock = GridBlock<ScalarElement>;
using VectorBlock = GridBlock<VectorElement>;
using VectorGrid = cubism::Grid<VectorBlock, std::allocator>;
using ScalarGrid = cubism::Grid<ScalarBlock, std::allocator>;
//using VectorLab = BlockLabOpen<VectorBlock, std::allocator>;
using VectorLab = BlockLabDirichlet<VectorBlock, std::allocator>;
using ScalarLab = BlockLabOpen<ScalarBlock, std::allocator>;
//using ScalarLab = BlockLabDirichlet<ScalarBlock, std::allocator>;

using ScalarAMR = cubism::MeshAdaptation<ScalarGrid,ScalarLab>;
using VectorAMR = cubism::MeshAdaptation<VectorGrid,VectorLab>;

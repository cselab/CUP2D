//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Definitions.h"
#include "../ObstacleBlock.h"

inline bool _is_touching(
  const cubism::BlockInfo& INFO, const Real BBOX[2][2], const Real safety )
{
  Real MINP[2], MAXP[2];
  INFO.pos(MINP, 0, 0);
  INFO.pos(MAXP, ObstacleBlock::sizeX-1, ObstacleBlock::sizeY-1);
  //for(int i=0; i<2; ++i) { MINP[i] -= safety; MAXP[i] += safety; }
  const Real intrsct[2][2] = {
   { std::max(MINP[0], BBOX[0][0]), std::min(MAXP[0], BBOX[0][1]) },
   { std::max(MINP[1], BBOX[1][0]), std::min(MAXP[1], BBOX[1][1]) }
  };
  return intrsct[0][1] - intrsct[0][0]>0 && intrsct[1][1] - intrsct[1][0]>0;
}

struct FillBlocks_Cylinder
{
  const Real radius, safety, rhoS, pos[2], bbox[2][2] = {
    { pos[0] - radius - safety, pos[0] + radius + safety },
    { pos[1] - radius - safety, pos[1] + radius + safety }
  };

  FillBlocks_Cylinder(Real R, Real h, Real C[2], Real rho) :
    radius(R), safety(2*h), rhoS(rho), pos{(Real)C[0], (Real)C[1]} {}

  inline Real distanceTocylinder(const Real x, const Real y) const {
      return radius - std::sqrt(x*x+y*y); // pos inside, neg outside
  }

  inline bool is_touching(const cubism::BlockInfo& INFO) const {
    return _is_touching(INFO, bbox, safety);
  }

  void operator()(const cubism::BlockInfo&, ScalarBlock&, ObstacleBlock&) const;
};

struct FillBlocks_HalfCylinder
{
  const Real radius, safety, pos[2], angle, rhoS;
  const Real cosang = std::cos(angle), sinang = std::sin(angle);
  const Real bbox[2][2] = {
    { pos[0] - radius - safety, pos[0] + radius + safety },
    { pos[1] - radius - safety, pos[1] + radius + safety }
  };

  FillBlocks_HalfCylinder(Real R, Real h, Real C[2], Real rho, Real ang):
    radius(R), safety(2*h), pos{(Real)C[0],(Real)C[1]}, angle(ang), rhoS(rho) {}

  inline Real distanceTocylinder(const Real x, const Real y) const {
    const Real X =   x * cosang + y * sinang;
    if(X>0) return -X;
    //const Real Y = - x*sinang + y*cosang; /// For default orientation
    //if(Y>0) return -Y;                    /// pointing downwards.
    else return radius - std::sqrt(x*x+y*y); // (pos inside, neg outside)
  }

  inline bool is_touching(const cubism::BlockInfo& INFO) const {
    return _is_touching(INFO, bbox, safety);
  }

  void operator()(const cubism::BlockInfo&, ScalarBlock&, ObstacleBlock&) const;
};

struct FillBlocks_Ellipse
{
  const Real e0, e1, safety, pos[2], angle, rhoS;
  const Real e[2] = {e0, e1}, sqMinSemiAx = e[0]>e[1] ? e[1]*e[1] : e[0]*e[0];
  const Real cosang = std::cos(angle), sinang = std::sin(angle);
  const Real bbox[2][2] = {
   { pos[0] -std::max(e0,e1) -safety, pos[0] +std::max(e0,e1) +safety },
   { pos[1] -std::max(e0,e1) -safety, pos[1] +std::max(e0,e1) +safety }
  };

  FillBlocks_Ellipse(const Real _e0, const Real _e1, const Real h,
    const Real C[2], Real ang, Real rho): e0(_e0), e1(_e1), safety(2*h),
    pos{(Real)C[0], (Real)C[1]}, angle(ang), rhoS(rho) {}

  inline bool is_touching(const cubism::BlockInfo& INFO) const {
    return _is_touching(INFO, bbox, safety);
  }

  void operator()(const cubism::BlockInfo&, ScalarBlock&, ObstacleBlock&) const;
};

struct FillBlocks_Rectangle
{
  const Real extentX, extentY, safety, pos[2], angle, rhoS;
  const Real cosang = std::cos(angle), sinang = std::sin(angle);
  const Real bbox[2][2] = {
    { pos[0] - extentX - extentY - safety, pos[0] + extentX + extentY + safety},
    { pos[1] - extentX - extentY - safety, pos[1] + extentX + extentY + safety}
  };

  FillBlocks_Rectangle(Real _extentX, Real _extentY, Real h, const Real C[2], Real ang, Real rho): extentX(_extentX), extentY(_extentY), safety(h*2), pos{ (Real) C[0], (Real) C[1] }, angle(ang), rhoS(rho) { }

  inline Real distance(const Real x, const Real y) const {
    const Real X =  x*cosang + y*sinang, Y = -x*sinang + y*cosang;
    return std::min(extentX / 2 - std::abs(X), extentY / 2 - std::abs(Y));
  }

  inline bool is_touching(const cubism::BlockInfo& INFO) const {
    return _is_touching(INFO, bbox, safety);
  }

  void operator()(const cubism::BlockInfo&, ScalarBlock&, ObstacleBlock&) const;
};

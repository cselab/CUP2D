//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "ShapeLibrary.h"
#include <bits/stdc++.h>
using namespace cubism;

//static inline Real mollified_heaviside(const Real x) {
//  const Real alpha = M_PI * std::min( (Real)1, std::max( (Real)0, (x+1)/2 ) );
//  return 0.5 + 0.5 * std::cos( alpha );
//}

static Real distPointEllipseSpecial(const Real e[2],const Real y[2],Real x[2]);
static Real distPointEllipse(const Real e[2], const Real y[2], Real x[2]);

void FillBlocks_Cylinder::operator()(const BlockInfo& I,
                                         ScalarBlock& B,
                                       ObstacleBlock& O) const
{
  if( _is_touching(I, bbox, safety) )
  for(int iy=0; iy<ObstacleBlock::sizeY; iy++)
  for(int ix=0; ix<ObstacleBlock::sizeX; ix++)
  {
    Real p[2]; I.pos(p, ix, iy); p[0] -= pos[0]; p[1] -= pos[1];
    const Real dist = distanceTocylinder(p[0], p[1]);
    if( dist > O.dist[iy][ix] ) {
      O.dist[iy][ix] = dist;
      B(ix,iy).s = std::max( B(ix,iy).s, dist );
      O.rho[iy][ix] = rhoS;
    }
  }
}

void FillBlocks_HalfCylinder::operator()(const BlockInfo& I,
                                             ScalarBlock& B,
                                           ObstacleBlock& O) const
{
  if( _is_touching(I, bbox, safety) )
  {
    for(int iy=0; iy<ObstacleBlock::sizeY; iy++)
    for(int ix=0; ix<ObstacleBlock::sizeX; ix++)
    {
      Real p[2]; I.pos(p, ix, iy); p[0] -= pos[0]; p[1] -= pos[1];
      const Real dist = distanceTocylinder(p[0], p[1]);
      if( dist > O.dist[iy][ix] ) {
        O.dist[iy][ix] = dist;
        B(ix,iy).s = std::max( B(ix,iy).s, dist );
        O.rho[iy][ix] = rhoS;
      }
    }
  }
}

void FillBlocks_Rectangle::operator()(const BlockInfo& I,
                                      ScalarBlock& B,
                                    ObstacleBlock& O) const
{
  if( _is_touching(I, bbox, safety) )
  {
    for(int iy=0; iy<ObstacleBlock::sizeY; iy++)
    for(int ix=0; ix<ObstacleBlock::sizeX; ix++)
    {
      Real p[2]; I.pos(p, ix, iy); p[0] -= pos[0]; p[1] -= pos[1];
      const Real dist = distance(p[0], p[1]);
      if( dist > O.dist[iy][ix] ) {
        O.dist[iy][ix] = dist;
        B(ix,iy).s = std::max( B(ix,iy).s, dist );
        O.rho[iy][ix] = rhoS;
      }
    }
  }
}

void FillBlocks_Ellipse::operator()(const BlockInfo& I,
                                        ScalarBlock& B,
                                      ObstacleBlock& O) const
{
  if( _is_touching(I, bbox, safety) )
  {
    for(int iy=0; iy<ObstacleBlock::sizeY; iy++)
    for(int ix=0; ix<ObstacleBlock::sizeX; ix++)
    {
      Real p[2], xs[2];
      I.pos(p, ix, iy); p[0] -= pos[0]; p[1] -= pos[1];
      const Real t[2] = {cosang*p[0]+sinang*p[1], cosang*p[1]-sinang*p[0]};
      const Real sqDist = p[0]*p[0] + p[1]*p[1];
      Real dist = 0;
      if (std::fabs(t[0]) > e[0]+safety || std::fabs(t[1]) > e[1]+safety )
        dist = -1; //is outside
      //else if (sqDist + safety*safety < sqMinSemiAx)
      //  dist =  1; //is inside
      else {
        const Real absdist = distPointEllipse (e, t, xs);
        const int sign = sqDist > (xs[0]*xs[0]+xs[1]*xs[1]) ? -1 : 1;
        dist = sign * absdist;
      }
      if( dist > O.dist[iy][ix] ) {
        O.dist[iy][ix] = dist;
        B(ix,iy).s = std::max( B(ix,iy).s, dist );
        O.rho[iy][ix] = rhoS;
      }
    }
  }
}
void FillBlocks_ElasticDisk::operator()(const BlockInfo& I,
                                        const VectorBlock& invmB,
                                        ScalarBlock& B,
                                        ObstacleBlock& O) const
{
    //retrieve sdf
    for(int iy=0; iy<ObstacleBlock::sizeY; iy++)
    for(int ix=0; ix<ObstacleBlock::sizeX; ix++)
    {
      if(invmB(ix,iy).u[0]==0&&invmB(ix,iy).u[1]==0) continue;
      Real p[2]={invmB(ix,iy).u[0]-center[0],invmB(ix,iy).u[1]-center[1]};
      const Real dist = distanceToDisk(p[0], p[1]);
      if( dist > O.dist[iy][ix] ) {
        O.dist[iy][ix] = dist;
        B(ix,iy).s = std::max( B(ix,iy).s, dist );
        O.rho[iy][ix] = rhoS;
      }
    }
}
void FastMarching::operator()(ScalarLab& lab,const BlockInfo& info) {
  // use fast marching method to reinitialize signed distance
  /*
  ==========================================================================
  # modified version using bellman-ford 
  initialize boundary(second order) ---skip for now
  for each point, if sdf(i,j)>0 continue, 
  else, use eikonal equation to update
  ** boundary case:
  1. inner boundary: keep static
  2. outter boundary: uninitialized value=0
  ==========================================================================
  */
  if(OBLOCK[info.blockID]==nullptr) return;
  ObstacleBlock& o = * OBLOCK[info.blockID];
  CHI_MAT & __restrict__ sdf = o.dist;
  for (int iy = 0; iy < ScalarBlock::sizeY; ++iy)
	for (int ix = 0; ix < ScalarBlock::sizeX; ++ix)
  {
    if(lab(ix,iy).s>0) continue;
    Real Ux,Uy;
    const Real h = info.h;
    // Outter boundary
    if (lab(ix-1,iy).s==signal) Ux=lab(ix+1,iy).s;
    else if (lab(ix+1,iy).s==signal) Ux=lab(ix-1,iy).s;
    else Ux=std::max(lab(ix-1,iy).s,lab(ix+1,iy).s);

    if (lab(ix,iy-1).s==signal) Uy=lab(ix,iy+1).s;
    else if (lab(ix,iy+1).s==signal) Uy=lab(ix,iy-1).s;
    else Uy=std::max(lab(ix,iy-1).s,lab(ix,iy+1).s);
    //inner boundary
    if(Ux>0&&Uy>0) 
    {
      //update bounding box
      Real p[2];
      info.pos(p,ix,iy);
      minx=std::min(minx,p[0]);
      maxx=std::max(maxx,p[0]);
      miny=std::min(miny,p[1]);
      maxy=std::max(maxy,p[1]);
      continue;
    }
    if(std::abs(Ux-Uy)<=h) 
      lab(ix,iy).s=0.5*(Ux+Uy-std::sqrt((Ux+Uy)*(Ux+Uy)-2*(Ux*Ux+Uy*Uy-h*h)));
    else
      lab(ix,iy).s=std::max(Ux-h,Uy-h);
    sdf[iy][ix]=lab(ix,iy).s;
  }
  
  
  /**=========================================================================*/
  //  original fast marching algorithm
  // 1.1) maintain a min heap of undecided nodes(blockID,i,j)
  // 1.2) keep track of decided and undecided
  // c.2) search neighbors of decided
  
  // find smallest value, mark it as decided
  // iteration
      // if it is at the edge load the lab, otherwise pull the neighbors straightforwardly
      // load lab 
      // find the neighbors
      // update the neighbors (change values in the heap)
      // how to update the neighbors in other grid
      
      // find next smallest in the miniheap

  
}
Real distPointEllipseSpecial(const Real e[2], const Real y[2], Real x[2])
{
  static constexpr int imax = 2*std::numeric_limits<Real>::max_exponent;
  static constexpr Real eps = std::numeric_limits<Real>::epsilon();
  if (y[1] > (Real)0) {
    if (y[0] > (Real)0) {
      // Bisect to compute the root of F(t) for t >= -e1*e1.
      const Real esqr[2] = { e[0]*e[0], e[1]*e[1] };
      const Real ey[2] = { e[0]*y[0], e[1]*y[1] };
      Real t0 = -esqr[1] + ey[1];
      Real t1 = -esqr[1] + sqrt(ey[0]*ey[0] + ey[1]*ey[1]);
      Real t = t0;
      for (int i = 0; i < imax; ++i) {
        t = ((Real)0.5)*(t0 + t1);
        if ( std::fabs(t-t0)<eps || std::fabs(t-t1)<eps ) break;

        const Real r[2] = { ey[0]/(t + esqr[0]), ey[1]/(t + esqr[1]) };
        const Real f = r[0]*r[0] + r[1]*r[1] - (Real)1;
        if (f > (Real)0) t0 = t;
        else if (f < (Real)0) t1 = t;
        else break;
      }

      x[0] = esqr[0]*y[0]/(t + esqr[0]);
      x[1] = esqr[1]*y[1]/(t + esqr[1]);
      const Real d[2] = { x[0] - y[0], x[1] - y[1] };
      return std::sqrt(d[0]*d[0] + d[1]*d[1]);
    } else { // y0 == 0
      x[0] = (Real)0;
      x[1] = e[1];
      return std::fabs(y[1] - e[1]);
    }
  } else { // y1 == 0
    const Real denom0 = e[0]*e[0] - e[1]*e[1];
    const Real e0y0 = e[0]*y[0];
    if (e0y0 < denom0) {
      // y0 is inside the subinterval.
      const Real x0de0 = e0y0/denom0;
      const Real x0de0sqr = x0de0*x0de0;
      x[0] = e[0]*x0de0;
      x[1] = e[1]*std::sqrt(std::fabs((Real)1 - x0de0sqr));
      const Real d0 = x[0] - y[0];
      return std::sqrt(d0*d0 + x[1]*x[1]);
    } else {
      // y0 is outside the subinterval.  The closest ellipse point has
      // x1 == 0 and is on the domain-boundary interval (x0/e0)^2 = 1.
      x[0] = e[0];
      x[1] = (Real)0;
      return std::fabs(y[0] - e[0]);
    }
  }
}
//----------------------------------------------------------------------------
// The ellipse is (x0/e0)^2 + (x1/e1)^2 = 1.  The query point is (y0,y1).
// The function returns the distance from the query point to the ellipse.
// It also computes the ellipse point (x0,x1) that is closest to (y0,y1).
//----------------------------------------------------------------------------

Real distPointEllipse(const Real e[2], const Real y[2], Real x[2])
{
  // Determine reflections for y to the first quadrant.
  bool reflect[2];
  for (int i = 0; i < 2; ++i) reflect[i] = (y[i] < (Real)0);

  // Determine the axis order for decreasing extents.
  int permute[2];
  if (e[0] < e[1]) { permute[0] = 1;  permute[1] = 0; }
  else { permute[0] = 0;  permute[1] = 1; }

  int invpermute[2];
  for (int i = 0; i < 2; ++i) invpermute[permute[i]] = i;

  Real locE[2], locY[2];
  for (int i = 0; i < 2; ++i) {
    const int j = permute[i];
    locE[i] = e[j];
    locY[i] = y[j];
    if (reflect[j]) locY[i] = -locY[i];
  }

  Real locX[2];
  const Real distance = distPointEllipseSpecial(locE, locY, locX);

  // Restore the axis order and reflections.
  for (int i = 0; i < 2; ++i) {
    const int j = invpermute[i];
    if (reflect[j]) locX[j] = -locX[j];
    x[i] = locX[j];
  }

  return distance;
}

//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//


//#include <cstdio>
//#include <iostream>
#include <sstream>
#include <string>
#include <fstream>


#include "ComputeForces.h"
#include "../Shape.h"

using UDEFMAT = Real[VectorBlock::sizeY][VectorBlock::sizeX][2];

struct KernelComputeForces
{
  KernelComputeForces(const SimulationData & s) : sim(s) {}
  const SimulationData & sim;
  cubism::StencilInfo stencil{-4, -4, 0, 5, 5, 1, true, {0,1}};
  cubism::StencilInfo stencil2{-4, -4, 0, 5, 5, 1, true, {0}};
  const std::vector<cubism::BlockInfo>& presInfo = sim.pres->getBlocksInfo();

  void operator()(VectorLab & lab, ScalarLab & chi, const cubism::BlockInfo& info, const cubism::BlockInfo& info2) const
  {
    VectorLab & V = lab;
    ScalarBlock & __restrict__ P = *(ScalarBlock*) presInfo[info.blockID].ptrBlock;

    const int big   = ScalarBlock::sizeX + 4;
    const int small = -4;
    for(const auto& _shape : sim.shapes)
    {
      const Shape * const shape = _shape.get();
      const std::vector<ObstacleBlock*> & OBLOCK = shape->obstacleBlocks;
      const Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];
      const Real vel_norm = std::sqrt(shape->u*shape->u + shape->v*shape->v);
      const Real vel_unit[2] = {
        vel_norm>0? (Real) shape->u / vel_norm : (Real)0,
        vel_norm>0? (Real) shape->v / vel_norm : (Real)0
      };
   
      const Real NUoH = sim.nu / info.h; // 2 nu / 2 h
      ObstacleBlock * const O = OBLOCK[info.blockID];
      if (O == nullptr) continue;
      assert(O->filled);
      for(size_t k = 0; k < O->n_surfPoints; ++k)
      {
        const int ix = O->surface[k]->ix, iy = O->surface[k]->iy;
        const std::array<Real,2> p = info.pos<Real>(ix, iy);

        const Real normX = O->surface[k]->dchidx; //*h^3 (multiplied in dchidx)
        const Real normY = O->surface[k]->dchidy; //*h^3 (multiplied in dchidy)
        const Real norm = 1.0/std::sqrt(normX*normX+normY*normY);
        const Real dx = normX*norm;
        const Real dy = normY*norm;
        //shear stresses
        //"lifted" surface: derivatives make no sense when the values used are in the object, 
        // so we take one-sided stencils with values outside of the object
        //Real D11 = 0.0;
        //Real D22 = 0.0;
        //Real D12 = 0.0;
        Real DuDx;
        Real DuDy;
        Real DvDx;
        Real DvDy;
        {
          //The integers x and y will be the coordinates of the point on the lifted surface.
          //To find them, we move along the normal vector to the surface, until we find a point
          //outside of the object (where chi = 0).
          int x = ix;
          int y = iy;
          const Real aa = std::atan2(dy,dx);
          const Real dx_a = cos(aa);
          const Real dy_a = sin(aa);
          int found = 0;
          for (int kk = 1 ; kk < 10 ; kk++) //10 is arbitrary
          {
            if ((int)abs(kk*dx_a) > 3 || (int)abs(kk*dy_a) > 3) break; //3 means we moved too far
            if (chi(x,y).s <3e-1 && found >= 1) break;

	    if (ix + kk*dx_a + 1 >= ScalarBlock::sizeX + big-1 || ix + kk*dx_a -1 < small) break;
            if (iy + kk*dy_a + 1 >= ScalarBlock::sizeY + big-1 || iy + kk*dy_a -1 < small) break;

            x  = ix + kk*dx_a; 
            y  = iy + kk*dy_a;
            if (chi(x,y).s < 1e-3 ) found ++;
          }

          //Now that we found the (x,y) of the point, we compute grad(u) there.
          //grad(u) is computed with biased stencils. If available, larger stencils are used.
          //Then, we compute higher order derivatives that are used to form a Taylor expansion
          //around (x,y). Finally, this expansion is used to extrapolate grad(u) to (ix,iy) of 
          //the actual solid surface. 

          Real dudx1 = normX > 0 ? (V(x+1,y).u[0]-V(x,y).u[0]) : (V(x,y).u[0]-V(x-1,y).u[0]);
          Real dvdx1 = normX > 0 ? (V(x+1,y).u[1]-V(x,y).u[1]) : (V(x,y).u[1]-V(x-1,y).u[1]);
          Real dudy1 = normY > 0 ? (V(x,y+1).u[0]-V(x,y).u[0]) : (V(x,y).u[0]-V(x,y-1).u[0]);
          Real dvdy1 = normY > 0 ? (V(x,y+1).u[1]-V(x,y).u[1]) : (V(x,y).u[1]-V(x,y-1).u[1]);
          Real dudxdy1 = 0.0;
          Real dvdxdy1 = 0.0;
	  /*
          Real dudy2dx = 0.0;
          Real dvdy2dx = 0.0;
          Real dudx2dy = 0.0;
          Real dvdx2dy = 0.0;
	  */
          Real dudx2 = 0.0;
          Real dvdx2 = 0.0;
          Real dudy2 = 0.0;
          Real dvdy2 = 0.0;
	  /*
          Real dudx3 = 0.0;
          Real dvdx3 = 0.0;
          Real dudy3 = 0.0;
          Real dvdy3 = 0.0;
          */
          if (normX > 0 && normY > 0)
          {
            dudxdy1 = (V(x+1,y+1).u[0]+V(x,y).u[0]-V(x+1,y).u[0]-V(x,y+1).u[0]);
            dvdxdy1 = (V(x+1,y+1).u[1]+V(x,y).u[1]-V(x+1,y).u[1]-V(x,y+1).u[1]);
            if ((x+2 < big) && (y+2 < big))
            {
               dudxdy1 = -0.5*( -1.5*V(x+2,y).u[0]+2*V(x+2,y+1).u[0]-0.5*V(x+2,y+2).u[0] ) + 2*(-1.5*V(x+1,y).u[0]+2*V(x+1,y+1).u[0]-0.5*V(x+1,y+2).u[0]) -1.5*(-1.5*V(x,y).u[0]+2*V(x,y+1).u[0]-0.5*V(x,y+2).u[0]);
               dvdxdy1 = -0.5*( -1.5*V(x+2,y).u[1]+2*V(x+2,y+1).u[1]-0.5*V(x+2,y+2).u[1] ) + 2*(-1.5*V(x+1,y).u[1]+2*V(x+1,y+1).u[1]-0.5*V(x+1,y+2).u[1]) -1.5*(-1.5*V(x,y).u[1]+2*V(x,y+1).u[1]-0.5*V(x,y+2).u[1]);
            }
	    /*
            if (x+3 < big && y+2 < big)
            {
              Real dudx2_yplus2= 2.0*V(x,y+2).u[0]-5.0*V(x+1,y+2).u[0]+4*V(x+2,y+2).u[0]-V(x+3,y+2).u[0];
              Real dudx2_yplus = 2.0*V(x,y+1).u[0]-5.0*V(x+1,y+1).u[0]+4*V(x+2,y+1).u[0]-V(x+3,y+1).u[0];
              Real dudx2_y     = 2.0*V(x,y  ).u[0]-5.0*V(x+1,y  ).u[0]+4*V(x+2,y  ).u[0]-V(x+3,y  ).u[0];
              Real dvdx2_yplus2= 2.0*V(x,y+2).u[1]-5.0*V(x+1,y+2).u[1]+4*V(x+2,y+2).u[1]-V(x+3,y+2).u[1];
              Real dvdx2_yplus = 2.0*V(x,y+1).u[1]-5.0*V(x+1,y+1).u[1]+4*V(x+2,y+1).u[1]-V(x+3,y+1).u[1];
              Real dvdx2_y     = 2.0*V(x,y  ).u[1]-5.0*V(x+1,y  ).u[1]+4*V(x+2,y  ).u[1]-V(x+3,y  ).u[1];
              //dudx2dy = dudx2_yplus - dudx2_y;
              //dvdx2dy = dvdx2_yplus - dvdx2_y;
              dudx2dy = -0.5*dudx2_yplus2 + 2.0*dudx2_yplus - 1.5*dudx2_y;
              dvdx2dy = -0.5*dvdx2_yplus2 + 2.0*dvdx2_yplus - 1.5*dvdx2_y;
            }
            if (y+3 < big && x+2 < big)
            {
              Real dudy2_xplus2= 2.0*V(x+2,y).u[0]-5.0*V(x+2,y+1).u[0]+4*V(x+2,y+2).u[0]-V(x+2,y+3).u[0];
              Real dudy2_xplus = 2.0*V(x+1,y).u[0]-5.0*V(x+1,y+1).u[0]+4*V(x+1,y+2).u[0]-V(x+1,y+3).u[0];
              Real dudy2_x     = 2.0*V(x  ,y).u[0]-5.0*V(x  ,y+1).u[0]+4*V(x  ,y+2).u[0]-V(x  ,y+3).u[0];
              Real dvdy2_xplus2= 2.0*V(x+2,y).u[1]-5.0*V(x+2,y+1).u[1]+4*V(x+2,y+2).u[1]-V(x+2,y+3).u[1];
              Real dvdy2_xplus = 2.0*V(x+1,y).u[1]-5.0*V(x+1,y+1).u[1]+4*V(x+1,y+2).u[1]-V(x+1,y+3).u[1];
              Real dvdy2_x     = 2.0*V(x  ,y).u[1]-5.0*V(x  ,y+1).u[1]+4*V(x  ,y+2).u[1]-V(x  ,y+3).u[1];
              //dudy2dx = dudy2_xplus - dudy2_x;
              //dvdy2dx = dvdy2_xplus - dvdy2_x;
              dudy2dx = -0.5*dudy2_xplus2 +  2.0*dudy2_xplus - 1.5*dudy2_x;
              dvdy2dx = -0.5*dvdy2_xplus2 +  2.0*dvdy2_xplus - 1.5*dvdy2_x;
            }
	    */
          }
          if (normX < 0 && normY > 0)
          {
            dudxdy1 = (V(x,y+1).u[0]+V(x-1,y).u[0]-V(x,y).u[0]-V(x-1,y+1).u[0]);
            dvdxdy1 = (V(x,y+1).u[1]+V(x-1,y).u[1]-V(x,y).u[1]-V(x-1,y+1).u[1]);
            if ((y+2 < big) && (x-2 >= small))
            {
               dudxdy1 = 0.5*( -1.5*V(x-2,y).u[0]+2*V(x-2,y+1).u[0]-0.5*V(x-2,y+2).u[0] ) - 2*(-1.5*V(x-1,y).u[0]+2*V(x-1,y+1).u[0]-0.5*V(x-1,y+2).u[0])+1.5*(-1.5*V(x,y).u[0]+2*V(x,y+1).u[0]-0.5*V(x,y+2).u[0]);
               dvdxdy1 = 0.5*( -1.5*V(x-2,y).u[1]+2*V(x-2,y+1).u[1]-0.5*V(x-2,y+2).u[1] ) - 2*(-1.5*V(x-1,y).u[1]+2*V(x-1,y+1).u[1]-0.5*V(x-1,y+2).u[1])+1.5*(-1.5*V(x,y).u[1]+2*V(x,y+1).u[1]-0.5*V(x,y+2).u[1]);
            }
            /*
            if (x-3 >=small && y+2 < big)
            {
              Real dudx2_yplus2= 2.0*V(x,y+2).u[0]-5.0*V(x-1,y+2).u[0]+4*V(x-2,y+2).u[0]-V(x-3,y+2).u[0];
              Real dudx2_yplus = 2.0*V(x,y+1).u[0]-5.0*V(x-1,y+1).u[0]+4*V(x-2,y+1).u[0]-V(x-3,y+1).u[0];
              Real dudx2_y     = 2.0*V(x,y  ).u[0]-5.0*V(x-1,y  ).u[0]+4*V(x-2,y  ).u[0]-V(x-3,y  ).u[0];
              Real dvdx2_yplus2= 2.0*V(x,y+2).u[1]-5.0*V(x-1,y+2).u[1]+4*V(x-2,y+2).u[1]-V(x-3,y+2).u[1];
              Real dvdx2_yplus = 2.0*V(x,y+1).u[1]-5.0*V(x-1,y+1).u[1]+4*V(x-2,y+1).u[1]-V(x-3,y+1).u[1];
              Real dvdx2_y     = 2.0*V(x,y  ).u[1]-5.0*V(x-1,y  ).u[1]+4*V(x-2,y  ).u[1]-V(x-3,y  ).u[1];
              //dudx2dy = dudx2_yplus - dudx2_y;
              //dvdx2dy = dvdx2_yplus - dvdx2_y;
              dudx2dy = -0.5*dudx2_yplus2 + 2.0*dudx2_yplus -1.5*dudx2_y;
              dvdx2dy = -0.5*dvdx2_yplus2 + 2.0*dvdx2_yplus -1.5*dvdx2_y;
            }
            if (y+3 < big && x-2>=small)
            {
              Real dudy2_xminus2= 2.0*V(x-2,y).u[0]-5.0*V(x-2,y+1).u[0]+4*V(x-2,y+2).u[0]-V(x-2,y+3).u[0];
              Real dudy2_xminus = 2.0*V(x-1,y).u[0]-5.0*V(x-1,y+1).u[0]+4*V(x-1,y+2).u[0]-V(x-1,y+3).u[0];
              Real dudy2_x      = 2.0*V(x  ,y).u[0]-5.0*V(x  ,y+1).u[0]+4*V(x  ,y+2).u[0]-V(x  ,y+3).u[0];
              Real dvdy2_xminus2= 2.0*V(x-2,y).u[1]-5.0*V(x-2,y+1).u[1]+4*V(x-2,y+2).u[1]-V(x-2,y+3).u[1];
              Real dvdy2_xminus = 2.0*V(x-1,y).u[1]-5.0*V(x-1,y+1).u[1]+4*V(x-1,y+2).u[1]-V(x-1,y+3).u[1];
              Real dvdy2_x      = 2.0*V(x  ,y).u[1]-5.0*V(x  ,y+1).u[1]+4*V(x  ,y+2).u[1]-V(x  ,y+3).u[1];
              //dudy2dx = dudy2_x - dudy2_xminus;
              //dvdy2dx = dvdy2_x - dvdy2_xminus;
              dudy2dx = 1.5*dudy2_x - 2.0*dudy2_xminus + 0.5*dudy2_xminus2;
              dvdy2dx = 1.5*dvdy2_x - 2.0*dvdy2_xminus + 0.5*dvdy2_xminus2;
            }
	    */
          }
          if (normX > 0 && normY < 0)
          {
            dudxdy1 = (V(x+1,y).u[0]+V(x,y-1).u[0]-V(x+1,y-1).u[0]-V(x,y).u[0]);
            dvdxdy1 = (V(x+1,y).u[1]+V(x,y-1).u[1]-V(x+1,y-1).u[1]-V(x,y).u[1]);
            if ((x+2 < big) && (y-2 >= small))
            {
               dudxdy1 = -0.5*( 1.5*V(x+2,y).u[0]-2*V(x+2,y-1).u[0]+0.5*V(x+2,y-2).u[0] ) + 2*(1.5*V(x+1,y).u[0]-2*V(x+1,y-1).u[0]+0.5*V(x+1,y-2).u[0]) -1.5*(1.5*V(x,y).u[0]-2*V(x,y-1).u[0]+0.5*V(x,y-2).u[0]);
               dvdxdy1 = -0.5*( 1.5*V(x+2,y).u[1]-2*V(x+2,y-1).u[1]+0.5*V(x+2,y-2).u[1] ) + 2*(1.5*V(x+1,y).u[1]-2*V(x+1,y-1).u[1]+0.5*V(x+1,y-2).u[1]) -1.5*(1.5*V(x,y).u[1]-2*V(x,y-1).u[1]+0.5*V(x,y-2).u[1]);
            }
            /*
            if (x+3 < big && y-2>=small)
            {
              Real dudx2_yminus2= 2.0*V(x,y-2).u[0]-5.0*V(x+1,y-2).u[0]+4*V(x+2,y-2).u[0]-V(x+3,y-2).u[0];
              Real dudx2_yminus = 2.0*V(x,y-1).u[0]-5.0*V(x+1,y-1).u[0]+4*V(x+2,y-1).u[0]-V(x+3,y-1).u[0];
              Real dudx2_y      = 2.0*V(x,y  ).u[0]-5.0*V(x+1,y  ).u[0]+4*V(x+2,y  ).u[0]-V(x+3,y  ).u[0];
              Real dvdx2_yminus2= 2.0*V(x,y-2).u[1]-5.0*V(x+1,y-2).u[1]+4*V(x+2,y-2).u[1]-V(x+3,y-2).u[1];
              Real dvdx2_yminus = 2.0*V(x,y-1).u[1]-5.0*V(x+1,y-1).u[1]+4*V(x+2,y-1).u[1]-V(x+3,y-1).u[1];
              Real dvdx2_y      = 2.0*V(x,y  ).u[1]-5.0*V(x+1,y  ).u[1]+4*V(x+2,y  ).u[1]-V(x+3,y  ).u[1];
              //dudx2dy = dudx2_y - dudx2_yminus;
              //dvdx2dy = dvdx2_y - dvdx2_yminus;
              dudx2dy = 1.5*dudx2_y - 2.0*dudx2_yminus + 0.5*dudx2_yminus2;
              dvdx2dy = 1.5*dvdx2_y - 2.0*dvdx2_yminus + 0.5*dvdx2_yminus2;
            }
            if (y-3 >= small && x+2 < big)
            {
              Real dudy2_xplus2= 2.0*V(x+2,y).u[0]-5.0*V(x+2,y-1).u[0]+4*V(x+2,y-2).u[0]-V(x+2,y-3).u[0];
              Real dudy2_xplus = 2.0*V(x+1,y).u[0]-5.0*V(x+1,y-1).u[0]+4*V(x+1,y-2).u[0]-V(x+1,y-3).u[0];
              Real dudy2_x     = 2.0*V(x  ,y).u[0]-5.0*V(x  ,y-1).u[0]+4*V(x  ,y-2).u[0]-V(x  ,y-3).u[0];
              Real dvdy2_xplus2= 2.0*V(x+2,y).u[1]-5.0*V(x+2,y-1).u[1]+4*V(x+2,y-2).u[1]-V(x+2,y-3).u[1];
              Real dvdy2_xplus = 2.0*V(x+1,y).u[1]-5.0*V(x+1,y-1).u[1]+4*V(x+1,y-2).u[1]-V(x+1,y-3).u[1];
              Real dvdy2_x     = 2.0*V(x  ,y).u[1]-5.0*V(x  ,y-1).u[1]+4*V(x  ,y-2).u[1]-V(x  ,y-3).u[1];
              //dudy2dx = dudy2_xplus - dudy2_x;
              //dvdy2dx = dvdy2_xplus - dvdy2_x;
              dudy2dx = -0.5*dudy2_xplus2 + 2.0*dudy2_xplus - 1.5*dudy2_x;
              dvdy2dx = -0.5*dvdy2_xplus2 + 2.0*dvdy2_xplus - 1.5*dvdy2_x;
            }
	    */
          }
          if (normX < 0 && normY < 0)
          {
            dudxdy1 = (V(x,y).u[0]+V(x-1,y-1).u[0]-V(x,y-1).u[0]-V(x-1,y).u[0]);
            dvdxdy1 = (V(x,y).u[1]+V(x-1,y-1).u[1]-V(x,y-1).u[1]-V(x-1,y).u[1]);
            if ((x-2 >= small) && (y-2 >= small))
            {
               dudxdy1 = 0.5*( 1.5*V(x-2,y).u[0]-2*V(x-2,y-1).u[0]+0.5*V(x-2,y-2).u[0] ) - 2*(1.5*V(x-1,y).u[0]-2*V(x-1,y-1).u[0]+0.5*V(x-1,y-2).u[0]) +1.5*(1.5*V(x,y).u[0]-2*V(x,y-1).u[0]+0.5*V(x,y-2).u[0]);
               dvdxdy1 = 0.5*( 1.5*V(x-2,y).u[1]-2*V(x-2,y-1).u[1]+0.5*V(x-2,y-2).u[1] ) - 2*(1.5*V(x-1,y).u[1]-2*V(x-1,y-1).u[1]+0.5*V(x-1,y-2).u[1]) +1.5*(1.5*V(x,y).u[1]-2*V(x,y-1).u[1]+0.5*V(x,y-2).u[1]);
            }

            /*
            if (x-3 >= small && y-2>=small)
            {
              Real dudx2_yminus2= 2.0*V(x,y-2).u[0]-5.0*V(x-1,y-2).u[0]+4*V(x-2,y-2).u[0]-V(x-3,y-2).u[0];
              Real dudx2_yminus = 2.0*V(x,y-1).u[0]-5.0*V(x-1,y-1).u[0]+4*V(x-2,y-1).u[0]-V(x-3,y-1).u[0];
              Real dudx2_y      = 2.0*V(x,y  ).u[0]-5.0*V(x-1,y  ).u[0]+4*V(x-2,y  ).u[0]-V(x-3,y  ).u[0];
              Real dvdx2_yminus2= 2.0*V(x,y-2).u[1]-5.0*V(x-1,y-2).u[1]+4*V(x-2,y-2).u[1]-V(x-3,y-2).u[1];
              Real dvdx2_yminus = 2.0*V(x,y-1).u[1]-5.0*V(x-1,y-1).u[1]+4*V(x-2,y-1).u[1]-V(x-3,y-1).u[1];
              Real dvdx2_y      = 2.0*V(x,y  ).u[1]-5.0*V(x-1,y  ).u[1]+4*V(x-2,y  ).u[1]-V(x-3,y  ).u[1];
              //dudx2dy = dudx2_y - dudx2_yminus;
              //dvdx2dy = dvdx2_y - dvdx2_yminus;
              dudx2dy = 1.5*dudx2_y - 2.0*dudx2_yminus + 0.5*dudx2_yminus2;
              dvdx2dy = 1.5*dvdx2_y - 2.0*dvdx2_yminus + 0.5*dvdx2_yminus2;
            }
            if (y-3 >= small && x-2>=small)
            {
              Real dudy2_xminus2= 2.0*V(x-2,y).u[0]-5.0*V(x-2,y-1).u[0]+4*V(x-2,y-2).u[0]-V(x-2,y-3).u[0];
              Real dudy2_xminus = 2.0*V(x-1,y).u[0]-5.0*V(x-1,y-1).u[0]+4*V(x-1,y-2).u[0]-V(x-1,y-3).u[0];
              Real dudy2_x      = 2.0*V(x  ,y).u[0]-5.0*V(x  ,y-1).u[0]+4*V(x  ,y-2).u[0]-V(x  ,y-3).u[0];
              Real dvdy2_xminus2= 2.0*V(x-2,y).u[1]-5.0*V(x-2,y-1).u[1]+4*V(x-2,y-2).u[1]-V(x-2,y-3).u[1];
              Real dvdy2_xminus = 2.0*V(x-1,y).u[1]-5.0*V(x-1,y-1).u[1]+4*V(x-1,y-2).u[1]-V(x-1,y-3).u[1];
              Real dvdy2_x      = 2.0*V(x  ,y).u[1]-5.0*V(x  ,y-1).u[1]+4*V(x  ,y-2).u[1]-V(x  ,y-3).u[1];
              //dudy2dx = dudy2_x - dudy2_xminus;
              //dvdy2dx = dvdy2_x - dvdy2_xminus;
              dudy2dx = 1.5*dudy2_x - 2.0*dudy2_xminus + 0.5*dudy2_xminus2;
              dvdy2dx = 1.5*dvdy2_x - 2.0*dvdy2_xminus + 0.5*dvdy2_xminus2;
            }
	    */
          }
          
          if (normX > 0 && x+2 <    big)
          {
            dudx1 = -1.5*V(x,y).u[0]+2.0*V(x+1,y).u[0]-0.5*V(x+2,y).u[0];
            dvdx1 = -1.5*V(x,y).u[1]+2.0*V(x+1,y).u[1]-0.5*V(x+2,y).u[1];
            dudx2 =      V(x,y).u[0]-2.0*V(x+1,y).u[0]+    V(x+2,y).u[0];
            dvdx2 =      V(x,y).u[1]-2.0*V(x+1,y).u[1]+    V(x+2,y).u[1];
          }
          if (normX < 0 && x-2 >= small)
          {
            dudx1 =  1.5*V(x,y).u[0]-2.0*V(x-1,y).u[0]+0.5*V(x-2,y).u[0];
            dvdx1 =  1.5*V(x,y).u[1]-2.0*V(x-1,y).u[1]+0.5*V(x-2,y).u[1];
            dudx2 =      V(x,y).u[0]-2.0*V(x-1,y).u[0]+    V(x-2,y).u[0];
            dvdx2 =      V(x,y).u[1]-2.0*V(x-1,y).u[1]+    V(x-2,y).u[1];
          }
          if (normY > 0 && y+2 <    big)
          {
            dudy1 = -1.5*V(x,y).u[0]+2.0*V(x,y+1).u[0]-0.5*V(x,y+2).u[0];
            dvdy1 = -1.5*V(x,y).u[1]+2.0*V(x,y+1).u[1]-0.5*V(x,y+2).u[1];
            dudy2 =      V(x,y).u[0]-2.0*V(x,y+1).u[0]+    V(x,y+2).u[0];
            dvdy2 =      V(x,y).u[1]-2.0*V(x,y+1).u[1]+    V(x,y+2).u[1];
          }
          if (normY < 0 && y-2 >= small)
          {
            dudy1 =  1.5*V(x,y).u[0]-2.0*V(x,y-1).u[0]+0.5*V(x,y-2).u[0];
            dvdy1 =  1.5*V(x,y).u[1]-2.0*V(x,y-1).u[1]+0.5*V(x,y-2).u[1];
            dudy2 =      V(x,y).u[0]-2.0*V(x,y-1).u[0]+    V(x,y-2).u[0];
            dvdy2 =      V(x,y).u[1]-2.0*V(x,y-1).u[1]+    V(x,y-2).u[1];
          }
	  /*
          if (normX > 0 && x+3 <    big)
          {
            dudx3 = -V(x,y).u[0] + 3*V(x+1,y).u[0] - 3*V(x+2,y).u[0] + V(x+3,y).u[0]; 
            dvdx3 = -V(x,y).u[1] + 3*V(x+1,y).u[1] - 3*V(x+2,y).u[1] + V(x+3,y).u[1];
          }
          if (normX < 0 && x-3 >= small)
          {
            dudx3 =  V(x,y).u[0] - 3*V(x-1,y).u[0] + 3*V(x-2,y).u[0] - V(x-3,y).u[0]; 
            dvdx3 =  V(x,y).u[1] - 3*V(x-1,y).u[1] + 3*V(x-2,y).u[1] - V(x-3,y).u[1];
          }
          if (normY > 0 && y+3 <    big) 
          {
            dudy3 = -V(x,y).u[0] + 3*V(x,y+1).u[0] - 3*V(x,y+2).u[0] + V(x,y+3).u[0];
            dvdy3 = -V(x,y).u[1] + 3*V(x,y+1).u[1] - 3*V(x,y+2).u[1] + V(x,y+3).u[1];
          }
          if (normY < 0 && y-3 >= small)
          {
            dudy3 =  V(x,y).u[0] - 3*V(x,y-1).u[0] + 3*V(x,y-2).u[0] - V(x,y-3).u[0];
            dvdy3 =  V(x,y).u[1] - 3*V(x,y-1).u[1] + 3*V(x,y-2).u[1] - V(x,y-3).u[1];
          }
          const Real dudx = dudx1 + dudx2*(ix-x)+ dudxdy1*(iy-y) + 0.5*dudx3*(ix-x)*(ix-x) +     dudx2dy*(ix-x)*(iy-y) + 0.5*dudy2dx*(iy-y)*(iy-y);
          const Real dvdx = dvdx1 + dvdx2*(ix-x)+ dvdxdy1*(iy-y) + 0.5*dvdx3*(ix-x)*(ix-x) +     dvdx2dy*(ix-x)*(iy-y) + 0.5*dvdy2dx*(iy-y)*(iy-y);
          const Real dudy = dudy1 + dudy2*(iy-y)+ dudxdy1*(ix-x) + 0.5*dudy3*(iy-y)*(iy-y) + 0.5*dudx2dy*(ix-x)*(ix-x) +     dudy2dx*(ix-x)*(iy-y);
          const Real dvdy = dvdy1 + dvdy2*(iy-y)+ dvdxdy1*(ix-x) + 0.5*dvdy3*(iy-y)*(iy-y) + 0.5*dvdx2dy*(ix-x)*(ix-x) +     dvdy2dx*(ix-x)*(iy-y);
	  */
          const Real dudx = dudx1 + dudx2*(ix-x)+ dudxdy1*(iy-y);
          const Real dvdx = dvdx1 + dvdx2*(ix-x)+ dvdxdy1*(iy-y);
          const Real dudy = dudy1 + dudy2*(iy-y)+ dudxdy1*(ix-x);
          const Real dvdy = dvdy1 + dvdy2*(iy-y)+ dvdxdy1*(ix-x);
          //D11 = 2.0*NUoH*dudx;
          //D22 = 2.0*NUoH*dvdy;
          //D12 = NUoH*(dudy+dvdx);
          DuDx = dudx;
          DuDy = dudy;
          DvDx = dvdx;
          DvDy = dvdy;
        }//shear stress computation ends here

        //normals computed with Towers 2009
        // Actually using the volume integral, since (/iint -P /hat{n} dS) =
        // (/iiint - /nabla P dV). Also, P*/nabla /Chi = /nabla P
        // penalty-accel and surf-force match up if resolution is high enough
        //const Real fXV = D11*normX + D12*normY, fXP = - P(ix,iy).s * normX;
        //const Real fYV = D12*normX + D22*normY, fYP = - P(ix,iy).s * normY;
        const Real fXV = NUoH*DuDx*normX + NUoH*DuDy*normY, fXP = - P(ix,iy).s * normX;
        const Real fYV = NUoH*DvDx*normX + NUoH*DvDy*normY, fYP = - P(ix,iy).s * normY;

        const Real fXT = fXV + fXP, fYT = fYV + fYP;

        //store:
        O->x_s    [k] = p[0];
        O->y_s    [k] = p[1];
        O->p_s    [k] = P(ix,iy).s;
        O->u_s    [k] = V(ix,iy).u[0];
        O->v_s    [k] = V(ix,iy).u[1];
        O->nx_s   [k] = dx;
        O->ny_s   [k] = dy;
        O->omega_s[k] = (DvDx - DuDy)/info.h;
        O->uDef_s [k] = O->udef[iy][ix][0];
        O->vDef_s [k] = O->udef[iy][ix][1];
        O->fX_s   [k] = fXT;
        O->fY_s   [k] = fYT;
        O->fXv_s  [k] = fXV;
        O->fYv_s  [k] = fYV;

        //perimeter:
        O->perimeter += std::sqrt(normX*normX + normY*normY);
        O->circulation += normX * O->v_s[k] - normY * O->u_s[k];
        //forces (total, visc, pressure):
        O->forcex += fXT;
        O->forcey += fYT;
        O->forcex_V += fXV;
        O->forcey_V += fYV;
        O->forcex_P += fXP;
        O->forcey_P += fYP;
        //torque:
        O->torque   += (p[0] - Cx) * fYT - (p[1] - Cy) * fXT;
        O->torque_P += (p[0] - Cx) * fYP - (p[1] - Cy) * fXP;
        O->torque_V += (p[0] - Cx) * fYV - (p[1] - Cy) * fXV;
        //thrust, drag:
        const Real forcePar = fXT * vel_unit[0] + fYT * vel_unit[1];
        O->thrust += .5*(forcePar + std::fabs(forcePar));
        O->drag   -= .5*(forcePar - std::fabs(forcePar));
        const Real forcePerp = fXT * vel_unit[1] - fYT * vel_unit[0];
        O->lift   += forcePerp;
        //power output (and negative definite variant which ensures no elastic energy absorption)
        // This is total power, for overcoming not only deformation, but also the oncoming velocity. Work done by fluid, not by the object (for that, just take -ve)
        const Real powOut = fXT * O->u_s[k]    + fYT * O->v_s[k];
        //deformation power output (and negative definite variant which ensures no elastic energy absorption)
        const Real powDef = fXT * O->uDef_s[k] + fYT * O->vDef_s[k];
        O->Pout        += powOut;
        O->defPower    += powDef;
        O->PoutBnd     += std::min((Real)0, powOut);
        O->defPowerBnd += std::min((Real)0, powDef);
      }
      O->PoutNew = O->forcex*shape->u +  O->forcey*shape->v;
    } 
  }
};

void ComputeForces::operator()(const Real dt)
{
  sim.startProfiler("ComputeForces");
  KernelComputeForces K(sim);
  compute<KernelComputeForces,VectorGrid,VectorLab,ScalarGrid,ScalarLab>(K,*sim.vel,*sim.chi);

  // finalize partial sums
  for (const auto& shape : sim.shapes)
    shape->computeForces();
  sim.stopProfiler();
}

ComputeForces::ComputeForces(SimulationData& s) : Operator(s) { }

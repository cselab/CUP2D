#pragma once

#include <cassert>
#include <iostream>
#include <vector>
#include <math.h>

namespace cubism // AMR_CUBISM
{

/** 
 * @brief Hilbert Space-Filling Curve(SFC) in 2D.
 * 
 * The Quadtree of GridBlocks of a simulation is traversed by an SFC.
 * Each node of the Quadtree (aka each GridBlock) is associated with 
 * (i) a refinement level 
 * (ii) indices (i,j) that indicate its coordinates in a uniform grid of the same refinement level
 * (iii) a Z-order index which is a unique integer along an SFC that would traverse a uniform grid 
 * of the same refinement level
 * (iv) a unique integer (blockID_2). This class provides trasformations from each of these 
 *  attributes to the others.
 */
class SpaceFillingCurve2D
{
 protected:
   int BX; ///< number of blocks in the x-direction at the coarsest level
   int BY; ///< number of blocks in the y-direction at the coarsest level
   int levelMax; ///< maximum level allowed
   bool isRegular; ///< true if BX,BY,BZ are powers of 2
   int base_level; ///< minimum (starting) level (determined from BX,BY,BZ)
   std::vector < std::vector <long long> > Zsave; ///< option to save block indices instead of computing them every time
   std::vector< std::vector<int> > i_inverse; ///< option to save blocks i index instead of computing it every time
   std::vector< std::vector<int> > j_inverse; ///< option to save blocks j index instead of computing it every time

   /// convert (x,y) to index
   long long AxestoTranspose(const int *X_in, int b) const
   {
      int x = X_in[0];
      int y = X_in[1];
      int n = 1 << b;
      int rx, ry, s, d=0;
      for (s=n/2; s>0; s/=2)
      {
          rx = (x & s) > 0;
          ry = (y & s) > 0;
          d += s * s * ((3 * rx) ^ ry);
          rot(n, &x, &y, rx, ry);
      }
      return d;
   }

   /// convert index to (x,y)
   void TransposetoAxes(long long index, int *X, int b) const 
   {
       // position, #bits, dimension
       int n = 1 << b;
       long long rx, ry, s, t=index;
       X[0] = 0;
       X[1] = 0;
       for (s=1; s<n; s*=2) {
           rx = 1 & (t/2);
           ry = 1 & (t ^ rx);
           rot(s, &X[0], &X[1], rx, ry);
           X[0] += s * rx;
           X[1] += s * ry;
           t /= 4;
       }
   }

   /// rotate/flip a quadrant appropriately
   void rot(long long n, int *x, int *y, long long rx, long long ry) const
   {
       if (ry == 0) {
           if (rx == 1) {
               *x = n-1 - *x;
               *y = n-1 - *y;
           }
   
           //Swap x and y
           int t  = *x;
           *x = *y;
           *y = t;
       }
   }

 public:

   SpaceFillingCurve2D(){};

   SpaceFillingCurve2D(int a_BX, int a_BY, int lmax) : BX(a_BX), BY(a_BY), levelMax(lmax)
   {
      const int n_max  = std::max(BX, BY);
      base_level = (log(n_max) / log(2));
      if (base_level < (double)(log(n_max) / log(2))) base_level++;

      i_inverse.resize(lmax);
      j_inverse.resize(lmax);
      Zsave.resize(lmax);
      {
        const int l = 0;
        const int aux = pow( pow(2,l) , 2);
        i_inverse[l].resize(BX*BY*aux,-1);
        j_inverse[l].resize(BX*BY*aux,-1);
        Zsave[l].resize(BX*BY*aux,-1);
      }

      isRegular = true;
      #pragma omp parallel for collapse(2)
      for (int j=0;j<BY;j++)
      for (int i=0;i<BX;i++)
      {
        const int c[2] = {i,j};
        long long index = AxestoTranspose( c, base_level);
        long long substract = 0;
        for (long long h=0; h<index; h++)
        {
          int X[2] = {0,0};
          TransposetoAxes(h, X, base_level);
          if (X[0] >= BX || X[1] >= BY) substract++;   
        }
        index -= substract;
        if (substract > 0) isRegular = false;
        i_inverse[0][index] = i;
        j_inverse[0][index] = j;
        Zsave[0][j*BX + i] = index;
      }
    }

   /// space-filling curve (i,j) --> 1D index (given level l)
   long long forward(const int l, const int i, const int j) //const
   {
      const int aux = 1 << l;

      if (l>=levelMax) return 0;
      long long retval;
      if (!isRegular)
      {
        const int I   = i / aux;
        const int J   = j / aux;
        const int c2_a[2] = {i-I*aux,j-J*aux};
        retval = AxestoTranspose(c2_a, l);
        retval += IJ_to_index(I,J)*aux*aux;
      }
      else
      {
        const int c2_a[2] = {i,j};
        retval = AxestoTranspose(c2_a, l + base_level);
      }
      return retval;
   }

   /// space-filling curve Z-index --> (i,j) (given level l)
   void inverse(long long Z, int l, int &i, int &j)
   {
      if (isRegular)
      {
        int X[2] = {0, 0};
        TransposetoAxes(Z, X, l + base_level);
        i = X[0];
        j = X[1];
      }
      else
      {
        int aux   = 1 << l;
        long long Zloc  = Z % (aux*aux);
        int X[2] = {0, 0};
        TransposetoAxes(Zloc, X, l);
        long long index = Z / (aux*aux);
        int I,J;
        index_to_IJ(index,I,J);
        i = X[0] + I*aux;
        j = X[1] + J*aux;
      }
      return;
   }

   /// space-filling curve (i,j) --> 1D index (at level 0)
   long long IJ_to_index(int I, int J)
   {
     //int index = (J + K * BY) * BX + I;
     long long index = Zsave[0][J*BX + I];
     return index;
   }

   /// space-filling curve Z-index --> (i,j) (at level 0)
   void index_to_IJ(long long index, int & I, int & J)
   {
      I = i_inverse[0][index];
      J = j_inverse[0][index];
      return;
   }

   /// convert Z-index, level and ij index to single unique number
   long long Encode(int level, long long Z, int index[2])
   {
      int lmax   = levelMax;
      long long retval = 0;

      int ix = index[0];
      int iy = index[1];
      for (int l = level; l >= 0; l--)
      {
         long long Zp = forward(l, ix, iy);
         retval += Zp;
         ix /= 2;
         iy /= 2;
      }

      ix = 2 * index[0];
      iy = 2 * index[1];
      for (int l = level + 1; l < lmax; l++)
      {
         long long Zc = forward(l, ix, iy);

         Zc -= Zc % 4;
         retval += Zc;

         int ix1, iy1;
         inverse(Zc, l, ix1, iy1);
         ix = 2 * ix1;
         iy = 2 * iy1;
      }

      retval += level;
  
      return retval;
   }
};

} // namespace cubism

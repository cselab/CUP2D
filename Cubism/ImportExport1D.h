#pragma once

#include <math.h> 
#include <vector>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <list>
#include <numeric>
#include <random>
#include <vector>
 
#if DIMENSION == 3
  #include "SpaceFillingCurve.h"
#else
  #include "SpaceFillingCurve2D.h"
#endif

namespace cubism{
	
template <typename Grid>
void convertVectorToGrid(Grid * grid, const typename Grid::ElementType * vec)
{
	using Block = typename Grid::BlockType;
	const int nx = Block::sizeX;
	const int ny = Block::sizeY;
	const int nz = Block::sizeZ;

	const int log2n = log2(nx);
	const int sfc_level = grid->getlevelMax() + log2n;

	assert ((nx & (nx-1)) == 0);//assert that nx is a power of 2
        
	std::array<int, 3> N = {nx*grid->NX,ny*grid->NY,nz*grid->NZ};
        #if DIMENSION == 3
	  static SpaceFillingCurve   sfc = SpaceFillingCurve(N[0],N[1],N[2],sfc_level);
        #else
	  static SpaceFillingCurve2D sfc = SpaceFillingCurve2D(N[0],N[1],sfc_level);
        #endif

	//Blocks might not be ordered. We create a sorted copy and loop over it
	std::vector<BlockInfo> SortedInfos = grid->getBlocksInfo();
        std::sort(SortedInfos.begin(), SortedInfos.end());

	size_t position = 0;

	std::vector<int> indices(nx*ny*nz);
	std::vector<long long> sortID(nx*ny*nz);
	for (const auto & info : SortedInfos)
        {
	    const int level = info.level + log2n; 
            for (int z = 0 ; z < nz; z++)
	    {
	       int index[3];
	       index[2] = info.index[2]*nz + z; 
               for (int y = 0 ; y < ny; y++)
	       {
	          index[1] = info.index[1]*ny + y; 
                  for (int x = 0 ; x < nx; x++)
	          {
	             index[0] = info.index[0]*nx + x;
		     #if DIMENSION == 3
		       const long long Z = sfc.forward(level,index[0],index[1],index[2]); 
		     #else
		       const long long Z = sfc.forward(level,index[0],index[1]); 
		     #endif
		     sortID [x + y*nx + z*nx*ny] = sfc.Encode(level,Z,index);
	          }
	       }
           }

	   std::iota(indices.begin(), indices.end(), 0);
	   std::sort(indices.begin(), indices.end(),[&](int A, int B) -> bool {return sortID[A] < sortID[B];});
           Block & b = *(Block*)info.ptrBlock;
           for (int xyz = 0 ; xyz < nx*ny*nz; xyz++)
           {
		   const int iz = indices[xyz]/(nx*ny);
		   const int iy = (indices[xyz] - iz*nx*ny)/nx;
		   const int ix = indices[xyz] - iz*nx*ny - iy*nx;
	           b(ix,iy,iz) = vec[position];
		   position ++;
           }
	}

}	

template <typename Grid>
void convertGridToVector(const Grid * const grid, typename Grid::ElementType * vec)
{
	using Block = typename Grid::BlockType;
	const int nx = Block::sizeX;
	const int ny = Block::sizeY;
	const int nz = Block::sizeZ;

	const int log2n = log2(nx);
	const int sfc_level = grid->getlevelMax() + log2n;
	const size_t blocks = grid->getBlocksInfo().size();
	const size_t length = blocks * nx * ny * nz;//Total number of grid points;

	assert ((nx & (nx-1)) == 0);//assert that nx is a power of 2
        
	std::array<int, 3> N = {nx*grid->NX,ny*grid->NY,nz*grid->NZ};
        #if DIMENSION == 3
	  static SpaceFillingCurve   sfc = SpaceFillingCurve(N[0],N[1],N[2],sfc_level);
        #else
	  static 
		  SpaceFillingCurve2D sfc = SpaceFillingCurve2D(N[0],N[1],sfc_level);
        #endif

	//Blocks might not be ordered. We create a sorted copy and loop over it
	std::vector<BlockInfo> SortedInfos = grid->getBlocksInfo();
        std::sort(SortedInfos.begin(), SortedInfos.end());

	size_t position = 0;

	std::vector<int> indices(nx*ny*nz);
	std::vector<long long> sortID(nx*ny*nz);
	for (const auto & info : SortedInfos)
        {
	    const int level = info.level + log2n; 
            for (int z = 0 ; z < nz; z++)
	    {
	       int index[3];
	       index[2] = info.index[2]*nz + z; 
               for (int y = 0 ; y < ny; y++)
	       {
	          index[1] = info.index[1]*ny + y; 
                  for (int x = 0 ; x < nx; x++)
	          {
	             index[0] = info.index[0]*nx + x;
		     #if DIMENSION == 3
		       const long long Z = sfc.forward(level,index[0],index[1],index[2]); 
		     #else
		       const long long Z = sfc.forward(level,index[0],index[1]); 
		     #endif
		     sortID [x + y*nx + z*nx*ny] = sfc.Encode(level,Z,index);
	          }
	       }
           }

	   std::iota(indices.begin(), indices.end(), 0);
	   std::sort(indices.begin(), indices.end(),[&](int A, int B) -> bool {return sortID[A] < sortID[B];});
           Block & b = *(Block*)info.ptrBlock;
           for (int xyz = 0 ; xyz < nx*ny*nz; xyz++)
           {
		   const int iz = indices[xyz]/(nx*ny);
		   const int iy = (indices[xyz] - iz*nx*ny)/nx;
		   const int ix = indices[xyz] - iz*nx*ny - iy*nx;
		   vec[position] = b(ix,iy,iz);
		   position ++;
           }
	}
}

}

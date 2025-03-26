#pragma once
#include "ConsistentOperations.h"
#include "StencilInfo.h"
#include "BlockLab.h"
#include "AMR_SynchronizerMPI.h"

namespace cubism {

template <typename Lab, typename Kernel, typename TGrid, typename TGrid_corr = TGrid>
void compute(Kernel &&kernel, TGrid *g, TGrid_corr *g_corr = nullptr)
{
  //If flux corrections are needed, prepare the flux correction object
  if (g_corr != nullptr) g_corr->Corrector.prepare(*g_corr);

  //Start sending and receiving of data for blocks at the boundary of each rank
  cubism::SynchronizerMPI_AMR<typename TGrid::Real,TGrid>& Synch = *(g->sync(kernel.stencil));

  //Access the inner blocks of each rank
  std::vector<cubism::BlockInfo*> * inner = & Synch.avail_inner();

  std::vector<cubism::BlockInfo*> * halo_next;
  bool done = false;
  #pragma omp parallel
  {
    Lab lab;
    lab.prepare(*g, kernel.stencil);

    //First compute for inner blocks
    #pragma omp for nowait
    for (const auto & I : * inner)
    {
      lab.load(*I, 0);
      kernel(lab, *I);
    }

    //Then compute for boundary blocks
#if 1
    while(done == false)
    {
      #pragma omp master
      halo_next = &Synch.avail_next();
      #pragma omp barrier

      #pragma omp for nowait
      for (const auto & I : * halo_next)
      {
        lab.load(*I, 0);
        kernel(lab, *I);
      }

      #pragma omp single
      {
        if (halo_next->size() == 0) done = true;
      }
    }
#else  
  std::vector<cubism::BlockInfo> & blk = g->getBlocksInfo();
  std::vector<bool> ready(blk.size(),false);
  std::vector<cubism::BlockInfo*> & avail1  = Synch .avail_halo_nowait();
  const int Nhalo = avail1.size();
  while(done == false)
  {
    done = true;
    for(int i=0; i<Nhalo; i++)
    {
      const cubism::BlockInfo &I  = *avail1 [i];
      if (ready[I.blockID] == false)
      {
        if (Synch.isready(I ))
        {
         ready[I.blockID] = true;
         lab.load(I, 0);
         kernel(lab, I);
        }
        else
        {
          done = false;
        }
      }
    }
  }
#endif
  }

  //Complete the send requests remaining
  Synch.avail_halo();

  //Carry out flux corrections
  if (g_corr != nullptr) g_corr->Corrector.FillBlockCases();
}

//Get two BlockLabs from two different Grids
template <typename Kernel, typename TGrid, typename LabMPI, typename TGrid2, typename LabMPI2, typename TGrid_corr = TGrid>
static void compute(const Kernel& kernel, TGrid& grid, TGrid2& grid2, const bool applyFluxCorrection = false, TGrid_corr * corrected_grid = nullptr)
{
  if (applyFluxCorrection) corrected_grid->Corrector.prepare(*corrected_grid);

  SynchronizerMPI_AMR<typename TGrid::Real,TGrid >& Synch  = * grid .sync(kernel.stencil);
  Kernel kernel2 = kernel;
  kernel2.stencil.sx = kernel2.stencil2.sx;
  kernel2.stencil.sy = kernel2.stencil2.sy;
  kernel2.stencil.sz = kernel2.stencil2.sz;
  kernel2.stencil.ex = kernel2.stencil2.ex;
  kernel2.stencil.ey = kernel2.stencil2.ey;
  kernel2.stencil.ez = kernel2.stencil2.ez;
  kernel2.stencil.tensorial = kernel2.stencil2.tensorial;
  kernel2.stencil.selcomponents.clear();
  kernel2.stencil.selcomponents = kernel2.stencil2.selcomponents;

  SynchronizerMPI_AMR<typename TGrid::Real,TGrid2>& Synch2 = * grid2.sync(kernel2.stencil);

  const StencilInfo & stencil = Synch.getstencil();
  const StencilInfo & stencil2 = Synch2.getstencil();

  std::vector<cubism::BlockInfo> & blk = grid.getBlocksInfo();
  std::vector<bool> ready(blk.size(),false);

  std::vector<BlockInfo*> & avail0  = Synch .avail_inner();
  std::vector<BlockInfo*> & avail02 = Synch2.avail_inner();
  const int Ninner = avail0.size();
  std::vector<cubism::BlockInfo*> avail1 ;
  std::vector<cubism::BlockInfo*> avail12;
  //bool done = false;
  #pragma omp parallel
  {
    LabMPI  lab;
    LabMPI2 lab2;
    lab.prepare(grid, stencil);
    lab2.prepare(grid2, stencil2);
    #pragma omp for
    for(int i=0; i<Ninner; i++)
    {
      const BlockInfo &I  = *avail0 [i];
      const BlockInfo &I2 = *avail02[i];
      lab.load(I, 0);
      lab2.load(I2, 0);
      kernel(lab,lab2,I,I2);
      ready[I.blockID] = true;
    }

    #if 1
    #pragma omp master
    {
        avail1  = Synch .avail_halo();
        avail12 = Synch2.avail_halo();
    }
    #pragma omp barrier

    const int Nhalo = avail1.size();

    #pragma omp for
    for(int i=0; i<Nhalo; i++)
    {
      const cubism::BlockInfo &I  = *avail1 [i];
      const cubism::BlockInfo &I2 = *avail12[i];
      lab.load(I, 0);
      lab2.load(I2, 0);
      kernel(lab, lab2, I, I2);
    }
    #else

    #pragma omp master
    {
        avail1  = Synch .avail_halo_nowait();
        avail12 = Synch2.avail_halo_nowait();
    }
    #pragma omp barrier
    const int Nhalo = avail1.size();

    while(done == false)
    {
      #pragma omp barrier

      #pragma omp single
      done = true;

      #pragma omp for
      for(int i=0; i<Nhalo; i++)
      {
        const cubism::BlockInfo &I  = *avail1 [i];
        const cubism::BlockInfo &I2 = *avail12[i];
        if (ready[I.blockID] == false)
        {
          bool blockready;
          #pragma omp critical
          {
            blockready = (Synch.isready(I ) && Synch.isready(I2));
          }
          if (blockready)
          {
           ready[I.blockID] = true;
           lab.load(I, 0);
           lab2.load(I2, 0);
           kernel(lab, lab2, I, I2);
          }
          else
          {
            #pragma omp atomic write
            done = false;
          }
        }
      }
    }
    avail1  = Synch .avail_halo();
    avail12 = Synch2.avail_halo();
  #endif
  }

  if (applyFluxCorrection) corrected_grid->Corrector.FillBlockCases();
}

///Example of a gridpoint element that is merely a scalar quantity of type 'Real' (double/float).
template <typename Real=double>
struct ScalarElement
{
  using RealType = Real; ///< definition of 'RealType', needed by BlockLab
  Real s = 0; ///< scalar quantity

  ///set scalar to zero
  inline void clear() { s = 0; }

  ///set scalar to a value
  inline void set(const Real v) { s = v; }

  ///copy a ScalarElement
  inline void copy(const ScalarElement& c) { s = c.s; }

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
  Real magnitude()
  {
    return s;
  }
  Real & member(int i)
  {
    return s;
  }
  static constexpr int DIM = 1;
};

///Example of a gridpoint element that is a vector quantity of type 'Real' (double/float); 'dim' are the number of dimensions of the vector
template <int dim, typename Real=double>
struct VectorElement
{
  using RealType = Real;  ///< definition of 'RealType', needed by BlockLab
  static constexpr int DIM = dim;
  Real u[DIM]; ///< vector quantity

  VectorElement() { clear(); }

  ///set vector components to zero
  inline void clear() { for(int i=0; i<DIM; ++i) u[i] = 0; }

  ///set vector components to a number
  inline void set(const Real v) { for(int i=0; i<DIM; ++i) u[i] = v; }

  ///set copy one VectorElement to another
  inline void copy(const VectorElement& c) {
    for(int i=0; i<DIM; ++i) u[i] = c.u[i];
  }

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
    Real s1 = 0.0;
    Real s2 = 0.0;
    for(int i=0; i<DIM; ++i)
    {
      s1 +=u[i]*u[i];
      s2 +=other.u[i]*other.u[i];
    }

    return (s1 < s2);
  }
  bool operator>(const VectorElement &other) const
  {
    Real s1 = 0.0;
    Real s2 = 0.0;
    for(int i=0; i<DIM; ++i)
    {
      s1 +=u[i]*u[i];
      s2 +=other.u[i]*other.u[i];
    }

    return (s1 > s2);
  }
  bool operator<=(const VectorElement &other) const
  {
    Real s1 = 0.0;
    Real s2 = 0.0;
    for(int i=0; i<DIM; ++i)
    {
      s1 +=u[i]*u[i];
      s2 +=other.u[i]*other.u[i];
    }

    return (s1 <= s2);
  }
  bool operator>=(const VectorElement &other) const
  {
    Real s1 = 0.0;
    Real s2 = 0.0;
    for(int i=0; i<DIM; ++i)
    {
      s1 +=u[i]*u[i];
      s2 +=other.u[i]*other.u[i];
    }

    return (s1 >= s2);
  }
  Real magnitude()
  {
    Real s1 = 0.0;
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

/// array of blocksize^dim gridpoints of type 'TElement'.
template <int blocksize, int dim, typename TElement>
struct GridBlock
{
  //these identifiers are required by cubism!
  static constexpr int BS = blocksize;
  static constexpr int sizeX = blocksize;
  static constexpr int sizeY = blocksize;
  static constexpr int sizeZ = dim > 2 ? blocksize : 1;
  static constexpr std::array<int, 3> sizeArray = {sizeX, sizeY, sizeZ};
  using ElementType = TElement;
  using RealType = typename TElement::RealType;

  ElementType data[sizeZ][sizeY][sizeX];

  ///set 'data' to zero (call 'clear()' of each TElement)
  inline void clear() {
      ElementType * const entry = &data[0][0][0];
      for(int i=0; i<sizeX*sizeY*sizeZ; ++i) entry[i].clear();
  }

  ///set 'data' to a value (call 'set()' of each TElement)
  inline void set(const RealType v) {
      ElementType * const entry = &data[0][0][0];
      for(int i=0; i<sizeX*sizeY*sizeZ; ++i) entry[i].set(v);
  }

  ///copy one GridBlock to another  (call 'copy()' of each TElement)
  inline void copy(const GridBlock<blocksize,dim,ElementType>& c) {
      ElementType * const entry = &data[0][0][0];
      const ElementType * const source = &c.data[0][0][0];
      for(int i=0; i<sizeX*sizeY*sizeZ; ++i) entry[i].copy(source[i]);
  }

  ///Access an element of this GridBlock (const.)
  const ElementType& operator()(int ix, int iy=0, int iz=0) const {
      assert(ix>=0 && iy>=0 && iz>=0 && ix<sizeX && iy<sizeY && iz<sizeZ);
      return data[iz][iy][ix];
  }

  ///Access an element of this GridBlock
  ElementType& operator()(int ix, int iy=0, int iz=0) {
      assert(ix>=0 && iy>=0 && iz>=0 && ix<sizeX && iy<sizeY && iz<sizeZ);
      return data[iz][iy][ix];
  }
  GridBlock(const GridBlock&) = delete;
  GridBlock& operator=(const GridBlock&) = delete;
};

/** BlockLab to apply zero Neumann boundary conditions (zero normal derivative to the boundary).
 * @tparam TGrid: Grid/GridMPI type to apply the boundary conditions to.
 * @tparam dim: = 2 or 3, depending on the spatial dimensions
 * @tparam allocator: allocator object, same as the one from BlockLab. 
 */
template<typename TGrid, int dim, template<typename X> class allocator=std::allocator>
class BlockLabNeumann: public cubism::BlockLab<TGrid,allocator>
{
  /*
   * Apply 2nd order Neumann boundary condition: du/dn_{i+1/2} = 0 => u_{i} = u_{i+1}
   */
  static constexpr int sizeX = TGrid::BlockType::sizeX;
  static constexpr int sizeY = TGrid::BlockType::sizeY;
  static constexpr int sizeZ = TGrid::BlockType::sizeZ;
  static constexpr int DIM = dim;
 protected:
  /// Apply bc on face of direction dir and side side (0 or 1):
  template<int dir, int side> void Neumann3D(const bool coarse = false)
  {
    int stenBeg[3];
    int stenEnd[3];
    int bsize[3];
    if (!coarse)
    {
      stenEnd[0] = this->m_stencilEnd[0];
      stenEnd[1] = this->m_stencilEnd[1];
      stenEnd[2] = this->m_stencilEnd[2];
      stenBeg[0] = this->m_stencilStart[0];
      stenBeg[1] = this->m_stencilStart[1];
      stenBeg[2] = this->m_stencilStart[2];
      bsize[0] = sizeX;
      bsize[1] = sizeY;
      bsize[2] = sizeZ;
    }
    else
    {
      stenEnd[0] = (this->m_stencilEnd[0])/2 + 1 + this->m_InterpStencilEnd[0] -1;
      stenEnd[1] = (this->m_stencilEnd[1])/2 + 1 + this->m_InterpStencilEnd[1] -1;
      stenEnd[2] = (this->m_stencilEnd[2])/2 + 1 + this->m_InterpStencilEnd[2] -1;
      stenBeg[0] = (this->m_stencilStart[0]-1)/2+  this->m_InterpStencilStart[0];
      stenBeg[1] = (this->m_stencilStart[1]-1)/2+  this->m_InterpStencilStart[1];
      stenBeg[2] = (this->m_stencilStart[2]-1)/2+  this->m_InterpStencilStart[2];
      bsize[0] = sizeX/2;
      bsize[1] = sizeY/2;
      bsize[2] = sizeZ/2;
    }

    auto * const cb = coarse ? this->m_CoarsenedBlock : this->m_cacheBlock;

    int s[3];
    int e[3];
    s[0] =  dir==0 ? (side==0 ? stenBeg[0] : bsize[0]                ) : 0;
    s[1] =  dir==1 ? (side==0 ? stenBeg[1] : bsize[1]                ) : 0;
    s[2] =  dir==2 ? (side==0 ? stenBeg[2] : bsize[2]                ) : 0;
    e[0] =  dir==0 ? (side==0 ? 0          : bsize[0] + stenEnd[0]-1 ) : bsize[0];
    e[1] =  dir==1 ? (side==0 ? 0          : bsize[1] + stenEnd[1]-1 ) : bsize[1];
    e[2] =  dir==2 ? (side==0 ? 0          : bsize[2] + stenEnd[2]-1 ) : bsize[2];

    //Fill face
    for(int iz=s[2]; iz<e[2]; iz++)
    for(int iy=s[1]; iy<e[1]; iy++)
    for(int ix=s[0]; ix<e[0]; ix++)
    {
      cb->Access(ix-stenBeg[0], iy-stenBeg[1], iz-stenBeg[2]) = cb->Access
          ( ( dir==0 ? (side==0 ? 0 : bsize[0]-1 ) : ix ) - stenBeg[0],
            ( dir==1 ? (side==0 ? 0 : bsize[1]-1 ) : iy ) - stenBeg[1],
            ( dir==2 ? (side==0 ? 0 : bsize[2]-1 ) : iz ) - stenBeg[2]);
    }

    //Fill edges and corners (necessary for the coarse block)
    s[dir] = stenBeg[dir]*(1-side) + bsize[dir]*side;
    e[dir] = (bsize[dir]-1+stenEnd[dir])*side;
    const int d1 = (dir + 1) % 3;
    const int d2 = (dir + 2) % 3;
    for(int b=0; b<2; ++b)
    for(int a=0; a<2; ++a)
    {
      s[d1] = stenBeg[d1] + a*b*(bsize[d1] - stenBeg[d1]);
      s[d2] = stenBeg[d2] + (a-a*b)*(bsize[d2] - stenBeg[d2]);
      e[d1] = (1-b+a*b)*(bsize[d1] - 1 + stenEnd[d1]);
      e[d2] = (a+b-a*b)*(bsize[d2] - 1 + stenEnd[d2]);
      for(int iz=s[2]; iz<e[2]; iz++)
      for(int iy=s[1]; iy<e[1]; iy++)
      for(int ix=s[0]; ix<e[0]; ix++)
      {
        cb->Access(ix-stenBeg[0], iy-stenBeg[1], iz-stenBeg[2]) = 
         dir==0? cb->Access(side*(bsize[0]-1)-stenBeg[0], iy               -stenBeg[1], iz               -stenBeg[2]) : 
        (dir==1? cb->Access(ix               -stenBeg[0], side*(bsize[1]-1)-stenBeg[1], iz               -stenBeg[2]) :
                 cb->Access(ix               -stenBeg[0], iy               -stenBeg[1], side*(bsize[2]-1)-stenBeg[2]));
      }
    }
  }

  /// Apply bc on face of direction dir and side side (0 or 1):
  template<int dir, int side> void Neumann2D(const bool coarse = false)
  {
    int stenBeg[2];
    int stenEnd[2];
    int bsize[2];
    if (!coarse)
    {
      stenEnd[0] = this->m_stencilEnd[0];
      stenEnd[1] = this->m_stencilEnd[1];
      stenBeg[0] = this->m_stencilStart[0];
      stenBeg[1] = this->m_stencilStart[1];
      bsize[0] = sizeX;
      bsize[1] = sizeY;
    }
    else
    {
      stenEnd[0] = (this->m_stencilEnd[0])/2 + 1 + this->m_InterpStencilEnd[0] -1;
      stenEnd[1] = (this->m_stencilEnd[1])/2 + 1 + this->m_InterpStencilEnd[1] -1;
      stenBeg[0] = (this->m_stencilStart[0]-1)/2+  this->m_InterpStencilStart[0];
      stenBeg[1] = (this->m_stencilStart[1]-1)/2+  this->m_InterpStencilStart[1];
      bsize[0] = sizeX/2;
      bsize[1] = sizeY/2;
    }

    auto * const cb = coarse ? this->m_CoarsenedBlock : this->m_cacheBlock;

    int s[2];
    int e[2];
    s[0] =  dir==0 ? (side==0 ? stenBeg[0] : bsize[0]                ): stenBeg[0];
    s[1] =  dir==1 ? (side==0 ? stenBeg[1] : bsize[1]                ): stenBeg[1];
    e[0] =  dir==0 ? (side==0 ? 0          : bsize[0] + stenEnd[0]-1 ): bsize[0] +  stenEnd[0]-1;
    e[1] =  dir==1 ? (side==0 ? 0          : bsize[1] + stenEnd[1]-1 ): bsize[1] +  stenEnd[1]-1;

    for(int iy=s[1]; iy<e[1]; iy++)
    for(int ix=s[0]; ix<e[0]; ix++)
      cb->Access(ix-stenBeg[0], iy-stenBeg[1], 0) =
          cb->Access(( dir==0? (side==0? 0: bsize[0]-1):ix ) - stenBeg[0],
                     ( dir==1? (side==0? 0: bsize[1]-1):iy ) - stenBeg[1], 0 );
  }

 public:
  typedef typename TGrid::BlockType::ElementType ElementTypeBlock;
  typedef typename TGrid::BlockType::ElementType ElementType;
   using Real = typename ElementType::RealType; ///< Number type used by Element (double/float etc.)
  ///Will return 'false' as the boundary conditions are not periodic for this BlockLab.
  virtual bool is_xperiodic() override { return false; }
  ///Will return 'false' as the boundary conditions are not periodic for this BlockLab.
  virtual bool is_yperiodic() override { return false; }
  ///Will return 'false' as the boundary conditions are not periodic for this BlockLab.
  virtual bool is_zperiodic() override { return false; }

  BlockLabNeumann() = default;
  BlockLabNeumann(const BlockLabNeumann&) = delete;
  BlockLabNeumann& operator=(const BlockLabNeumann&) = delete;

  /// Apply the boundary condition; 'coarse' is set to true if the boundary condition should be applied to the coarsened version of the BlockLab (see also BlockLab).
  void _apply_bc(const cubism::BlockInfo& info, const Real t=0, const bool coarse = false) override
  {
    if (DIM == 2)
    {
      if(info.index[0]==0 )          this->template Neumann2D<0,0>(coarse);
      if(info.index[0]==this->NX-1 ) this->template Neumann2D<0,1>(coarse);
      if(info.index[1]==0 )          this->template Neumann2D<1,0>(coarse);
      if(info.index[1]==this->NY-1 ) this->template Neumann2D<1,1>(coarse);
    }
    else if (DIM == 3)
    {
      if(info.index[0]==0 )          this->template Neumann3D<0,0>(coarse);
      if(info.index[0]==this->NX-1 ) this->template Neumann3D<0,1>(coarse);
      if(info.index[1]==0 )          this->template Neumann3D<1,0>(coarse);
      if(info.index[1]==this->NY-1 ) this->template Neumann3D<1,1>(coarse);
      if(info.index[2]==0 )          this->template Neumann3D<2,0>(coarse);
      if(info.index[2]==this->NZ-1 ) this->template Neumann3D<2,1>(coarse);
    }
  }
};

}//namespace cubism

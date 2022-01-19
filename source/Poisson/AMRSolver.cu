//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cusparse.h"

#include "helper_cuda.h"
#include "AMRSolver.cuh"

using namespace cubism;

#ifdef BICGSTAB_PROFILER
deviceProfiler::deviceProfiler() : elapsed_(0.)
{
  checkCudaErrors(cudaEventCreate(&start_));
  checkCudaErrors(cudaEventCreate(&stop_));
}

deviceProfiler::~deviceProfiler() 
{
  checkCudaErrors(cudaEventDestroy(start_));
  checkCudaErrors(cudaEventDestroy(stop_));
}

void deviceProfiler::startProfiler(cudaStream_t stream)
{
  checkCudaErrors(cudaEventRecord(start_, stream));
}

void deviceProfiler::stopProfiler(cudaStream_t stream)
{
  checkCudaErrors(cudaEventRecord(stop_, stream));
  checkCudaErrors(cudaEventSynchronize(stop_));

  float event_time = 0.;
  checkCudaErrors(cudaEventElapsedTime(&event_time, start_, stop_));
  elapsed_ += event_time;
}
#endif

// -------------------------------- Host-side construction of linear system -----------------------------------

class CellIndexer{
  public:
    static int This(const BlockInfo &info, const int &ix, const int &iy)
    { return info.blockID*BSX*BSY + iy*BSX + ix; }
    static int WestNeighbour(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return info.blockID*BSX*BSY + iy*BSX + ix-dist; }
    static int NorthNeighbour(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return info.blockID*BSX*BSY + (iy+dist)*BSX + ix;}
    static int EastNeighbour(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return info.blockID*BSX*BSY + iy*BSX + ix+dist; }
    static int SouthNeighbour(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return info.blockID*BSX*BSY + (iy-dist)*BSX + ix; }

    static int WestmostCell(const BlockInfo &info, const int &ix, const int &iy, const int offset = 0)
    { return info.blockID*BSX*BSY + iy*BSX + offset; }
    static int NorthmostCell(const BlockInfo &info, const int &ix, const int &iy, const int offset = 0)
    { return info.blockID*BSX*BSY + (BSY-1-offset)*BSX + ix; }
    static int EastmostCell(const BlockInfo &info, const int &ix, const int &iy, const int offset = 0)
    { return info.blockID*BSX*BSY + iy*BSX + (BSX-1-offset); }
    static int SouthmostCell(const BlockInfo &info, const int &ix, const int &iy, const int offset = 0)
    { return info.blockID*BSX*BSY + offset*BSX + ix; }

    static constexpr int BSX = VectorBlock::sizeX;
    static constexpr int BSY = VectorBlock::sizeY;
};

class NorthEdgeCell : public CellIndexer{
  public:
    static int inblock_n1(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return EastNeighbour(info, ix, iy, dist); }
    static int inblock_n2(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return SouthNeighbour(info, ix, iy, dist); }
    static int inblock_n3(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return WestNeighbour(info, ix, iy, dist); }
    static int neiblock_n(const BlockInfo &nei_info, const int &ix, const int &iy, const int offset = 0)
    { return SouthmostCell(nei_info, ix, iy, offset); }

    static int forward(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return EastNeighbour(info, ix, iy, dist); }
    static int backward(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return WestNeighbour(info, ix, iy, dist); }

    static bool back_corner(const int &ix, const int &iy)
    { return ix == 0; }
    static bool front_corner(const int &ix, const int &iy)
    { return ix == BSX - 1; }
    static bool mod(const int &ix, const int &iy)
    { return ix % 2 == 0; }

    static int ix_c(const BlockInfo &info, const int &ix, const int &iy)
    { return (info.index[0] % 2 == 1) ? (ix/2 + BSX/2) : (ix/2); }
    static int iy_c(const BlockInfo &info, const int &ix, const int &iy)
    { return -1; }

    static long long Zchild(const BlockInfo &nei_info, const int &ix, const int &iy)
    {return ix < BSX/2 ? nei_info.Zchild[0][0][0] : nei_info.Zchild[1][0][0];}
};

class EastEdgeCell : public CellIndexer{
  public:
    static int inblock_n1(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return SouthNeighbour(info, ix, iy, dist); }
    static int inblock_n2(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return WestNeighbour(info, ix, iy, dist); }
    static int inblock_n3(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return NorthNeighbour(info, ix, iy, dist); }
    static int neiblock_n(const BlockInfo &nei_info, const int &ix, const int &iy, const int offset = 0)
    { return WestmostCell(nei_info, ix, iy, offset); }

    static int forward(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return NorthNeighbour(info, ix, iy, dist); }
    static int backward(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return SouthNeighbour(info, ix, iy, dist); }

    static bool back_corner(const int &ix, const int &iy)
    { return iy == 0; }
    static bool front_corner(const int &ix, const int &iy)
    { return iy == BSY - 1; }
    static bool mod(const int &ix, const int &iy)
    { return iy % 2 == 0; }

    static int ix_c(const BlockInfo &info, const int &ix, const int &iy)
    { return -1; }
    static int iy_c(const BlockInfo &info, const int &ix, const int &iy)
    { return (info.index[1] % 2 == 1) ? (iy/2 + BSY/2) : (iy/2); }

    static long long Zchild(const BlockInfo &nei_info, const int &ix, const int &iy)
    {return iy < BSY/2 ? nei_info.Zchild[0][0][0] : nei_info.Zchild[0][1][0];}
};

class SouthEdgeCell : public CellIndexer{
  public:
    static int inblock_n1(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return WestNeighbour(info, ix, iy, dist); }
    static int inblock_n2(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return NorthNeighbour(info, ix, iy, dist); }
    static int inblock_n3(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return EastNeighbour(info, ix, iy, dist); }
    static int neiblock_n(const BlockInfo &nei_info, const int &ix, const int &iy, const int offset = 0)
    { return NorthmostCell(nei_info, ix, iy, offset); }

    static int forward(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return EastNeighbour(info, ix, iy, dist); }
    static int backward(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return WestNeighbour(info, ix, iy, dist); }

    static bool back_corner(const int &ix, const int &iy)
    { return ix == 0; }
    static bool front_corner(const int &ix, const int &iy)
    { return ix == BSX - 1; }
    static bool mod(const int &ix, const int &iy)
    { return ix % 2 == 0; }

    static int ix_c(const BlockInfo &info, const int &ix, const int &iy)
    { return (info.index[0] % 2 == 1) ? (ix/2 + BSX/2) : (ix/2); }
    static int iy_c(const BlockInfo &info, const int &ix, const int &iy)
    { return -1; }

    static long long Zchild(const BlockInfo &nei_info, const int &ix, const int &iy)
    {return ix < BSX/2 ? nei_info.Zchild[0][1][0] : nei_info.Zchild[1][1][0];}
};

class WestEdgeCell : public CellIndexer{
  public:
    static int inblock_n1(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return NorthNeighbour(info, ix, iy, dist); }
    static int inblock_n2(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return EastNeighbour(info, ix, iy, dist); }
    static int inblock_n3(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return SouthNeighbour(info, ix, iy, dist); }
    static int neiblock_n(const BlockInfo &nei_info, const int &ix, const int &iy, const int offset = 0)
    { return EastmostCell(nei_info, ix, iy, offset); }

    static int forward(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return NorthNeighbour(info, ix, iy, dist); }
    static int backward(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return SouthNeighbour(info, ix, iy, dist); }

    static bool back_corner(const int &ix, const int &iy)
    { return iy == 0; }
    static bool front_corner(const int &ix, const int &iy)
    { return iy == BSY - 1; }
    static bool mod(const int &ix, const int &iy)
    { return iy % 2 == 0; }

    static int ix_c(const BlockInfo &info, const int &ix, const int &iy)
    { return -1; }
    static int iy_c(const BlockInfo &info, const int &ix, const int &iy)
    { return (info.index[1] % 2 == 1) ? (iy/2 + BSY/2) : (iy/2); }

    static long long Zchild(const BlockInfo &nei_info, const int &ix, const int &iy)
    {return iy < BSY/2 ? nei_info.Zchild[1][0][0] : nei_info.Zchild[1][1][0];}
};

enum FDType {FDLower, FDUpper, BDLower, BDUpper, CDLower, CDUpper};

class PolyO3I {
  public:
    template<class EdgeCellIndexer>
    PolyO3I(const EdgeCellIndexer &indexer, bool neiCoarser, 
        const BlockInfo &info_c, const int &ix_c, const int &iy_c,
        const BlockInfo &info_f, const int &ix_f, const int &iy_f)
      : neiCoarser_(neiCoarser), sign_(neiCoarser ? 1. : -1.)
    {
      if (neiCoarser_)
      {
        coarse_centre_idx_ = indexer.neiblock_n(info_c, ix_c, iy_c);
        if (indexer.back_corner(ix_c, iy_c))
        { // Forward Differences
          coarse_offset1_idx_ = indexer.neiblock_n(info_c, ix_c+1, iy_c+1); 
          coarse_offset2_idx_ = indexer.neiblock_n(info_c, ix_c+2, iy_c+2); 
          type_ = indexer.mod(ix_f, iy_f) ? FDLower : FDUpper;
        }
        else if (indexer.front_corner(ix_c, iy_c))
        { // BD
          coarse_offset1_idx_ = indexer.neiblock_n(info_c, ix_c-1, iy_c-1); 
          coarse_offset2_idx_ = indexer.neiblock_n(info_c, ix_c-2, iy_c-2); 
          type_ = indexer.mod(ix_f, iy_f) ? BDLower : BDUpper;
        }
        else
        { // CD
          coarse_offset1_idx_ = indexer.neiblock_n(info_c, ix_c-1, iy_c-1); 
          coarse_offset2_idx_ = indexer.neiblock_n(info_c, ix_c+1, iy_c+1); 
          type_ = indexer.mod(ix_f, iy_f) ? CDLower : CDUpper;
        }
        fine_close_idx_ = CellIndexer::This(info_f, ix_f, iy_f);
        fine_far_idx_ = indexer.inblock_n2(info_f, ix_f, iy_f);
      }
      else // neiFiner
      {
        coarse_centre_idx_ = CellIndexer::This(info_c, ix_c, iy_c);

        if (indexer.back_corner(ix_c, iy_c))
        { // FD
          coarse_offset1_idx_ = indexer.forward(info_c, ix_c, iy_c, 1);
          coarse_offset2_idx_ = indexer.forward(info_c, ix_c, iy_c, 2);
          type_ = indexer.mod(ix_f, iy_f) ? FDLower : FDUpper;
        }
        else if (indexer.front_corner(ix_c, iy_c))
        { // BD
          coarse_offset1_idx_ = indexer.backward(info_c, ix_c, iy_c, 1);
          coarse_offset2_idx_ = indexer.backward(info_c, ix_c, iy_c, 2);
          type_ = indexer.mod(ix_f, iy_f) ? BDLower : BDUpper;
        }
        else
        { // CD
          coarse_offset1_idx_ = indexer.backward(info_c, ix_c, iy_c);
          coarse_offset2_idx_ = indexer.forward(info_c, ix_c, iy_c);
          type_ = indexer.mod(ix_f, iy_f) ? CDLower : CDUpper;
        }

        fine_close_idx_ = indexer.neiblock_n(info_f, ix_f, iy_f, 0);
        fine_far_idx_ = indexer.neiblock_n(info_f, ix_f, iy_f, 1);
      }
    }

  private:
    // Central Difference "Upper" Taylor approximation (positive 1st order term)
    void CDUpperTaylor(std::map<int,double> &row_map)
    {
      // 8./15. comes from coeff of polynomial at 2nd step of interpolation
      static constexpr double p_centre = (8./15.) * (1. - 1./16.);
      static constexpr double p_bottom = (8./15.) * (-1./8. + 1./32.);
      static constexpr double p_top = (8./15.) * ( 1./8. + 1./32.);

      row_map[coarse_centre_idx_] += sign_*p_centre;
      row_map[coarse_offset1_idx_] += sign_*p_bottom;
      row_map[coarse_offset2_idx_] += sign_*p_top;
    }

    // Central Difference "Lower" Taylor approximation (negative 1st order term)
    void CDLowerTaylor(std::map<int,double> &row_map)
    {
      static constexpr double p_centre = (8./15.) * (1. - 1./16.);
      static constexpr double p_bottom = (8./15.) * ( 1./8. + 1./32.);
      static constexpr double p_top = (8./15.) * (-1./8. + 1./32.);

      row_map[coarse_centre_idx_] += sign_*p_centre;
      row_map[coarse_offset1_idx_] += sign_*p_bottom;
      row_map[coarse_offset2_idx_] += sign_*p_top;
    }

    void BiasedUpperTaylor(std::map<int,double> &row_map)
    {
      static constexpr double p_centre = (8./15.) * (1 + 3./8. + 1./32.);
      static constexpr double p_offset1 = (8./15.) * (-1./2. - 1./16.);
      static constexpr double p_offset2 = (8./15.) * ( 1./8. + 1./32.);

      row_map[coarse_centre_idx_] += sign_*p_centre;
      row_map[coarse_offset1_idx_] += sign_*p_offset1;
      row_map[coarse_offset2_idx_] += sign_*p_offset2;
    }

    void BiasedLowerTaylor(std::map<int,double> &row_map)
    {
      static constexpr double p_centre = (8./15.) * (1 - 3./8. + 1./32.);
      static constexpr double p_offset1 = (8./15.) * ( 1./2. - 1./16.);
      static constexpr double p_offset2 = (8./15.) * (-1./8. + 1./32.);

      row_map[coarse_centre_idx_] += sign_*p_centre;
      row_map[coarse_offset1_idx_] += sign_*p_offset1;
      row_map[coarse_offset2_idx_] += sign_*p_offset2;
    }

    // Aliases for offset based biased functionals to forward/backward differences in corners
    void BDUpperTaylor(std::map<int,double> &row_map) { return BiasedUpperTaylor(row_map); }
    void BDLowerTaylor(std::map<int,double> &row_map) { return BiasedLowerTaylor(row_map); }
    void FDUpperTaylor(std::map<int,double> &row_map) { return BiasedLowerTaylor(row_map); }
    void FDLowerTaylor(std::map<int,double> &row_map) { return BiasedUpperTaylor(row_map); }

  public:
    // F_i = sign * (p_{interpolation} - p_{fine_cell_idx})
    void interpolate(std::map<int,double> &row_map)
    {
      static constexpr double p_fine_close = 2./3.;
      static constexpr double p_fine_far = -1./5.;

      // Interpolated flux sign * p_{interpolation}
      switch(type_)
      {
        case FDLower:
          FDLowerTaylor(row_map);
          break;
        case FDUpper:
          FDUpperTaylor(row_map);
          break;
        case BDLower:
          BDLowerTaylor(row_map);
          break;
        case BDUpper:
          BDUpperTaylor(row_map);
          break;
        case CDLower:
          CDLowerTaylor(row_map);
          break;
        case CDUpper:
          CDUpperTaylor(row_map);
          break;
      }

      row_map[fine_close_idx_] += sign_ * p_fine_close;
      row_map[fine_far_idx_] += sign_ * p_fine_far;

      // Non-interpolated flux contribution -sign * p_{fine_cell_idx}
      row_map[fine_close_idx_] -= sign_;
    } 

  private:
    const bool neiCoarser_;
    const double sign_;
    int coarse_centre_idx_;
    int coarse_offset1_idx_; // bottom/left
    int coarse_offset2_idx_; // top/right
    int fine_close_idx_;
    int fine_far_idx_;
    FDType type_;
};

void AMRSolver::cooMatPushBackVal(
    const double &val, 
    const int &row, 
    const int &col){
  this->cooValA_.push_back(val);
  this->cooRowA_.push_back(row);
  this->cooColA_.push_back(col);
}

void AMRSolver::cooMatPushBackRow(
    const int &row_idx,
    const std::map<int,double> &row_map)
{
  // Sorted by key by default!
  for (const auto &[col_idx, val] : row_map)
    this->cooMatPushBackVal(val, row_idx, col_idx);
}

template<class EdgeIndexer >
void AMRSolver::makeFlux(
  const BlockInfo &rhs_info,
  const int &ix,
  const int &iy,
  std::map<int,double> &row_map,
  const BlockInfo &rhsNei,
  const EdgeIndexer &indexer) const 
{
  const int sfc_idx = CellIndexer::This(rhs_info, ix, iy);

  if (this->sim.tmp->Tree(rhsNei).Exists())
  { 
    const int n_idx = indexer.neiblock_n(rhsNei, ix, iy);

    // Map flux associated to out-of-block edges at the same level of refinement
    row_map[n_idx] += 1.; row_map[sfc_idx] += -1.;
  }
  else if (this->sim.tmp->Tree(rhsNei).CheckCoarser())
  {
    const BlockInfo &rhsNei_c = this->sim.tmp->getBlockInfoAll(rhs_info.level - 1 , rhsNei.Zparent);

    const int ix_c = indexer.ix_c(rhs_info, ix, iy);
    const int iy_c = indexer.iy_c(rhs_info, ix, iy);

    // Perform intepolation to calculate flux at interface with coarse cell
    PolyO3I pts(indexer, true, rhsNei_c, ix_c, iy_c, rhs_info, ix, iy);
    pts.interpolate(row_map); 
  }
  else if (this->sim.tmp->Tree(rhsNei).CheckFiner())
  {
    const BlockInfo &rhsNei_f = this->sim.tmp->getBlockInfoAll(rhs_info.level + 1, indexer.Zchild(rhsNei, ix, iy));

    const int ix_f = (ix % (BSX_/2)) * 2;
    const int iy_f = (iy % (BSY_/2)) * 2;

    // Interpolate flux at interfaces with both fine neighbour cells
    PolyO3I pts1(indexer, false, rhs_info, ix, iy, rhsNei_f, ix_f, iy_f);
    pts1.interpolate(row_map);
    PolyO3I pts2(indexer, false, rhs_info, ix, iy, rhsNei_f, ix_f+1, iy_f+1); // same coarse indices here...
    pts2.interpolate(row_map);
  }
  else { throw std::runtime_error("Neighbour doesn't exist, isn't coarser, nor finer..."); }
}

template<class EdgeIndexer>
void AMRSolver::makeEdgeCellRow( // excluding corners
    const BlockInfo &rhs_info,
    const int &ix,
    const int &iy,
    const bool &isBoundary,
    const BlockInfo &rhsNei,
    const EdgeIndexer &indexer)
{
    std::map<int,double> row_map;

    const int sfc_idx = CellIndexer::This(rhs_info, ix, iy);
    const int n1_idx = indexer.inblock_n1(rhs_info, ix, iy); // in-block neighbour 1
    const int n2_idx = indexer.inblock_n2(rhs_info, ix, iy); // in-block neighbour 2
    const int n3_idx = indexer.inblock_n3(rhs_info, ix, iy); // in-block neighbour 3

    // Map fluxes associated to in-block edges
    row_map[n1_idx] += 1.;
    row_map[n2_idx] += 1.;
    row_map[n3_idx] += 1.;
    row_map[sfc_idx] += -3.;

    if (!isBoundary)
      this->makeFlux(rhs_info, ix, iy, row_map, rhsNei, indexer);

    this->cooMatPushBackRow(sfc_idx, row_map);
}

template<class EdgeIndexer1, class EdgeIndexer2>
void AMRSolver::makeCornerCellRow(
    const BlockInfo &rhs_info,
    const int &ix,
    const int &iy,
    const bool &isBoundary1,
    const BlockInfo &rhsNei_1,
    const EdgeIndexer1 &indexer1, 
    const bool &isBoundary2,
    const BlockInfo &rhsNei_2,
    const EdgeIndexer2 &indexer2)
{
    std::map<int,double> row_map;

    const int sfc_idx = CellIndexer::This(rhs_info, ix, iy);
    const int n1_idx = indexer1.inblock_n2(rhs_info, ix, iy); // indexer.inblock_n2 is the opposite edge
    const int n2_idx = indexer2.inblock_n2(rhs_info, ix, iy); // makes corner input order invariant

    // Map fluxes associated to in-block edges
    row_map[n1_idx] += 1.;
    row_map[n2_idx] += 1.;
    row_map[sfc_idx] += -2.;

    if (!isBoundary1)
      this->makeFlux(rhs_info, ix, iy, row_map, rhsNei_1, indexer1);
    if (!isBoundary2)
      this->makeFlux(rhs_info, ix, iy, row_map, rhsNei_2, indexer2);

    this->cooMatPushBackRow(sfc_idx, row_map);
}

void AMRSolver::get_LS()
{
  sim.startProfiler("Poisson solver: LS");

  //This returns an array with the blocks that the coarsest possible 
  //mesh would have (i.e. all blocks are at level 0)
  std::array<int, 3> blocksPerDim = sim.pres->getMaxBlocks();

  //Get a vector of all BlockInfos of the grid we're interested in
  std::vector<cubism::BlockInfo>&  RhsInfo = sim.tmp->getBlocksInfo();
  std::vector<cubism::BlockInfo>&  zInfo = sim.pres->getBlocksInfo();
  const int Nblocks = RhsInfo.size();
  const int N = BSX_*BSY_*Nblocks;

  if (updateA_ || virginA_)
  {
    // Allocate memory for solution 'x' and RHS vector 'b' on host
    this->x_.resize(N);
    this->b_.resize(N);
    // Clear contents from previous call of AMRSolver::solve() and reserve memory 
    // for sparse LHS matrix 'A' (for uniform grid at most 5 elements per row).
    this->cooValA_.clear();
    this->cooRowA_.clear();
    this->cooColA_.clear();
    this->cooValA_.reserve(6 * N);
    this->cooRowA_.reserve(6 * N);
    this->cooColA_.reserve(6 * N);

    // No parallel for to ensure COO are ordered at construction
    for(int i=0; i< Nblocks; i++)
    {    
      const BlockInfo &rhs_info = RhsInfo[i];
      const ScalarBlock & __restrict__ rhs  = *(ScalarBlock*) RhsInfo[i].ptrBlock;
      const ScalarBlock & __restrict__ p  = *(ScalarBlock*) zInfo[i].ptrBlock;

      //1.Check if this is a boundary block
      const int aux = 1 << rhs_info.level; // = 2^level
      const int MAX_X_BLOCKS = blocksPerDim[0]*aux - 1; //this means that if level 0 has blocksPerDim[0] blocks in the x-direction, level rhs.level will have this many blocks
      const int MAX_Y_BLOCKS = blocksPerDim[1]*aux - 1; //this means that if level 0 has blocksPerDim[1] blocks in the y-direction, level rhs.level will have this many blocks

      //index is the (i,j) coordinates of a block at the current level 
      const bool isWestBoundary  = (rhs_info.index[0] == 0           ); // don't check for west neighbor
      const bool isEastBoundary  = (rhs_info.index[0] == MAX_X_BLOCKS); // don't check for east neighbor
      const bool isSouthBoundary = (rhs_info.index[1] == 0           ); // don't check for south neighbor
      const bool isNorthBoundary = (rhs_info.index[1] == MAX_Y_BLOCKS); // don't check for north neighbor

      //2.Access the block's neighbors (for the Poisson solve in two dimensions we care about four neighbors in total)
      const long long Z_west  = rhs_info.Znei[1-1][1][1];
      const long long Z_east  = rhs_info.Znei[1+1][1][1];
      const long long Z_south = rhs_info.Znei[1][1-1][1];
      const long long Z_north = rhs_info.Znei[1][1+1][1];
      //rhs.Z == rhs.Znei[1][1][1] is true always

      const BlockInfo &rhsNei_west  = this->sim.tmp->getBlockInfoAll(rhs_info.level,Z_west );
      const BlockInfo &rhsNei_east  = this->sim.tmp->getBlockInfoAll(rhs_info.level,Z_east );
      const BlockInfo &rhsNei_south = this->sim.tmp->getBlockInfoAll(rhs_info.level,Z_south);
      const BlockInfo &rhsNei_north = this->sim.tmp->getBlockInfoAll(rhs_info.level,Z_north);

      //For later: there's a total of three boolean variables:
      // I.   grid->Tree(rhsNei_west).Exists()
      // II.  grid->Tree(rhsNei_west).CheckCoarser()
      // III. grid->Tree(rhsNei_west).CheckFiner()
      // And only one of them is true

      // Add matrix elements associated to interior cells of a block
      for(int iy=0; iy<BSY_; iy++)
      for(int ix=0; ix<BSX_; ix++)
      {
        // Following logic needs to be in for loop to assure cooRows are ordered
        if ((ix > 0 && ix<BSX_-1) && (iy > 0 && iy<BSY_-1))
        { // Inner cells
          const int sn_idx = CellIndexer::SouthNeighbour(rhs_info, ix, iy);
          const int wn_idx = CellIndexer::WestNeighbour(rhs_info, ix, iy);
          const int sfc_idx = CellIndexer::This(rhs_info, ix, iy);
          const int en_idx = CellIndexer::EastNeighbour(rhs_info, ix, iy);
          const int nn_idx = CellIndexer::NorthNeighbour(rhs_info, ix, iy);
          
          // Push back in ascending order for 'col_idx'
          this->cooMatPushBackVal(1., sfc_idx, sn_idx);
          this->cooMatPushBackVal(1., sfc_idx, wn_idx);
          this->cooMatPushBackVal(-4, sfc_idx, sfc_idx);
          this->cooMatPushBackVal(1., sfc_idx, en_idx);
          this->cooMatPushBackVal(1., sfc_idx, nn_idx);
        }
        else if (ix == 0 && (iy > 0 && iy < BSY_-1))
        { // west cells excluding corners
          this->makeEdgeCellRow(rhs_info, ix, iy, isWestBoundary, rhsNei_west, WestEdgeCell());
        }
        else if (ix == BSX_-1 && (iy > 0 && iy < BSY_-1))
        { // east cells excluding corners
          this->makeEdgeCellRow(rhs_info, ix, iy, isEastBoundary, rhsNei_east, EastEdgeCell());
        }
        else if ((ix > 0 && ix < BSX_-1) && iy == 0)
        { // south cells excluding corners
          this->makeEdgeCellRow(rhs_info, ix, iy, isSouthBoundary, rhsNei_south, SouthEdgeCell());
        }
        else if ((ix > 0 && ix < BSX_-1) && iy == BSY_-1)
        { // north cells excluding corners
          this->makeEdgeCellRow(rhs_info, ix, iy, isNorthBoundary, rhsNei_north, NorthEdgeCell());
        }
        else if (ix == 0 && iy == 0)
        { // south-west corner
          this->makeCornerCellRow(
              rhs_info, ix, iy, 
              isSouthBoundary, rhsNei_south, SouthEdgeCell(), 
              isWestBoundary, rhsNei_west, WestEdgeCell());
        }
        else if (ix == BSX_-1 && iy == 0)
        { // south-east corner
          this->makeCornerCellRow(
              rhs_info, ix, iy, 
              isEastBoundary, rhsNei_east, EastEdgeCell(), 
              isSouthBoundary, rhsNei_south, SouthEdgeCell());
        }
        else if (ix == 0 && iy == BSY_-1)
        { // north-west corner
          this->makeCornerCellRow(
              rhs_info, ix, iy, 
              isWestBoundary, rhsNei_west, WestEdgeCell(), 
              isNorthBoundary, rhsNei_north, NorthEdgeCell());
        }
        else if (ix == BSX_-1 && iy == BSY_-1)
        { // north-east corner
          this->makeCornerCellRow(
              rhs_info, ix, iy, 
              isNorthBoundary, rhsNei_north, NorthEdgeCell(), 
              isEastBoundary, rhsNei_east, EastEdgeCell());
        }
      } // for(int iy=0; iy<BSY_; iy++) for(int ix=0; ix<BSX_; ix++)
    } // for(int i=0; i< Nblocks; i++)
  } // if (updateA_)

  #pragma omp parallel for
  for(int i=0; i< Nblocks; i++)
  {    
    const BlockInfo &rhs_info = RhsInfo[i];
    const ScalarBlock & __restrict__ rhs  = *(ScalarBlock*) RhsInfo[i].ptrBlock;
    const ScalarBlock & __restrict__ p  = *(ScalarBlock*) zInfo[i].ptrBlock;

    // Construct RHS and x_0 vectors for linear system
    for(int iy=0; iy<BSY_; iy++)
    for(int ix=0; ix<BSX_; ix++)
    {
      const int sfc_idx = CellIndexer::This(rhs_info, ix, iy);
      b_[sfc_idx] = rhs(ix,iy).s;
      x_[sfc_idx] = p(ix,iy).s;
    }
  }

  // Save params of current linear system
  m_ = N; // rows
  n_ = N; // cols
  nnz_ = this->cooValA_.size(); // non-zero elements
  std::cout << "  [AMRSolver] linear system " 
            << "rows: " << m_  << " cols: " << n_ 
            << " non-zero elements: " << nnz_ << std::endl;

  sim.stopProfiler();
}

// -------------------------------------------------------------------------------------------------------------------- 
// -------------------------------------------------------------------------------------------------------------------- 

double AMRSolver::getA_local(int I1,int I2) //matrix for Poisson's equation on a uniform grid
{
   int j1 = I1 / BSX_;
   int i1 = I1 % BSX_;
   int j2 = I2 / BSX_;
   int i2 = I2 % BSX_;
   if (i1==i2 && j1==j2)
     return 4.0;
   else if (abs(i1-i2) + abs(j1-j2) == 1)
     return -1.0;
   else
     return 0.0;
}

AMRSolver::AMRSolver(SimulationData& s):sim(s)
{
  std::vector<std::vector<double>> L; // lower triangular matrix of Cholesky decomposition
  std::vector<std::vector<double>> L_inv; // inverse of L

  L.resize(BSZ_);
  L_inv.resize(BSZ_);
  for (int i(0); i<BSZ_ ; i++)
  {
    L[i].resize(i+1);
    L_inv[i].resize(i+1);
    // L_inv will act as right block in GJ algorithm, init as identity
    for (int j(0); j<=i; j++){
      L_inv[i][j] = (i == j) ? 1. : 0.;
    }
  }

  // compute the Cholesky decomposition of the preconditioner with Cholesky-Crout
  for (int i(0); i<BSZ_ ; i++)
  {
    double s1 = 0;
    for (int k(0); k<=i-1; k++)
      s1 += L[i][k]*L[i][k];
    L[i][i] = sqrt(getA_local(i,i) - s1);
    for (int j(i+1); j<BSZ_; j++)
    {
      double s2 = 0;
      for (int k(0); k<=i-1; k++)
        s2 += L[i][k]*L[j][k];
      L[j][i] = (getA_local(j,i)-s2) / L[i][i];
    }
  }

  /* Compute the inverse of the Cholesky decomposition L using Gauss-Jordan elimination.
     L will act as the left block (it does not need to be modified in the process), 
     L_inv will act as the right block and at the end of the algo will contain the inverse */
  for (int br(0); br<BSZ_; br++)
  { // 'br' - base row in which all columns up to L_lb[br][br] are already zero
    const double bsf = 1. / L[br][br];
    for (int c(0); c<=br; c++)
      L_inv[br][c] *= bsf;

    for (int wr(br+1); wr<BSZ_; wr++)
    { // 'wr' - working row where elements below L_lb[br][br] will be set to zero
      const double wsf = L[wr][br];
      for (int c(0); c<=br; c++)
        L_inv[wr][c] -= (wsf * L_inv[br][c]);
    }
  }

  // P_inv_ holds inverse preconditionner in row major order!
  P_inv_.resize(BSZ_ * BSZ_);
  for (int i(0); i<BSZ_; i++)
  for (int j(0); j<BSZ_; j++)
  {
    double aux = 0.;
    for (int k(0); k<BSZ_; k++) // P_inv_ = (L^T)^{-1} L^{-1}
      aux += (i <= k && j <=k) ? L_inv[k][i] * L_inv[k][j] : 0.;

    P_inv_[i*BSZ_+j] = aux;
  }

  // Set-up CUDA streams and handles
  checkCudaErrors(cudaStreamCreate(&solver_stream_));
  checkCudaErrors(cublasCreate(&cublas_handle_)); 
  checkCudaErrors(cusparseCreate(&cusparse_handle_)); 
  // Set handles to stream
  checkCudaErrors(cublasSetStream(cublas_handle_, solver_stream_));
  checkCudaErrors(cusparseSetStream(cusparse_handle_, solver_stream_));

  virginA_ = true;
  updateA_ = true;
  // NULL if device memory not allocated
  d_cooValA_ = NULL;
  d_cooRowA_ = NULL;
  d_cooColA_ = NULL;
  d_x_ = NULL;
  d_r_ = NULL;
  d_P_inv_ = NULL;

  d_rhat_ = NULL;
  d_p_ = NULL;
  d_nu_ = NULL;
  d_t_ = NULL;
  d_z_ = NULL;

  // Copy preconditionner
  checkCudaErrors(cudaMalloc(&d_P_inv_, BSZ_ * BSZ_ * sizeof(double)));
  checkCudaErrors(cudaMemcpyAsync(d_P_inv_, P_inv_.data(), BSZ_ * BSZ_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream_));
}

AMRSolver::~AMRSolver()
{
  std::cout << "---------------- Calling on AMRSolver() destructor ------------\n";

  // Free preconditionner
  checkCudaErrors(cudaFree(d_P_inv_));
  // Cleanup after last timestep
  checkCudaErrors(cudaFree(d_cooValA_));
  checkCudaErrors(cudaFree(d_cooRowA_));
  checkCudaErrors(cudaFree(d_cooColA_));
  checkCudaErrors(cudaFree(d_x_));
  checkCudaErrors(cudaFree(d_r_));
  checkCudaErrors(cusparseDestroySpMat(spDescrA_));
  checkCudaErrors(cusparseDestroyDnVec(spDescrX0_));
  checkCudaErrors(cusparseDestroyDnVec(spDescrZ_));
  checkCudaErrors(cusparseDestroyDnVec(spDescrNu_));
  checkCudaErrors(cusparseDestroyDnVec(spDescrT_));
  checkCudaErrors(cudaFree(d_rhat_));
  checkCudaErrors(cudaFree(d_p_));
  checkCudaErrors(cudaFree(d_nu_));
  checkCudaErrors(cudaFree(d_t_));
  checkCudaErrors(cudaFree(d_z_));
  checkCudaErrors(cudaFree(SpMVBuff_));

#ifdef BICGSTAB_PROFILER
  std::cout << "  [AMRSolver]: total elapsed time: " << pGlob_.elapsed() << " [ms]." << std::endl;
  std::cout << "  [AMRSolver]: memory transfers:   " << (pMemcpy_.elapsed()/pGlob_.elapsed())*100. << "%." << std::endl;
  std::cout << "  [AMRSolver]: preconditioning:    " << (pPrec_.elapsed()/pGlob_.elapsed())*100. << "%." << std::endl;
  std::cout << "  [AMRSolver]: SpVM:               " << (pSpMV_.elapsed()/pGlob_.elapsed())*100. << "%." << std::endl;
#endif
  // Destroy CUDA streams and handles
  checkCudaErrors(cublasDestroy(cublas_handle_)); 
  checkCudaErrors(cusparseDestroy(cusparse_handle_)); 
  checkCudaErrors(cudaStreamDestroy(solver_stream_));
}

void AMRSolver::alloc()
{
  if (updateA_)
  {
    if (!virginA_) // Previous time-step does not exist
    {
      // Free device memory allocated for linear system from previous time-step
      checkCudaErrors(cudaFree(d_cooValA_));
      checkCudaErrors(cudaFree(d_cooRowA_));
      checkCudaErrors(cudaFree(d_cooColA_));
      checkCudaErrors(cudaFree(d_x_));
      checkCudaErrors(cudaFree(d_r_));
      // Cleanup memory allocated for BiCGSTAB arrays
      checkCudaErrors(cusparseDestroySpMat(spDescrA_));
      checkCudaErrors(cusparseDestroyDnVec(spDescrX0_));
      checkCudaErrors(cusparseDestroyDnVec(spDescrZ_));
      checkCudaErrors(cusparseDestroyDnVec(spDescrNu_));
      checkCudaErrors(cusparseDestroyDnVec(spDescrT_));
      checkCudaErrors(cudaFree(d_rhat_));
      checkCudaErrors(cudaFree(d_p_));
      checkCudaErrors(cudaFree(d_nu_));
      checkCudaErrors(cudaFree(d_t_));
      checkCudaErrors(cudaFree(d_z_));
      checkCudaErrors(cudaFree(SpMVBuff_));
    }
    virginA_ = false;

    // Allocate device memory for linear system
    checkCudaErrors(cudaMalloc(&d_cooValA_, nnz_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_cooRowA_, nnz_ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_cooColA_, nnz_ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_x_, m_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_r_, m_ * sizeof(double)));

#ifdef BICGSTAB_PROFILER
    pMemcpy_.startProfiler(solver_stream_);
#endif
    // H2D transfer of linear system
    checkCudaErrors(cudaMemcpyAsync(d_cooValA_, cooValA_.data(), nnz_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream_));
    checkCudaErrors(cudaMemcpyAsync(d_cooRowA_, cooRowA_.data(), nnz_ * sizeof(int), cudaMemcpyHostToDevice, solver_stream_));
    checkCudaErrors(cudaMemcpyAsync(d_cooColA_, cooColA_.data(), nnz_ * sizeof(int), cudaMemcpyHostToDevice, solver_stream_));
#ifdef BICGSTAB_PROFILER
    pMemcpy_.stopProfiler(solver_stream_);
#endif

    // Allocate arrays for BiCGSTAB storage
    checkCudaErrors(cudaMalloc(&d_rhat_, m_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_p_, m_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_nu_, m_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_t_, m_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_z_, m_ * sizeof(double)));
    // Create descriptors for variables that will pass through cuSPARSE
    checkCudaErrors(cusparseCreateCoo(&spDescrA_, m_, n_, nnz_, d_cooRowA_, d_cooColA_, d_cooValA_, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&spDescrX0_, m_, d_x_, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&spDescrZ_, m_, d_z_, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&spDescrNu_, m_, d_nu_, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&spDescrT_, m_, d_t_, CUDA_R_64F));
    // Allocate work buffer for cusparseSpMV
    checkCudaErrors(cusparseSpMV_bufferSize(
          cusparse_handle_, 
          CUSPARSE_OPERATION_NON_TRANSPOSE, 
          &eye_, 
          spDescrA_, 
          spDescrX0_, 
          &nil_, 
          spDescrNu_, 
          CUDA_R_64F, 
          CUSPARSE_MV_ALG_DEFAULT, 
          &SpMVBuffSz_));
    checkCudaErrors(cudaMalloc(&SpMVBuff_, SpMVBuffSz_ * sizeof(char)));
  }

#ifdef BICGSTAB_PROFILER
  pMemcpy_.startProfiler(solver_stream_);
#endif
  // Copy RHS and initial guess in any case
  checkCudaErrors(cudaMemcpyAsync(d_x_, x_.data(), m_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream_));
  checkCudaErrors(cudaMemcpyAsync(d_r_, b_.data(), m_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream_));
#ifdef BICGSTAB_PROFILER
  pMemcpy_.stopProfiler(solver_stream_);
#endif
}

void AMRSolver::BiCGSTAB()
{
#ifdef BICGSTAB_PROFILER
  pGlob_.startProfiler(solver_stream_);
#endif 
  // Allocate device memory and copy from host
  this->alloc();

  const double max_error = this->sim.step < 10 ? 0.0 : sim.PoissonTol * sim.uMax_measured / sim.dt;
  const double max_rel_error = this->sim.step < 10 ? 0.0 : min(1e-2,sim.PoissonTolRel * sim.uMax_measured / sim.dt );
  const int max_restarts = this->sim.step < 10 ? 100 : sim.maxPoissonRestarts;

  // Initialize variables to evaluate convergence
  double error = 1e50;
  double error_init = 1e50;

  // 1. r <- b - A*x_0.  Add bias with cuBLAS like in "NVIDIA_CUDA-11.4_Samples/7_CUDALibraries/conjugateGradient"
#ifdef BICGSTAB_PROFILER
  pSpMV_.startProfiler(solver_stream_);
#endif
  checkCudaErrors(cusparseSpMV( // A*x_0
        cusparse_handle_, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, 
        &eye_, 
        spDescrA_, 
        spDescrX0_, 
        &nil_, 
        spDescrNu_, // Use d_nu_ as temporary storage for result A*x_0 
        CUDA_R_64F, 
        CUSPARSE_MV_ALG_DEFAULT, 
        SpMVBuff_)); 
#ifdef BICGSTAB_PROFILER
  pSpMV_.stopProfiler(solver_stream_);
#endif
  checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &nye_, d_nu_, 1, d_r_, 1)); // r <- -A*x_0 + b

  // Check norm of A*x_0
  checkCudaErrors(cublasDnrm2(cublas_handle_, m_, d_nu_, 1, &error_init));
  checkCudaErrors(cudaStreamSynchronize(solver_stream_));
  std::cout << "  [AMRSolver]: || A*x_0 || = " << error_init << std::endl;
  
  // Calculate x_error_init for max_rel_error comparisons
  checkCudaErrors(cublasDnrm2(cublas_handle_, m_, d_r_, 1, &error_init));
  checkCudaErrors(cudaStreamSynchronize(solver_stream_));
  std::cout << "  [AMRSolver]: Initial norm: " << error_init << std::endl;

  // 2. Set r_hat = r
  checkCudaErrors(cublasDcopy(cublas_handle_, m_, d_r_, 1, d_rhat_, 1));

  // 3. Set initial values to scalars
  bool bConverged = false;
  int restarts = 0;
  double rho_curr = 1.;
  double rho_prev = 1.;
  double alpha = 1.;
  double omega = 1.;
  double beta = 0.;
  const double eps = 1e-21;

  // 4. Set initial values of vectors to zero
  checkCudaErrors(cudaMemsetAsync(d_nu_, 0, m_ * sizeof(double), solver_stream_));
  checkCudaErrors(cudaMemsetAsync(d_p_, 0, m_ * sizeof(double), solver_stream_));

  // 5. Start iterations
  const size_t max_iter = 1000;
  for(size_t k(0); k<max_iter; k++)
  {
    // 1. rho_i = (r_hat, r)
    checkCudaErrors(cublasDdot(cublas_handle_, m_, d_rhat_, 1, d_r_, 1, &rho_curr));
    
    double norm_1 = 0.;
    double norm_2 = 0.;
    checkCudaErrors(cublasDnrm2(cublas_handle_, m_, d_r_, 1, &norm_1));
    checkCudaErrors(cublasDnrm2(cublas_handle_, m_, d_rhat_, 1, &norm_2));
    checkCudaErrors(cudaStreamSynchronize(solver_stream_)); // sync for 2. which happens on host
    // 2. beta = (rho_i / rho_{i-1}) * (alpha / omega_{i-1})
    beta = (rho_curr / (rho_prev+eps)) * (alpha / (omega+eps));

    // Numerical convergence trick
    const double cosTheta = rho_curr / norm_1 / norm_2;
    bool serious_breakdown = std::fabs(cosTheta) < 1e-8; 
    if(serious_breakdown && max_restarts > 0)
    {
      restarts++;
      if(restarts >= max_restarts){
        break;
      }
      std::cout << "  [AMRSolver]: Restart at iteration: " << k << " norm: " << error <<" Initial norm: " << error_init << std::endl;
      checkCudaErrors(cublasDcopy(cublas_handle_, m_, d_r_, 1, d_rhat_, 1));
      checkCudaErrors(cublasDnrm2(cublas_handle_, m_, d_rhat_, 1, &rho_curr));
      checkCudaErrors(cudaStreamSynchronize(solver_stream_)); 
      rho_curr *= rho_curr;
      rho_prev = 1.;
      alpha = 1.;
      omega = 1.;
      beta = (rho_curr / (rho_prev+eps)) * (alpha / (omega+eps));
    }

    // 3. p_i = r_{i-1} + beta(p_{i-1} - omega_{i-1}*nu_i)
    double nomega = -omega;
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &nomega, d_nu_, 1, d_p_, 1)); // p <- -omega_{i-1}*nu_i + p
    checkCudaErrors(cublasDscal(cublas_handle_, m_, &beta, d_p_, 1));            // p <- beta * p
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &eye_, d_r_, 1, d_p_, 1));    // p <- r_{i-1} + p

    // 4. z <- K_2^{-1} * p_i
#ifdef BICGSTAB_PROFILER
    pPrec_.startProfiler(solver_stream_);
#endif
    checkCudaErrors(cublasDgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, BSZ_, m_ / BSZ_, BSZ_, &eye_, d_P_inv_, BSZ_, d_p_, BSZ_, &nil_, d_z_, BSZ_));
#ifdef BICGSTAB_PROFILER
    pPrec_.stopProfiler(solver_stream_);
#endif

    // 5. nu_i = A * z
#ifdef BICGSTAB_PROFILER
    pSpMV_.startProfiler(solver_stream_);
#endif
    checkCudaErrors(cusparseSpMV(
          cusparse_handle_,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          &eye_,
          spDescrA_,
          spDescrZ_,
          &nil_,
          spDescrNu_,
          CUDA_R_64F,
          CUSPARSE_MV_ALG_DEFAULT,
          SpMVBuff_));
#ifdef BICGSTAB_PROFILER
    pSpMV_.stopProfiler(solver_stream_);
#endif

    // 6. alpha = rho_i / (r_hat, nu_i)
    double alpha_den;
    checkCudaErrors(cublasDdot(cublas_handle_, m_, d_rhat_, 1, d_nu_, 1, &alpha_den)); // alpha <- (r_hat, nu_i)
    checkCudaErrors(cudaStreamSynchronize(solver_stream_)); // sync for host division
    alpha = rho_curr / (alpha_den+eps); // alpha <- rho_i / alpha

    // 7. h = alpha*z + x_{i-1}
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &alpha, d_z_, 1, d_x_, 1));

    // 9. s = -alpha * nu_i + r_{i-1}
    const double nalpha = -alpha;
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &nalpha, d_nu_, 1, d_r_, 1));

    // 10. z <- K_2^{-1} * s
#ifdef BICGSTAB_PROFILER
    pPrec_.startProfiler(solver_stream_);
#endif
    checkCudaErrors(cublasDgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, BSZ_, m_ / BSZ_, BSZ_, &eye_, d_P_inv_, BSZ_, d_r_, BSZ_, &nil_, d_z_, BSZ_));
#ifdef BICGSTAB_PROFILER
    pPrec_.stopProfiler(solver_stream_);
#endif

    // 11. t = A * z
#ifdef BICGSTAB_PROFILER
    pSpMV_.startProfiler(solver_stream_);
#endif
    checkCudaErrors(cusparseSpMV(
          cusparse_handle_,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          &eye_,
          spDescrA_,
          spDescrZ_,
          &nil_,
          spDescrT_,
          CUDA_R_64F,
          CUSPARSE_MV_ALG_DEFAULT,
          SpMVBuff_));
#ifdef BICGSTAB_PROFILER
    pSpMV_.stopProfiler(solver_stream_);
#endif
    
    // 12. omega_i = (t,s)/(t,t), variables alpha & beta no longer in use this iter
    double omega_num;
    double omega_den;
    checkCudaErrors(cublasDdot(cublas_handle_, m_, d_t_, 1, d_r_, 1, &omega_num)); // alpha <- (t,s)
    checkCudaErrors(cublasDnrm2(cublas_handle_, m_, d_t_, 1, &omega_den));          // beta <- sqrt(t,t)
    checkCudaErrors(cudaStreamSynchronize(solver_stream_)); // sync for host arithmetic
    omega = omega_num / (omega_den * omega_den + eps);

    // 13. x_i = omega_i * z + h
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &omega, d_z_, 1, d_x_, 1));

    // 15. r_i = -omega_i * t + s
    nomega = -omega;
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &nomega, d_t_, 1, d_r_, 1));

    // If x_i accurate enough then quit
    checkCudaErrors(cublasDnrm2(cublas_handle_, m_, d_r_, 1, &error));
    checkCudaErrors(cudaStreamSynchronize(solver_stream_));

    if((error <= max_error) || (error / error_init <= max_rel_error))
    // if(x_error <= max_error)
    {
      std::cout << "  [AMRSolver]: Converged after " << k << " iterations" << std::endl;;
      bConverged = true;
      break;
    }

    // Update *_prev values for next iteration
    rho_prev = rho_curr;
  }

  if( bConverged )
    std::cout <<  "  [AMRSolver] Error norm (relative) = " << error << "/" << max_error 
              << " (" << error/error_init  << "/" << max_rel_error << ")" << std::endl;
  else
    std::cout <<  "  [AMRSolver]: Iteration " << max_iter 
              << ". Error norm (relative) = " << error << "/" << max_error 
              << " (" << error/error_init  << "/" << max_rel_error << ")" << std::endl;

#ifdef BICGSTAB_PROFILER
  pMemcpy_.startProfiler(solver_stream_);
#endif
  // Copy result back to host
  checkCudaErrors(cudaMemcpyAsync(x_.data(), d_x_, m_ * sizeof(double), cudaMemcpyDeviceToHost, solver_stream_));
#ifdef BICGSTAB_PROFILER
  pMemcpy_.stopProfiler(solver_stream_);
  pGlob_.stopProfiler(solver_stream_);
#endif
}

// ----------------------------------------------------------------------------------------------------------------------------------- 
// ----------------------------------------------------------------------------------------------------------------------------------- 

void AMRSolver::solve()
{

  std::cout << "--------------------- Calling on AMRSolver.solve() ------------------------ \n";

  if (sim.pres->UpdateFluxCorrection)
  {
    sim.pres->UpdateFluxCorrection = false;
    updateA_ = true;
  }
  else
  {
    updateA_ = false;
  }

  this->get_LS();
  this->BiCGSTAB();
  //Now that we found the solution, we just substract the mean to get a zero-mean solution. 
  //This can be done because the solver only cares about grad(P) = grad(P-mean(P))
  this->zero_mean();
}

void AMRSolver::zero_mean()
{
  std::vector<cubism::BlockInfo>&  zInfo = sim.pres->getBlocksInfo();
  const int Nblocks = zInfo.size();

  double avg = 0;
  double avg1 = 0;
  #pragma omp parallel
  {
     #pragma omp for reduction (+:avg,avg1)
     for(int i=0; i< Nblocks; i++)
     {
        ScalarBlock& P  = *(ScalarBlock*) zInfo[i].ptrBlock;
        const double vv = zInfo[i].h*zInfo[i].h;
        for(int iy=0; iy<BSY_; iy++)
        for(int ix=0; ix<BSX_; ix++)
        {
            P(ix,iy).s = x_[i*BSX_*BSY_ + iy*BSX_ + ix];
            avg += P(ix,iy).s * vv;
            avg1 += vv;
        }
     }
     #pragma omp single
     {
        avg = avg/avg1;
     }
     #pragma omp for
     for(int i=0; i< Nblocks; i++)
     {
        ScalarBlock& P  = *(ScalarBlock*) zInfo[i].ptrBlock;
        for(int iy=0; iy<BSY_; iy++)
        for(int ix=0; ix<BSX_; ix++)
           P(ix,iy).s += -avg;
     }
  }
}

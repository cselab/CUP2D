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

// -------------------------------- Host-side construction of linear system -----------------------------------

enum Compass {North, East, South, West};

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

  protected:
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
    static int neiblock_n(const BlockInfo &info, const int &ix, const int &iy, const int offset = 0)
    { return SouthmostCell(info, ix, iy, offset); }

    constexpr static Compass EdgeType = {North};
    constexpr static std::array<int,3> Zchild1_idx = {0,0,0};
    constexpr static std::array<int,3> Zchild2_idx = {1,0,0};
};

class EastEdgeCell : public CellIndexer{
  public:
    static int inblock_n1(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return SouthNeighbour(info, ix, iy, dist); }
    static int inblock_n2(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return WestNeighbour(info, ix, iy, dist); }
    static int inblock_n3(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return NorthNeighbour(info, ix, iy, dist); }
    static int neiblock_n(const BlockInfo &info, const int &ix, const int &iy, const int offset = 0)
    { return WestmostCell(info, ix, iy, offset); }

    constexpr static Compass EdgeType = {East};
    constexpr static std::array<int,3> Zchild1_idx = {0,0,0};
    constexpr static std::array<int,3> Zchild2_idx = {0,1,0};
};

class SouthEdgeCell : public CellIndexer{
  public:
    static int inblock_n1(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return WestNeighbour(info, ix, iy, dist); }
    static int inblock_n2(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return NorthNeighbour(info, ix, iy, dist); }
    static int inblock_n3(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return EastNeighbour(info, ix, iy, dist); }
    static int neiblock_n(const BlockInfo &info, const int &ix, const int &iy, const int offset = 0)
    { return NorthmostCell(info, ix, iy, offset); }

    constexpr static Compass EdgeType = {South};
    constexpr static std::array<int,3> Zchild1_idx = {0,1,0};
    constexpr static std::array<int,3> Zchild2_idx = {1,1,0};
};

class WestEdgeCell : public CellIndexer{
  public:
    static int inblock_n1(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return NorthNeighbour(info, ix, iy, dist); }
    static int inblock_n2(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return EastNeighbour(info, ix, iy, dist); }
    static int inblock_n3(const BlockInfo &info, const int &ix, const int &iy, const int dist = 1)
    { return SouthNeighbour(info, ix, iy, dist); }
    static int neiblock_n(const BlockInfo &info, const int &ix, const int &iy, const int offset = 0)
    { return EastmostCell(info, ix, iy, offset); }

    constexpr static Compass EdgeType = {West};
    constexpr static std::array<int,3> Zchild1_idx = {1,0,0};
    constexpr static std::array<int,3> Zchild2_idx = {1,1,0};
};

static long long getZchild(const BlockInfo& info, const std::array<int,3> &Zchild_idx)
{
  return info.Zchild[Zchild_idx[0]][Zchild_idx[1]][Zchild_idx[2]];
}

// Central Difference "Upper" Taylor approximation (positive 1st order term)
class CDUpperTaylor {
  public:
    void operator()(
        const double &sign,
        const int &centre_idx,
        const int &bottom_idx,
        const int &top_idx,
        std::map<int,double> &row_map) const
    {
      row_map[centre_idx] += sign*p_centre;
      row_map[bottom_idx] += sign*p_bottom;
      row_map[top_idx] += sign*p_top;
    }
  private:
    // 8./15. comes from coeff of polynomial at 2nd step of interpolation
    static constexpr double p_centre = (8./15.) * (1. - 1./16.);
    static constexpr double p_bottom = (8./15.) * (-1./8. + 1./32.);
    static constexpr double p_top = (8./15.) * ( 1./8. + 1./32.);
};

// Central Difference "Lower" Taylor approximation (negative 1st order term)
class CDLowerTaylor {
  public:
    void operator()(
        const double& sign,
        const int &centre_idx,
        const int &bottom_idx,
        const int &top_idx,
        std::map<int,double> &row_map) const
    {
      row_map[centre_idx] += sign*p_centre;
      row_map[bottom_idx] += sign*p_bottom;
      row_map[top_idx] += sign*p_top;
    }
  private:
    static constexpr double p_centre = (8./15.) * (1. - 1./16.);
    static constexpr double p_bottom = (8./15.) * ( 1./8. + 1./32.);
    static constexpr double p_top = (8./15.) * (-1./8. + 1./32.);
};

class BiasedUpperTaylor {
  public:
    void operator()(
        const double& sign,
        const int &centre_idx,
        const int &offset1_idx,
        const int &offset2_idx,
        std::map<int,double> &row_map) const
    {
      row_map[centre_idx] += sign*p_centre;
      row_map[offset1_idx] += sign*p_offset1;
      row_map[offset2_idx] += sign*p_offset2;
    }
  private:
    static constexpr double p_centre = (8./15.) * (1 + 3./8. + 1./32.);
    static constexpr double p_offset1 = (8./15.) * (-1./2. - 1./16.);
    static constexpr double p_offset2 = (8./15.) * ( 1./8. + 1./32.);
};

class BiasedLowerTaylor {
  public:
    void operator()(
        const double& sign,
        const int &centre_idx,
        const int &offset1_idx,
        const int &offset2_idx,
        std::map<int,double> &row_map) const
    {
      row_map[centre_idx] += sign*p_centre;
      row_map[offset1_idx] += sign*p_offset1;
      row_map[offset2_idx] += sign*p_offset2;
    }
  private:
    static constexpr double p_centre = (8./15.) * (1 - 3./8. + 1./32.);
    static constexpr double p_offset1 = (8./15.) * ( 1./2. - 1./16.);
    static constexpr double p_offset2 = (8./15.) * (-1./8. + 1./32.);
};

// Typedefs to map offset based biased functionals to forward/backward differences in corners
typedef BiasedUpperTaylor BDUpperTaylor;
typedef BiasedLowerTaylor BDLowerTaylor;
typedef BiasedLowerTaylor FDUpperTaylor;
typedef BiasedUpperTaylor FDLowerTaylor;

class PolyInterpolation {
  public:
    template<class F>
    void operator()(
        const double& sign,
        const int &coarse_centre_idx,
        const int &coarse_offset1_idx, // bottom
        const int &coarse_offset2_idx, // top
        const int &fine_close_idx,
        const int &fine_far_idx,
        std::map<int,double> &row_map,
        const F &taylor) const
    {
      taylor(sign, coarse_centre_idx, coarse_offset1_idx, coarse_offset2_idx, row_map);
      row_map[fine_close_idx] += sign * p_fine_close;
      row_map[fine_far_idx] += sign * p_fine_far;
    } 
  private:
    static constexpr double p_fine_close = 2./3.;
    static constexpr double p_fine_far = -1./5.;
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
  const PolyInterpolation interpolator;

  if (this->sim.tmp->Tree(rhsNei).Exists())
  { 
    const int n_idx = indexer.neiblock_n(rhsNei, ix, iy);

    // Map flux associated to out-of-block edges at the same level of refinement
    row_map[n_idx] += 1.; row_map[sfc_idx] += -1.;
  }
  else if (this->sim.tmp->Tree(rhsNei).CheckCoarser())
  {
    const BlockInfo &rhsNei_c = this->sim.tmp->getBlockInfoAll(rhs_info.level - 1 , rhsNei.Zparent);

    const bool NorthSouthEdge = indexer.EdgeType == North || indexer.EdgeType == South;
    const bool EastWestEdge = indexer.EdgeType == East || indexer.EdgeType == West;
    int ix_c = ix / 2;
    int iy_c = iy / 2;
    if (NorthSouthEdge && (rhs_info.index[0] % 2 == 1)) // North/South edge leftward fine block 
      ix_c += BSX_ / 2;
    else if (EastWestEdge && (rhs_info.index[1] % 2 == 1)) // East/West edge top fine block
      iy_c += BSY_ / 2;

    // Map flux associated to interpolation at interface
    const int coarse_centre_idx = indexer.neiblock_n(rhsNei_c, ix_c, iy_c);
    const int fine_far_idx = indexer.inblock_n2(rhs_info, ix, iy); // index of cell on opposite inblock edge
    if ( (ix_c == 0 && NorthSouthEdge) || (iy_c == 0 && EastWestEdge) )
    {
      const int coarse_offset1_idx = indexer.neiblock_n(rhsNei_c, ix_c+1,iy_c+1);
      const int coarse_offset2_idx = indexer.neiblock_n(rhsNei_c, ix_c+2,iy_c+2);

      if ( (ix == 0 && NorthSouthEdge) || (iy == 0 && EastWestEdge) )
        interpolator(1., coarse_centre_idx, coarse_offset1_idx, coarse_offset2_idx, sfc_idx, fine_far_idx, row_map, FDLowerTaylor());
      else if ( (ix == 1 && NorthSouthEdge) || (iy == 1 && EastWestEdge) )
        interpolator(1., coarse_centre_idx, coarse_offset1_idx, coarse_offset2_idx, sfc_idx, fine_far_idx, row_map, FDUpperTaylor());
    }
    else if ( (ix_c == (BSX_-1) && NorthSouthEdge) || (iy_c == (BSY_-1) && EastWestEdge) )
    {
      const int coarse_offset1_idx = indexer.neiblock_n(rhsNei_c, ix_c-1,iy_c-1);
      const int coarse_offset2_idx = indexer.neiblock_n(rhsNei_c, ix_c-2,iy_c-2);

      if ( (ix == BSX_-2 && NorthSouthEdge) || (iy == BSY_-2 && EastWestEdge) )
        interpolator(1., coarse_centre_idx, coarse_offset1_idx, coarse_offset2_idx, sfc_idx, fine_far_idx, row_map, BDLowerTaylor());
      else if ( (ix == BSX_-1 && NorthSouthEdge) || (iy == BSY_-1 && EastWestEdge) )
        interpolator(1., coarse_centre_idx, coarse_offset1_idx, coarse_offset2_idx, sfc_idx, fine_far_idx, row_map, BDUpperTaylor());
    }
    else
    {
      const int coarse_offset1_idx = indexer.neiblock_n(rhsNei_c, ix_c-1,iy_c-1); // bottom
      const int coarse_offset2_idx = indexer.neiblock_n(rhsNei_c, ix_c+1,iy_c+1); // top

      if ( (ix % 2 == 0 && NorthSouthEdge) || (iy % 2 == 0 && EastWestEdge) )
        interpolator(1., coarse_centre_idx, coarse_offset1_idx, coarse_offset2_idx, sfc_idx, fine_far_idx, row_map, CDLowerTaylor());
      else if ( (ix % 2 == 1 && NorthSouthEdge) || (iy % 2 == 1 && EastWestEdge) )
        interpolator(1., coarse_centre_idx, coarse_offset1_idx, coarse_offset2_idx, sfc_idx, fine_far_idx, row_map, CDUpperTaylor());
    }

    // Map negative diagonal contribution to flux
    row_map[sfc_idx] += -1.;
  }
  else if (this->sim.tmp->Tree(rhsNei).CheckFiner())
  {
    const bool NorthSouthEdge = indexer.EdgeType == North || indexer.EdgeType == South;
    const bool EastWestEdge = indexer.EdgeType == East || indexer.EdgeType == West;

    /* Determine which fine block the current coarse edge neighbours.
       It is assumed that members 'Zchild_1' and 'Zchild_2' respect this order:
       Zchild[0][0][0] is (2i  ,2j  ,2k  )
       Zchild[1][0][0] is (2i+1,2j  ,2k  )
       Zchild[0][1][0] is (2i  ,2j+1,2k  )
       Zchild[1][1][0] is (2i+1,2j+1,2k  ) */

    long long rhsNei_Zchild;
    if (NorthSouthEdge)
      rhsNei_Zchild = ix < BSX_ / 2 ? getZchild(rhsNei, indexer.Zchild1_idx) : getZchild(rhsNei, indexer.Zchild2_idx);
    else if (EastWestEdge)
      rhsNei_Zchild = iy < BSY_ / 2 ? getZchild(rhsNei, indexer.Zchild1_idx) : getZchild(rhsNei, indexer.Zchild2_idx);
    const BlockInfo &rhsNei_f = this->sim.tmp->getBlockInfoAll(rhs_info.level + 1, rhsNei_Zchild);

    const int ix_f = (ix % (BSX_/2)) * 2;
    const int iy_f = (iy % (BSY_/2)) * 2;

    int coarse_offset1_idx;
    int coarse_offset2_idx;
    int fine_close_idx;
    int fine_far_idx;

    if ( (ix == 0 && iy == 0) || (ix == 0 && iy == BSY_-1) || (ix == BSX_-1 && iy == 0) || (ix == BSX_-1 && iy == BSY_-1) )
    {
      if ( (ix == 0) && NorthSouthEdge || (iy == 0) && EastWestEdge )
      {
        if ( (ix == 0) && NorthSouthEdge )
        {
          coarse_offset1_idx = CellIndexer::EastNeighbour(rhs_info, ix, iy, 1);
          coarse_offset2_idx = CellIndexer::EastNeighbour(rhs_info, ix, iy, 2);
        }
        else if ( (iy == 0) && EastWestEdge )
        {
          coarse_offset1_idx = CellIndexer::NorthNeighbour(rhs_info, ix, iy, 1);
          coarse_offset2_idx = CellIndexer::NorthNeighbour(rhs_info, ix, iy, 2);
        }
        // Add flux at left/lower corner interface
        fine_close_idx = indexer.neiblock_n(rhsNei_f, ix_f, iy_f, 0);
        fine_far_idx = indexer.neiblock_n(rhsNei_f, ix_f, iy_f, 1);
        row_map[fine_close_idx] += 1.;
        interpolator(-1., sfc_idx, coarse_offset1_idx, coarse_offset2_idx, fine_close_idx, fine_far_idx, row_map, FDLowerTaylor());

        // Add flux at right/higher corner interface
        fine_close_idx = indexer.neiblock_n(rhsNei_f, ix_f+1, iy_f+1, 0);
        fine_far_idx = indexer.neiblock_n(rhsNei_f, ix_f+1, iy_f+1, 1);
        row_map[fine_close_idx] += 1.;
        interpolator(-1., sfc_idx, coarse_offset1_idx, coarse_offset2_idx, fine_close_idx, fine_far_idx, row_map, FDUpperTaylor());
      }
      else if ( (ix == BSX_-1) && NorthSouthEdge || (iy == BSY_-1) && EastWestEdge )
      {
        if ( (ix == BSX_-1) && NorthSouthEdge )
        {
          coarse_offset1_idx = CellIndexer::WestNeighbour(rhs_info, ix, iy, 1);
          coarse_offset2_idx = CellIndexer::WestNeighbour(rhs_info, ix, iy, 2);
        }
        else if ( (iy == BSY_-1) && EastWestEdge )
        {
          coarse_offset1_idx = CellIndexer::SouthNeighbour(rhs_info, ix, iy, 1);
          coarse_offset2_idx = CellIndexer::SouthNeighbour(rhs_info, ix, iy, 2);
        }
        // Add flux at left/lower corner interface
        fine_close_idx = indexer.neiblock_n(rhsNei_f, ix_f, iy_f, 0);
        fine_far_idx = indexer.neiblock_n(rhsNei_f, ix_f, iy_f, 1);
        row_map[fine_close_idx] += 1.;
        interpolator(-1., sfc_idx, coarse_offset1_idx, coarse_offset2_idx, fine_close_idx, fine_far_idx, row_map, BDLowerTaylor());

        // Add flux at right/higher corner interface
        fine_close_idx = indexer.neiblock_n(rhsNei_f, ix_f+1, iy_f+1, 0);
        fine_far_idx = indexer.neiblock_n(rhsNei_f, ix_f+1, iy_f+1, 1);
        row_map[fine_close_idx] += 1.;
        interpolator(-1., sfc_idx, coarse_offset1_idx, coarse_offset2_idx, fine_close_idx, fine_far_idx, row_map, BDUpperTaylor());
      }
    }
    else
    {
      if (NorthSouthEdge)
      {
        coarse_offset1_idx = CellIndexer::WestNeighbour(rhs_info, ix, iy);  // bottom
        coarse_offset2_idx = CellIndexer::EastNeighbour(rhs_info, ix, iy);  // top
      } 
      else if (EastWestEdge)
      {
        coarse_offset1_idx = CellIndexer::SouthNeighbour(rhs_info, ix, iy); // bottom
        coarse_offset2_idx = CellIndexer::NorthNeighbour(rhs_info, ix, iy); // top 
      } 
      // Add flux at left/lower corner interface
      fine_close_idx = indexer.neiblock_n(rhsNei_f, ix_f, iy_f, 0);
      fine_far_idx = indexer.neiblock_n(rhsNei_f, ix_f, iy_f, 1);
      row_map[fine_close_idx] += 1.;
      interpolator(-1., sfc_idx, coarse_offset1_idx, coarse_offset2_idx, fine_close_idx, fine_far_idx, row_map, CDLowerTaylor());

      // Add flux at right/higher corner interface
      fine_close_idx = indexer.neiblock_n(rhsNei_f, ix_f+1, iy_f+1, 0);
      fine_far_idx = indexer.neiblock_n(rhsNei_f, ix_f+1, iy_f+1, 1);
      row_map[fine_close_idx] += 1.;
      interpolator(-1., sfc_idx, coarse_offset1_idx, coarse_offset2_idx, fine_close_idx, fine_far_idx, row_map, CDUpperTaylor());
    }
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
    this->csrRowA_.clear();
    this->cooColA_.clear();
    this->cooValA_.reserve(6 * N);
    this->cooRowA_.reserve(6 * N);
    this->csrRowA_.reserve(N+1);
    this->cooColA_.reserve(6 * N);

    // No parallel for to ensure COO and CSR are ordered at construction
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
        this->csrRowA_.push_back(cooValA_.size());
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
  this->csrRowA_.push_back(nnz_);
  std::cout << "Rows: " << m_  
            << " cols: " << n_ 
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
  d_csrRowA_ = NULL;
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

#ifdef BICGSTAB_PROFILER
  // Setup profiler
  elapsed_memcpy_ = 0.;
  elapsed_precondition_ = 0.;
  elapsed_bicgstab_ = 0.;
  checkCudaErrors(cudaEventCreate(&start_memcpy_));
  checkCudaErrors(cudaEventCreate(&stop_memcpy_));
  checkCudaErrors(cudaEventCreate(&start_precondition_));
  checkCudaErrors(cudaEventCreate(&stop_precondition_));
  checkCudaErrors(cudaEventCreate(&start_bicgstab_));
  checkCudaErrors(cudaEventCreate(&stop_bicgstab_));
#endif // BICGSTAB_PROFILER
}

AMRSolver::~AMRSolver()
{
  std::cout << "---------------- Calling on AMRSolver() destructor ------------\n";

  // Free preconditionner
  checkCudaErrors(cudaFree(d_P_inv_));
  // Cleanup after last timestep
  checkCudaErrors(cudaFree(d_cooValA_));
  checkCudaErrors(cudaFree(d_csrRowA_));
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
  std::cout << "  [AMRSolver]: total elapsed time: " << elapsed_bicgstab_ << " [ms]." << std::endl;
  std::cout << "  [AMRSolver]: memory transfers:   " << (elapsed_memcpy_/elapsed_bicgstab_)*100. << "%." << std::endl;
  std::cout << "  [AMRSolver]: preconditioning:    " << (elapsed_precondition_/elapsed_bicgstab_)*100. << "%." << std::endl;
  checkCudaErrors(cudaEventDestroy(start_memcpy_));
  checkCudaErrors(cudaEventDestroy(stop_memcpy_));
  checkCudaErrors(cudaEventDestroy(start_precondition_));
  checkCudaErrors(cudaEventDestroy(stop_precondition_));
  checkCudaErrors(cudaEventDestroy(start_bicgstab_));
  checkCudaErrors(cudaEventDestroy(stop_bicgstab_));
#endif
  // Destroy CUDA streams and handles
  checkCudaErrors(cublasDestroy(cublas_handle_)); 
  checkCudaErrors(cusparseDestroy(cusparse_handle_)); 
  checkCudaErrors(cudaStreamDestroy(solver_stream_));
}


#ifdef BICGSTAB_PROFILER
static void startProfiler(cudaEvent_t start, cudaStream_t stream)
{
  checkCudaErrors(cudaEventRecord(start, stream));
}

static void stopProfiler(float &total_elapsed_ms, cudaEvent_t start, cudaEvent_t stop, cudaStream_t stream)
{
  checkCudaErrors(cudaEventRecord(stop, stream));
  checkCudaErrors(cudaEventSynchronize(stop));

  float this_ms = 0.;
  checkCudaErrors(cudaEventElapsedTime(&this_ms, start, stop));
  total_elapsed_ms += this_ms;
}
#endif // BICGSTAB_PROFILER

void AMRSolver::alloc()
{
#ifdef BICGSTAB_PROFILER
  startProfiler(start_bicgstab_, solver_stream_);
  startProfiler(start_memcpy_, solver_stream_);
#endif 

  if (updateA_)
  {
    if (!virginA_) // Previous time-step does not exist
    {
      // Free device memory allocated for linear system from previous time-step
      checkCudaErrors(cudaFree(d_cooValA_));
      checkCudaErrors(cudaFree(d_csrRowA_));
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
    checkCudaErrors(cudaMalloc(&d_csrRowA_, (m_+1) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_cooColA_, nnz_ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_x_, m_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_r_, m_ * sizeof(double)));
    // H2D transfer of linear system
    checkCudaErrors(cudaMemcpyAsync(d_cooValA_, cooValA_.data(), nnz_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream_));
    checkCudaErrors(cudaMemcpyAsync(d_csrRowA_, csrRowA_.data(), (m_+1) * sizeof(int), cudaMemcpyHostToDevice, solver_stream_));
    checkCudaErrors(cudaMemcpyAsync(d_cooColA_, cooColA_.data(), nnz_ * sizeof(int), cudaMemcpyHostToDevice, solver_stream_));

    // Allocate arrays for BiCGSTAB storage
    checkCudaErrors(cudaMalloc(&d_rhat_, m_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_p_, m_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_nu_, m_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_t_, m_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_z_, m_ * sizeof(double)));
    // Create descriptors for variables that will pass through cuSPARSE
    checkCudaErrors(cusparseCreateCsr(&spDescrA_, m_, n_, nnz_, d_csrRowA_, d_cooColA_, d_cooValA_, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
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
  // Copy RHS and initial guess in any case
  checkCudaErrors(cudaMemcpyAsync(d_x_, x_.data(), m_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream_));
  checkCudaErrors(cudaMemcpyAsync(d_r_, b_.data(), m_ * sizeof(double), cudaMemcpyHostToDevice, solver_stream_));

#ifdef BICGSTAB_PROFILER
  stopProfiler(elapsed_memcpy_, start_memcpy_, stop_memcpy_, solver_stream_);
#endif
}

void AMRSolver::BiCGSTAB()
{
#ifdef BICGSTAB_PROFILER
  startProfiler(start_bicgstab_, solver_stream_);
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
  checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &nye_, d_nu_, 1, d_r_, 1)); // r <- -A*x_0 + b

#ifdef BICGSTAB_PROFILER
  // Check norm of A*x_0
  checkCudaErrors(cublasDnrm2(cublas_handle_, m_, d_nu_, 1, &error_init));
  checkCudaErrors(cudaStreamSynchronize(solver_stream_));
  std::cout << "  [AMRSolver]: || A*x_0 || = " << error_init << std::endl;
#endif
  
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
    startProfiler(start_precondition_, solver_stream_);
#endif
    checkCudaErrors(cublasDgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, BSZ_, m_ / BSZ_, BSZ_, &eye_, d_P_inv_, BSZ_, d_p_, BSZ_, &nil_, d_z_, BSZ_));
#ifdef BICGSTAB_PROFILER
    stopProfiler(elapsed_precondition_, start_precondition_, stop_precondition_, solver_stream_);
#endif

    // 5. nu_i = A * z
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
    startProfiler(start_precondition_, solver_stream_);
#endif
    checkCudaErrors(cublasDgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, BSZ_, m_ / BSZ_, BSZ_, &eye_, d_P_inv_, BSZ_, d_r_, BSZ_, &nil_, d_z_, BSZ_));
#ifdef BICGSTAB_PROFILER
    stopProfiler(elapsed_precondition_, start_precondition_, stop_precondition_, solver_stream_);
#endif

    // 11. t = A * z
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
  startProfiler(start_memcpy_, solver_stream_);
#endif
  // Copy result back to host
  checkCudaErrors(cudaMemcpyAsync(x_.data(), d_x_, m_ * sizeof(double), cudaMemcpyDeviceToHost, solver_stream_));
#ifdef BICGSTAB_PROFILER
  stopProfiler(elapsed_memcpy_, start_memcpy_, stop_memcpy_, solver_stream_);
  stopProfiler(elapsed_bicgstab_, start_bicgstab_, stop_bicgstab_, solver_stream_);
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

//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "ExpAMRSolver.h"

using namespace cubism;

// Five-point interpolation machinery
enum FDType {FDLower, FDUpper, BDLower, BDUpper, CDLower, CDUpper};

class PolyO3I {
  public:
    template<class EdgeCellIndexer>
    PolyO3I(SimulationData &s,
        const BlockInfo &info_c, const int &ix_c, const int &iy_c,
        const BlockInfo &info_f, const int &ix_f, const int &iy_f,
        const bool &neiCoarser, const EdgeCellIndexer &indexer,
        SpRowInfo &row)
      : rank_c_(s.tmp->Tree(info_c).rank()),
        rank_f_(s.tmp->Tree(info_f).rank()),
        sign_(neiCoarser ? 1. : -1.), row_(row)
    {
      if (neiCoarser) // thisFiner
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

        fine_close_idx_ = indexer.This(info_f, ix_f, iy_f);
        fine_far_idx_ = indexer.inblock_n2(info_f, ix_f, iy_f);
      }
      else // neiFiner, thisCoarser
      {
        coarse_centre_idx_ = indexer.This(info_c, ix_c, iy_c);

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

      this->interpolate();
    }

  private:
    // Central Difference "Upper" Taylor approximation (positive 1st order term)
    void CDUpperTaylor()
    {
      // 8./15. comes from coeff of polynomial at 2nd step of interpolation
      static constexpr double p_centre = (8./15.) * (1. - 1./16.);
      static constexpr double p_bottom = (8./15.) * (-1./8. + 1./32.);
      static constexpr double p_top = (8./15.) * ( 1./8. + 1./32.);

      row_.mapColVal(rank_c_, coarse_centre_idx_, sign_*p_centre);
      row_.mapColVal(rank_c_, coarse_offset1_idx_, sign_*p_bottom);
      row_.mapColVal(rank_c_, coarse_offset2_idx_, sign_*p_top);
    }

    // Central Difference "Lower" Taylor approximation (negative 1st order term)
    void CDLowerTaylor()
    {
      static constexpr double p_centre = (8./15.) * (1. - 1./16.);
      static constexpr double p_bottom = (8./15.) * ( 1./8. + 1./32.);
      static constexpr double p_top = (8./15.) * (-1./8. + 1./32.);

      row_.mapColVal(rank_c_, coarse_centre_idx_, sign_*p_centre);
      row_.mapColVal(rank_c_, coarse_offset1_idx_, sign_*p_bottom);
      row_.mapColVal(rank_c_, coarse_offset2_idx_, sign_*p_top);
    }

    void BiasedUpperTaylor()
    {
      static constexpr double p_centre = (8./15.) * (1 + 3./8. + 1./32.);
      static constexpr double p_offset1 = (8./15.) * (-1./2. - 1./16.);
      static constexpr double p_offset2 = (8./15.) * ( 1./8. + 1./32.);

      row_.mapColVal(rank_c_, coarse_centre_idx_, sign_*p_centre);
      row_.mapColVal(rank_c_, coarse_offset1_idx_, sign_*p_offset1);
      row_.mapColVal(rank_c_, coarse_offset2_idx_, sign_*p_offset2);
    }

    void BiasedLowerTaylor()
    {
      static constexpr double p_centre = (8./15.) * (1 - 3./8. + 1./32.);
      static constexpr double p_offset1 = (8./15.) * ( 1./2. - 1./16.);
      static constexpr double p_offset2 = (8./15.) * (-1./8. + 1./32.);

      row_.mapColVal(rank_c_, coarse_centre_idx_, sign_*p_centre);
      row_.mapColVal(rank_c_, coarse_offset1_idx_, sign_*p_offset1);
      row_.mapColVal(rank_c_, coarse_offset2_idx_, sign_*p_offset2);
    }

    // Aliases for offset based biased functionals to forward/backward differences in corners
    void BDUpperTaylor() { return BiasedUpperTaylor(); }
    void BDLowerTaylor() { return BiasedLowerTaylor(); }
    void FDUpperTaylor() { return BiasedLowerTaylor(); }
    void FDLowerTaylor() { return BiasedUpperTaylor(); }

    // F_i = sign * (p_{interpolation} - p_{fine_cell_idx})
    void interpolate()
    {
      static constexpr double p_fine_close = 2./3.;
      static constexpr double p_fine_far = -1./5.;

      // Interpolated flux sign * p_{interpolation}
      switch(type_)
      {
        case FDLower:
          FDLowerTaylor();
          break;
        case FDUpper:
          FDUpperTaylor();
          break;
        case BDLower:
          BDLowerTaylor();
          break;
        case BDUpper:
          BDUpperTaylor();
          break;
        case CDLower:
          CDLowerTaylor();
          break;
        case CDUpper:
          CDUpperTaylor();
          break;
      }

      row_.mapColVal(rank_f_, fine_close_idx_, sign_*p_fine_close);
      row_.mapColVal(rank_f_, fine_far_idx_, sign_*p_fine_far);

      // Non-interpolated flux contribution -sign * p_{fine_cell_idx}
      row_.mapColVal(rank_f_, fine_close_idx_, -sign_);
    } 

  private:
    const int rank_c_;
    const int rank_f_;
    const double sign_;
    SpRowInfo &row_;
    long long coarse_centre_idx_;
    long long coarse_offset1_idx_; // bottom/left
    long long coarse_offset2_idx_; // top/right
    long long fine_close_idx_;
    long long fine_far_idx_;
    FDType type_;
};



double ExpAMRSolver::getA_local(int I1,int I2) //matrix for Poisson's equation on a uniform grid
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

ExpAMRSolver::ExpAMRSolver(SimulationData& s)
  : sim(s), m_comm_(sim.comm), 
    NorthCell(s, Nblocks_xcumsum_), EastCell(s, Nblocks_xcumsum_), 
    SouthCell(s, Nblocks_xcumsum_), WestCell(s, Nblocks_xcumsum_)
{
  // MPI
  MPI_Comm_rank(m_comm_, &rank_);
  MPI_Comm_size(m_comm_, &comm_size_);

  Nblocks_xcumsum_.resize(comm_size_ + 1);
  Nrows_xcumsum_.resize(comm_size_ + 1);

  std::vector<std::vector<double>> L; // lower triangular matrix of Cholesky decomposition
  std::vector<std::vector<double>> L_inv; // inverse of L

  static constexpr int BLEN = BSX_*BSY_;
  L.resize(BLEN);
  L_inv.resize(BLEN);
  for (int i(0); i<BLEN ; i++)
  {
    L[i].resize(i+1);
    L_inv[i].resize(i+1);
    // L_inv will act as right block in GJ algorithm, init as identity
    for (int j(0); j<=i; j++){
      L_inv[i][j] = (i == j) ? 1. : 0.;
    }
  }

  // compute the Cholesky decomposition of the preconditioner with Cholesky-Crout
  for (int i(0); i<BLEN ; i++)
  {
    double s1 = 0;
    for (int k(0); k<=i-1; k++)
      s1 += L[i][k]*L[i][k];
    L[i][i] = sqrt(getA_local(i,i) - s1);
    for (int j(i+1); j<BLEN; j++)
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
  for (int br(0); br<BLEN; br++)
  { // 'br' - base row in which all columns up to L_lb[br][br] are already zero
    const double bsf = 1. / L[br][br];
    for (int c(0); c<=br; c++)
      L_inv[br][c] *= bsf;

    for (int wr(br+1); wr<BLEN; wr++)
    { // 'wr' - working row where elements below L_lb[br][br] will be set to zero
      const double wsf = L[wr][br];
      for (int c(0); c<=br; c++)
        L_inv[wr][c] -= (wsf * L_inv[br][c]);
    }
  }

  // P_inv_ holds inverse preconditionner in row major order!
  std::vector<double> P_inv(BLEN * BLEN);
  for (int i(0); i<BLEN; i++)
  for (int j(0); j<BLEN; j++)
  {
    double aux = 0.;
    for (int k(0); k<BLEN; k++) // P_inv_ = (L^T)^{-1} L^{-1}
      aux += (i <= k && j <=k) ? L_inv[k][i] * L_inv[k][j] : 0.;

    P_inv[i*BLEN+j] = -aux; // Up to now Cholesky of negative P to avoid complex numbers
  }

  // Create Linear system and backend solver objects
  LocalLS_ = std::make_unique<LocalSpMatDnVec>(m_comm_, BSX_*BSY_, P_inv);
}

// Methods for cell centric construction of discrete Laplace operator
template<class EdgeIndexer >
void ExpAMRSolver::makeFlux(
  const BlockInfo &rhs_info,
  const int &ix,
  const int &iy,
  const BlockInfo &rhsNei,
  const EdgeIndexer &indexer,
  SpRowInfo &row) const
{
  const long long sfc_idx = indexer.This(rhs_info, ix, iy);

  if (this->sim.tmp->Tree(rhsNei).Exists())
  { 
    const int nei_rank = sim.tmp->Tree(rhsNei).rank();
    const long long nei_idx = indexer.neiblock_n(rhsNei, ix, iy);

    // Map flux associated to out-of-block edges at the same level of refinement
    row.mapColVal(nei_rank, nei_idx, 1.);
    row.mapColVal(sfc_idx, -1.);
  }
  else if (this->sim.tmp->Tree(rhsNei).CheckCoarser())
  {
    const BlockInfo &rhsNei_c = this->sim.tmp->getBlockInfoAll(rhs_info.level - 1 , rhsNei.Zparent);



    const int ix_c = indexer.ix_c(rhs_info, ix, iy);
    const int iy_c = indexer.iy_c(rhs_info, ix, iy);

    // Perform intepolation to calculate flux at interface with coarse cell
    PolyO3I f(sim, rhsNei_c, ix_c, iy_c, rhs_info, ix, iy, true, indexer, row);
  }
  else if (this->sim.tmp->Tree(rhsNei).CheckFiner())
  {
    const BlockInfo &rhsNei_f = this->sim.tmp->getBlockInfoAll(rhs_info.level + 1, indexer.Zchild(rhsNei, ix, iy));

    const int ix_f = (ix % (BSX_/2)) * 2;
    const int iy_f = (iy % (BSY_/2)) * 2;

    // Interpolate flux at interfaces with both fine neighbour cells
    PolyO3I f1(sim, rhs_info, ix, iy, rhsNei_f, ix_f, iy_f, false, indexer, row);
    PolyO3I f2(sim, rhs_info, ix, iy, rhsNei_f, ix_f+1, iy_f+1, false, indexer, row); // same coarse indices here...
  }
  else { throw std::runtime_error("Neighbour doesn't exist, isn't coarser, nor finer..."); }
}

template<class EdgeIndexer>
void ExpAMRSolver::makeEdgeCellRow( // excluding corners
    const BlockInfo &rhs_info,
    const int &ix,
    const int &iy,
    const bool &isBoundary,
    const BlockInfo &rhsNei,
    const EdgeIndexer &indexer)
{
    const long long sfc_idx = indexer.This(rhs_info, ix, iy);
    const long long n1_idx = indexer.inblock_n1(rhs_info, ix, iy); // in-block neighbour 1
    const long long n2_idx = indexer.inblock_n2(rhs_info, ix, iy); // in-block neighbour 2
    const long long n3_idx = indexer.inblock_n3(rhs_info, ix, iy); // in-block neighbour 3

    SpRowInfo row(sim.tmp->Tree(rhs_info).rank(), sfc_idx, 4); // worst case: this coarse with four fine out-of-rank nei

    // Map fluxes associated to in-block edges
    row.mapColVal(n1_idx, 1.);
    row.mapColVal(n2_idx, 1.);
    row.mapColVal(n3_idx, 1.);
    row.mapColVal(sfc_idx, -3.);

    if (!isBoundary)
      this->makeFlux(rhs_info, ix, iy, rhsNei, indexer, row);

    LocalLS_->cooPushBackRow(row);
}

template<class EdgeIndexer1, class EdgeIndexer2>
void ExpAMRSolver::makeCornerCellRow(
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
    const long long sfc_idx = indexer1.This(rhs_info, ix, iy);
    const long long n1_idx = indexer1.inblock_n2(rhs_info, ix, iy); // indexer.inblock_n2 is the opposite edge
    const long long n2_idx = indexer2.inblock_n2(rhs_info, ix, iy); // makes corner input order invariant

    SpRowInfo row(sim.tmp->Tree(rhs_info).rank(), sfc_idx, 8); // worst case: this coarse with four fine out-of-rank nei at both corner edges

    // Map fluxes associated to in-block edges
    row.mapColVal(n1_idx, 1.);
    row.mapColVal(n2_idx, 1.);
    row.mapColVal(sfc_idx, -2.);

    if (!isBoundary1)
      this->makeFlux(rhs_info, ix, iy, rhsNei_1, indexer1, row);
    if (!isBoundary2)
      this->makeFlux(rhs_info, ix, iy, rhsNei_2, indexer2, row);

    LocalLS_->cooPushBackRow(row);
}

void ExpAMRSolver::getMat()
{
  sim.startProfiler("Poisson solver: LS");

  //This returns an array with the blocks that the coarsest possible 
  //mesh would have (i.e. all blocks are at level 0)
  std::array<int, 3> blocksPerDim = sim.pres->getMaxBlocks();

  //Get a vector of all BlockInfos of the grid we're interested in
  sim.tmp->UpdateBlockInfoAll_States(true); // update blockID's for blocks from other ranks
  std::vector<cubism::BlockInfo>&  RhsInfo = sim.tmp->getBlocksInfo();
  const int Nblocks = RhsInfo.size();
  const int N = BSX_*BSY_*Nblocks;

  // Reserve sufficient memory for LS proper to the rank
  LocalLS_->reserve(N);

  // Calculate cumulative sums for blocks and rows for correct global indexing
  const long long Nblocks_long = Nblocks;
  MPI_Allgather(&Nblocks_long, 1, MPI_LONG_LONG, Nblocks_xcumsum_.data(), 1, MPI_LONG_LONG, m_comm_);
  for (int i(Nblocks_xcumsum_.size()-1); i > 0; i--)
  {
    Nblocks_xcumsum_[i] = Nblocks_xcumsum_[i-1]; // shift to right for rank 'i+1' to have cumsum of rank 'i'
  }
  
  // Set cumsum for rank 0 to zero
  Nblocks_xcumsum_[0] = 0;
  Nrows_xcumsum_[0] = 0;

  // Perform cumulative sum
  static constexpr long long BLEN = BSX_*BSY_;
  for (size_t i(1); i < Nblocks_xcumsum_.size(); i++)
  {
    Nblocks_xcumsum_[i] += Nblocks_xcumsum_[i-1];
    Nrows_xcumsum_[i] = BLEN*Nblocks_xcumsum_[i];
  }

  // No parallel for to ensure COO are ordered at construction
  for(int i=0; i<Nblocks; i++)
  {    
    const BlockInfo &rhs_info = RhsInfo[i];

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
        const long long sn_idx = NorthCell.SouthNeighbour(rhs_info, ix, iy); // indexer type irrelevant here
        const long long wn_idx = NorthCell.WestNeighbour(rhs_info, ix, iy);
        const long long sfc_idx = NorthCell.This(rhs_info, ix, iy);
        const long long en_idx = NorthCell.EastNeighbour(rhs_info, ix, iy);
        const long long nn_idx = NorthCell.NorthNeighbour(rhs_info, ix, iy);
        
        // Push back in ascending order for 'col_idx'
        LocalLS_->cooPushBackVal(1., sfc_idx, sn_idx);
        LocalLS_->cooPushBackVal(1., sfc_idx, wn_idx);
        LocalLS_->cooPushBackVal(-4, sfc_idx, sfc_idx);
        LocalLS_->cooPushBackVal(1., sfc_idx, en_idx);
        LocalLS_->cooPushBackVal(1., sfc_idx, nn_idx);
      }
      else if (ix == 0 && (iy > 0 && iy < BSY_-1))
      { // west cells excluding corners
        this->makeEdgeCellRow(rhs_info, ix, iy, isWestBoundary, rhsNei_west, WestCell);
      }
      else if (ix == BSX_-1 && (iy > 0 && iy < BSY_-1))
      { // east cells excluding corners
        this->makeEdgeCellRow(rhs_info, ix, iy, isEastBoundary, rhsNei_east, EastCell);
      }
      else if ((ix > 0 && ix < BSX_-1) && iy == 0)
      { // south cells excluding corners
        this->makeEdgeCellRow(rhs_info, ix, iy, isSouthBoundary, rhsNei_south, SouthCell);
      }
      else if ((ix > 0 && ix < BSX_-1) && iy == BSY_-1)
      { // north cells excluding corners
        this->makeEdgeCellRow(rhs_info, ix, iy, isNorthBoundary, rhsNei_north, NorthCell);
      }
      else if (ix == 0 && iy == 0)
      { // south-west corner
        this->makeCornerCellRow(
            rhs_info, ix, iy, 
            isSouthBoundary, rhsNei_south, SouthCell, 
            isWestBoundary, rhsNei_west, WestCell);
      }
      else if (ix == BSX_-1 && iy == 0)
      { // south-east corner
        this->makeCornerCellRow(
            rhs_info, ix, iy, 
            isEastBoundary, rhsNei_east, EastCell, 
            isSouthBoundary, rhsNei_south, SouthCell);
      }
      else if (ix == 0 && iy == BSY_-1)
      { // north-west corner
        this->makeCornerCellRow(
            rhs_info, ix, iy, 
            isWestBoundary, rhsNei_west, WestCell, 
            isNorthBoundary, rhsNei_north, NorthCell);
      }
      else if (ix == BSX_-1 && iy == BSY_-1)
      { // north-east corner
        this->makeCornerCellRow(
            rhs_info, ix, iy, 
            isNorthBoundary, rhsNei_north, NorthCell, 
            isEastBoundary, rhsNei_east, EastCell);
      }
    } // for(int iy=0; iy<BSY_; iy++) for(int ix=0; ix<BSX_; ix++)
  } // for(int i=0; i< Nblocks; i++)

  LocalLS_->make(Nrows_xcumsum_);

  sim.stopProfiler();
}

void ExpAMRSolver::getVec()
{
  //Get a vector of all BlockInfos of the grid we're interested in
  std::vector<cubism::BlockInfo>&  RhsInfo = sim.tmp->getBlocksInfo();
  std::vector<cubism::BlockInfo>&  zInfo = sim.pres->getBlocksInfo();
  const int Nblocks = RhsInfo.size();
  std::vector<double>& x = LocalLS_->get_x();
  std::vector<double>& b = LocalLS_->get_b();
  const long long shift = -Nrows_xcumsum_[rank_];

  // Copy RHS LHS vec initial guess, if LS was updated, updateMat reallocates sufficient memory
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
      const long long sfc_loc = NorthCell.This(rhs_info, ix, iy) + shift;
      b[sfc_loc] = rhs(ix,iy).s;
      x[sfc_loc] = p(ix,iy).s;
    }
  }
}

void ExpAMRSolver::solve(
    const ScalarGrid *input, 
    ScalarGrid * const output)
{

  if (rank_ == 0) {
    if (sim.verbose)
      std::cout << "--------------------- Calling on ExpAMRSolver.solve() ------------------------\n";
    else
      std::cout << '\n';
  }

  const double max_error = this->sim.step < 10 ? 0.0 : sim.PoissonTol;
  const double max_rel_error = this->sim.step < 10 ? 0.0 : sim.PoissonTolRel;
  const int max_restarts = this->sim.step < 10 ? 100 : sim.maxPoissonRestarts;

  if (sim.pres->UpdateFluxCorrection)
  {
    sim.pres->UpdateFluxCorrection = false;
    this->getMat();
    this->getVec();
    LocalLS_->solveWithUpdate(max_error, max_rel_error, max_restarts);
  }
  else
  {
    this->getVec();
    LocalLS_->solveNoUpdate(max_error, max_rel_error, max_restarts);
  }

  //Now that we found the solution, we just substract the mean to get a zero-mean solution. 
  //This can be done because the solver only cares about grad(P) = grad(P-mean(P))
  std::vector<cubism::BlockInfo>&  zInfo = sim.pres->getBlocksInfo();
  const int Nblocks = zInfo.size();
  const std::vector<double>& x = LocalLS_->get_x();

  double avg = 0;
  double avg1 = 0;
  #pragma omp parallel for reduction (+:avg,avg1)
  for(int i=0; i< Nblocks; i++)
  {
     ScalarBlock& P  = *(ScalarBlock*) zInfo[i].ptrBlock;
     const double vv = zInfo[i].h*zInfo[i].h;
     for(int iy=0; iy<BSY_; iy++)
     for(int ix=0; ix<BSX_; ix++)
     {
         P(ix,iy).s = x[i*BSX_*BSY_ + iy*BSX_ + ix];
         avg += P(ix,iy).s * vv;
         avg1 += vv;
     }
  }
  double quantities[2] = {avg,avg1};
  MPI_Allreduce(MPI_IN_PLACE, &quantities, 2, MPI_DOUBLE, MPI_SUM, m_comm_);
  avg = quantities[0]; avg1 = quantities[1] ;
  avg = avg/avg1;
  #pragma omp parallel for 
  for(int i=0; i< Nblocks; i++)
  {
     ScalarBlock& P  = *(ScalarBlock*) zInfo[i].ptrBlock;
     for(int iy=0; iy<BSY_; iy++)
     for(int ix=0; ix<BSX_; ix++)
        P(ix,iy).s += -avg;
  }
}

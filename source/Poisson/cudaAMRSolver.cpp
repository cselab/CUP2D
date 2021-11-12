//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "cudaAMRSolver.h"
#include "bicgstab.h"

using namespace cubism;

int WestNeighbourIdx(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy){ 
  return block_idx*BSX*BSY + iy*BSX + ix-1; 
}

int NorthNeighbourIdx(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy){ 
  return block_idx*BSX*BSY + (iy+1)*BSX + ix;
}

int EastNeighbourIdx(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy){ 
  return block_idx*BSX*BSY + iy*BSX + ix+1; 
}

int SouthNeighbourIdx(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy)
{ return block_idx*BSX*BSY + (iy-1)*BSX + ix; }

int WestmostCellIdx(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy)
{ return block_idx*BSX*BSY + iy*BSX + 0; }

int NorthmostCellIdx(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy)
{ return block_idx*BSX*BSY + (BSY-1)*BSX + ix; }

int EastmostCellIdx(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy)
{ return block_idx*BSX*BSY + iy*BSX + (BSX-1); }

int SouthmostCellIdx(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy)
{ return block_idx*BSX*BSY + 0*BSX + ix; }

long long get_Zchild(BlockInfo &info, const std::array<int,3> &Zchild_idx){
  return info.Zchild[Zchild_idx[0]][Zchild_idx[1]][Zchild_idx[2]];
}

enum Compass {North, East, South, West};

class NorthEdge{
  public:
    static int block_n1(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy)
    { return EastNeighbourIdx(block_idx, BSY, BSY, ix, iy); }
    static int block_n2(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy)
    { return SouthNeighbourIdx(block_idx, BSY, BSY, ix, iy); }
    static int block_n3(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy)
    { return WestNeighbourIdx(block_idx, BSY, BSY, ix, iy); }
    static int neiBlock_n(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy)
    { return SouthmostCellIdx(block_idx, BSY, BSY, ix, iy); }

    constexpr static Compass EdgeType = {North};
    constexpr static std::array<int,3> Zchild1_idx = {0,0,0};
    constexpr static std::array<int,3> Zchild2_idx = {1,0,0};
};

class EastEdge{
  public:
    static int block_n1(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy)
    { return SouthNeighbourIdx(block_idx, BSY, BSY, ix, iy); }
    static int block_n2(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy)
    { return WestNeighbourIdx(block_idx, BSY, BSY, ix, iy); }
    static int block_n3(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy)
    { return NorthNeighbourIdx(block_idx, BSY, BSY, ix, iy); }
    static int neiBlock_n(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy)
    { return WestmostCellIdx(block_idx, BSY, BSY, ix, iy); }

    constexpr static Compass EdgeType = {East};
    constexpr static std::array<int,3> Zchild1_idx = {0,0,0};
    constexpr static std::array<int,3> Zchild2_idx = {0,1,0};
};

class SouthEdge{
  public:
    static int block_n1(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy)
    { return WestNeighbourIdx(block_idx, BSY, BSY, ix, iy); }
    static int block_n2(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy)
    { return NorthNeighbourIdx(block_idx, BSY, BSY, ix, iy); }
    static int block_n3(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy)
    { return EastNeighbourIdx(block_idx, BSY, BSY, ix, iy); }
    static int neiBlock_n(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy)
    { return NorthmostCellIdx(block_idx, BSY, BSY, ix, iy); }

    constexpr static Compass EdgeType = {South};
    constexpr static std::array<int,3> Zchild1_idx = {0,1,0};
    constexpr static std::array<int,3> Zchild2_idx = {1,1,0};
};

class WestEdge{
  public:
    static int block_n1(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy)
    { return NorthNeighbourIdx(block_idx, BSY, BSY, ix, iy); }
    static int block_n2(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy)
    { return EastNeighbourIdx(block_idx, BSY, BSY, ix, iy); }
    static int block_n3(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy)
    { return SouthNeighbourIdx(block_idx, BSY, BSY, ix, iy); }
    static int neiBlock_n(const int &block_idx, const int &BSX, const int &BSY, const int &ix, const int &iy)
    { return EastmostCellIdx(block_idx, BSY, BSY, ix, iy); }

    constexpr static Compass EdgeType = {West};
    constexpr static std::array<int,3> Zchild1_idx = {1,0,0};
    constexpr static std::array<int,3> Zchild2_idx = {1,1,0};
};

double cudaAMRSolver::getA_local(int I1,int I2) //matrix for Poisson's equation on a uniform grid
{
   static constexpr int BSX = VectorBlock::sizeX;
   int j1 = I1 / BSX;
   int i1 = I1 % BSX;
   int j2 = I2 / BSX;
   int i2 = I2 % BSX;
   if (i1==i2 && j1==j2)
   {
     return 4.0;
   }
   else if (abs(i1-i2) + abs(j1-j2) == 1)
   {
     return -1.0;
   }
   else
   {
     return 0.0;
   }

}
cudaAMRSolver::cudaAMRSolver(SimulationData& s):sim(s)
{
  std::vector<std::vector<double>> L; // lower triangular matrix of Cholesky decomposition
  std::vector<std::vector<double>> L_inv; // inverse of L

  int BSX = VectorBlock::sizeX;
  int BSY = VectorBlock::sizeY;
  int N = BSX*BSY;
  L.resize(N);
  L_inv.resize(N);
  for (int i = 0 ; i<N ; i++)
  {
    L[i].resize(i+1);
    L_inv[i].resize(i+1);
    // L_inv will act as right block in GJ algorithm, init it as identity
    for (int j(0); j<=i; j++){
      L_inv[i][j] = (i == j) ? 1. : 0.;
    }
  }

  // compute the Cholesky decomposition of the preconditioner with Cholesky-Crout
  for (int i = 0 ; i<N ; i++)
  {
    double s1=0;
    for (int k=0; k<=i-1; k++)
      s1 += L[i][k]*L[i][k];
    L[i][i] = sqrt(getA_local(i,i) - s1);
    for (int j=i+1; j<N; j++)
    {
      double s2 = 0;
      for (int k=0; k<=i-1; k++)
        s2 += L[i][k]*L[j][k];
      L[j][i] = (getA_local(j,i)-s2) / L[i][i];
    }
  }

  /* Compute the inverse of the Cholesky decomposition L using Gauss-Jordan elimination.
     L will act as the left block (it does not need to be modified in the process), 
     L_inv will act as the right block and at the end of the algo will contain the inverse*/
  for (int br(0); br<N; br++)
    { // 'br' - base row in which all columns up to L_lb[br][br] are already zero
    const double bsf = 1. / L[br][br]; // scaling factor for base row
    for (int c(0); c<=br; c++)
    {
      L_inv[br][c] *= bsf;
    }

    for (int wr(br+1); wr<N; wr++)
    { // 'wr' - working row where elements below L_lb[br][br] will be set to zero
      const double wsf = L[wr][br];
      for (int c(0); c<=br; c++)
      { // For the right block matrix the trasformation has to be applied for the whole row
        L_inv[wr][c] -= (wsf * L_inv[br][c]);
      }
    }
  }

  // P_inv_ holds inverse preconditionner in row major order!  This is leads to better memory access
  // in the kernel that applies this preconditioner, but note that cuBLAS assumes column major
  // matrices by default
  P_inv_.resize(N * N); // use linear indexing for this matrix
  for (int i(0); i<N; i++){
    for (int j(0); j<N; j++){
      double aux = 0.;
      for (int k(0); k<N; k++){
        aux += (i <= k && j <=k) ? L_inv[k][i] * L_inv[k][j] : 0.; // P_inv_ = (L^T)^{-1} L^{-1}
      }
      P_inv_[i*N+j] = aux;
    }
  }
}

cudaAMRSolver::~cudaAMRSolver()
{
  std::cout << "---------------- Calling on cudaAMRSolver() destructor ------------\n";
}

void cudaAMRSolver::cooMatPushBack(
    const double &val, 
    const int &row, 
    const int &col){
  this->cooValA_.push_back(val);
  this->cooRowA_.push_back(row);
  this->cooColA_.push_back(col);
}

template<class EdgeHelper >
void cudaAMRSolver::neiBlockElement(
  BlockInfo &rhs_info,
  const int &BSX,
  const int &BSY,
  const int &ix,
  const int &iy,
  double &diag_val,
  BlockInfo &rhsNei,
  EdgeHelper helper)
{
  const int block_idx = rhs_info.blockID;
  const int sfc_idx = block_idx*BSX*BSY + iy*BSX + ix;

  if (this->sim.tmp->Tree(rhsNei).Exists())
  { //then out-of-block neighbour exists and we can safely use rhsNei and access the gridpoint-data etc.
    const int n_block_idx = rhsNei.blockID;
    const int n_idx = helper.neiBlock_n(n_block_idx, BSX, BSY, ix, iy);
    this->cooMatPushBack(1., sfc_idx, n_idx);

    // Flux contribution to diagonal value in case of uniorm grid
    diag_val--;
  }
  else if (this->sim.tmp->Tree(rhsNei).CheckCoarser())
  {
    BlockInfo &rhsNei_c = this->sim.tmp->getBlockInfoAll(rhs_info.level - 1 ,rhsNei.Zparent );
    if (this->sim.tmp->Tree(rhsNei_c).Exists())
    {
      /*
              -----------------------------
              |      |      |      .      |
              |      | f_2  | c_12 . c_22 |
              |      |      |      .      |
              |______|______|. . . . . . .|
              |      |      |      .      |
              |      | f_1  | c_11 . c_21 |
              |      |      |      .      |
              -----------------------------
                   fine         coarse
        
        Note from diagram that the location of fine cell with respect to its coarse
        neighbour affects which interpolation it gets.  Hence, logic should consider
        whether the current fine cell that we're constructing matrix row for is
        even/odd, and western/northern/eastern/southern with respect to coarse neigbour 
      */
      // Determine the orientation of the finer rhs_info wrt to the coarse neighbour
      int ix_c = ix / 2;
      int iy_c = iy / 2;
      if (helper.EdgeType == East || helper.EdgeType == West)
      { // Adding LS columns associated to flux from Western/Eastern boundary
        if (rhs_info.index[1] % 2 == 1) iy_c += (BSY / 2);
      }
      else if (helper.EdgeType == North || helper.EdgeType == South)
      { // Adding LS columns associated to flux from Northern/Southern boundary 
        if (rhs_info.index[0] % 2 == 1) ix_c += (BSX / 2);
      }
      else { abort(); } // Something went wrong

      const int nc_block_idx = rhsNei_c.blockID;
      const int nc_idx = helper.neiBlock_n(nc_block_idx, BSX, BSY, ix_c, iy_c);
      // At the moment interpolation c_11 = c_21 = c_12 = c_22 = p_{nc_idx}
      this->cooMatPushBack(1., sfc_idx, nc_idx);
      // If interpolation turns out to be depenant on second edge (corners), may need to use
      // a hash map with 'col' as index to accumulate contributions

      // With current interpolation flux contribution to diagonal element is unaffected for finer cell
      diag_val--;

    }
    else { abort(); }
  }
  else if (this->sim.tmp->Tree(rhsNei).CheckFiner())
  {
    // It is assumed that input agruments 'Zchild_1' and 'Zchild_2' respect this order:
    //Zchild[0][0][0] is (2i  ,2j  ,2k  )
    //Zchild[1][0][0] is (2i+1,2j  ,2k  )
    //Zchild[0][1][0] is (2i  ,2j+1,2k  )
    //Zchild[1][1][0] is (2i+1,2j+1,2k  )

    // Determine which fine block the current coarse edge neighbours
    long long rhsNei_Zchild;
    if (helper.EdgeType == East || helper.EdgeType == West)
    { // Adding LS columns associated to flux from Western/Eastern boundary
      if (iy < BSY / 2)
        rhsNei_Zchild = get_Zchild(rhsNei, helper.Zchild1_idx);
      else
        rhsNei_Zchild = get_Zchild(rhsNei, helper.Zchild2_idx);
    }
    else if (helper.EdgeType == North || helper.EdgeType == South)
    { // Adding LS columns associated to flux from Northern/Southern boundary 
      if (ix < BSX / 2)
        rhsNei_Zchild = get_Zchild(rhsNei, helper.Zchild1_idx);
      else
        rhsNei_Zchild = get_Zchild(rhsNei, helper.Zchild2_idx);
    }
    else { abort(); } // Something went wrong

    BlockInfo &rhsNei_f = this->sim.tmp->getBlockInfoAll(rhs_info.level + 1 , rhsNei_Zchild);
    if (this->sim.tmp->Tree(rhsNei_f).Exists())
    { // Extract indices for two fine neighbour edges.  Depending on the boundary, either 
      // 'ix_f' or 'iy_f' will be completely wrong, but n_func ignores that one
      const int ix_f = (ix % (BSX/2)) * 2;
      const int iy_f = (iy % (BSY/2)) * 2;
      // Two fine neighbours
      const int nf_block_idx = rhsNei_f.blockID;   
      const int nf1_idx = helper.neiBlock_n(nf_block_idx, BSX, BSY, ix_f, iy_f);
      this->cooMatPushBack(1., sfc_idx, nf1_idx);

      const int nf2_idx = helper.neiBlock_n(nf_block_idx, BSX, BSY, ix_f+1, iy_f+1);
      this->cooMatPushBack(1., sfc_idx, nf2_idx);

      // For now with simple interpolation c_11 = c_21 = c_12 = c_22 = p_{ij}
      // two fluxes ==> two diagonal contributions
      diag_val -= 2.;
    }
    else { abort(); }
  }
  else { abort(); }
}

template<class EdgeHelper>
void cudaAMRSolver::edgeBoundaryCell( // excluding corners
    BlockInfo &rhs_info,
    const int &BSX,
    const int &BSY,
    const int &ix,
    const int &iy,
    const bool &isBoundary,
    BlockInfo &rhsNei,
    EdgeHelper helper)
{
    const int block_idx = rhs_info.blockID;
    const int sfc_idx = block_idx*BSX*BSY + iy*BSX + ix;
    const int n1_idx = helper.block_n1(block_idx, BSX, BSY, ix, iy); // in-block neighbour 1
    const int n2_idx = helper.block_n2(block_idx, BSX, BSY, ix, iy); // in-block neighbour 2
    const int n3_idx = helper.block_n3(block_idx, BSX, BSY, ix, iy); // in-block neighbour 3

    // Add matrix element associated to in-block neighbours
    this->cooMatPushBack(1., sfc_idx, n1_idx);
    this->cooMatPushBack(1., sfc_idx, n2_idx);
    this->cooMatPushBack(1., sfc_idx, n3_idx);

    if (isBoundary)
    { // Adapt diagonal element to satisfy one Neumann BC
      this->cooMatPushBack(-3., sfc_idx, sfc_idx);
    }
    else
    {
      double diag_val = -3.;
      this->neiBlockElement(
          rhs_info, 
          BSX, 
          BSY, 
          ix, 
          iy, 
          diag_val, 
          rhsNei, 
          helper);
      // Adapt diagonal element to account for contributions from coarser/finer mesh
      this->cooMatPushBack(diag_val, sfc_idx, sfc_idx);
    }
}

template<class EdgeHelper1, class EdgeHelper2>
void cudaAMRSolver::cornerBoundaryCell(
    BlockInfo &rhs_info,
    const int &BSX,
    const int &BSY,
    const int &ix,
    const int &iy,
    const bool &isBoundary1,
    BlockInfo &rhsNei_1,
    EdgeHelper1 helper1, 
    const bool &isBoundary2,
    BlockInfo &rhsNei_2,
    EdgeHelper2 helper2)
{
    const int block_idx = rhs_info.blockID;    
    const int sfc_idx = block_idx*BSX*BSY + iy*BSX + ix;
    const int n1_idx = helper1.block_n2(block_idx, BSX, BSY, ix, iy); // helper.block_n1 is the other corner edge
    const int n2_idx = helper1.block_n3(block_idx, BSX, BSY, ix, iy); // 

    // Add matrix element associated to in-block neighbours
    this->cooMatPushBack(1., sfc_idx, n1_idx);
    this->cooMatPushBack(1., sfc_idx, n2_idx);

    if (isBoundary1 && isBoundary2)
    { // Adapt diagonal element to satisfy two Neumann BC
      this->cooMatPushBack(-2., sfc_idx, sfc_idx);
    }
    else 
    {
      double diag_val = -2.;
      if (!isBoundary1)
      { // Add matrix element associated to out-of-block neighbour 3
        this->neiBlockElement(
            rhs_info, 
            BSX, 
            BSY, 
            ix, 
            iy, 
            diag_val, 
            rhsNei_1, 
            helper1);
      }
      if (!isBoundary2)
      { // Add matrix element associated to out-of-block neighbour 4
        this->neiBlockElement(
            rhs_info, 
            BSX, 
            BSY, 
            ix, 
            iy, 
            diag_val, 
            rhsNei_2, 
            helper2);
      }
      // Adapt diagonal element to account for contributions from coarser/finer mesh
      this->cooMatPushBack(diag_val, sfc_idx, sfc_idx);
    }
}

void cudaAMRSolver::Get_LS()
{
  sim.startProfiler("Poisson solver: LS");

  static constexpr int BSX = VectorBlock::sizeX;
  static constexpr int BSY = VectorBlock::sizeY;

  //This returns an array with the blocks that the coarsest possible 
  //mesh would have (i.e. all blocks are at level 0)
  std::array<int, 3> blocksPerDim = sim.pres->getMaxBlocks();

  //Get a vector of all BlockInfos of the grid we're interested in
  std::vector<cubism::BlockInfo>&  RhsInfo = sim.tmp->getBlocksInfo();
  std::vector<cubism::BlockInfo>&  zInfo = sim.pres->getBlocksInfo();
  const int Nblocks = RhsInfo.size();
  const int N = BSX*BSY*Nblocks;

  // Allocate memory for solution 'x' and RHS vector 'b' on host
  this->x_.resize(N);
  this->b_.resize(N);
  // Clear contents from previous call of cudaAMRSolver::solve() and reserve memory 
  // for sparse LHS matrix 'A' (for uniform grid at most 5 elements per row).
  this->cooValA_.clear();
  this->cooRowA_.clear();
  this->cooColA_.clear();
  this->cooValA_.reserve(5 * N);
  this->cooRowA_.reserve(5 * N);
  this->cooColA_.reserve(5 * N);

  // No 'parallel for' to avoid accidental reorderings of COO elements during push_back
  // adding a critical section to push_back makes things worse as threads fight for access
  for(int i=0; i< Nblocks; i++)
  {    
    BlockInfo &rhs_info = RhsInfo[i];
    ScalarBlock & __restrict__ rhs  = *(ScalarBlock*) RhsInfo[i].ptrBlock;
    ScalarBlock & __restrict__ p  = *(ScalarBlock*) zInfo[i].ptrBlock;

    // Construct RHS and x_0 vectors for linear system
    for(int iy=0; iy<BSY; iy++)
    for(int ix=0; ix<BSX; ix++)
    {
      const int sfc_idx = i*BSX*BSY+iy*BSX+ix;
      //b_[sfc_idx] = 0.;
      //x_[sfc_idx] = 5.;
      b_[sfc_idx] = rhs(ix,iy).s;
      x_[sfc_idx] = p(ix,iy).s;
    }

    //1.Check if this is a boundary block
    int aux = 1 << rhs_info.level; // = 2^level
    int MAX_X_BLOCKS = (blocksPerDim[0] - 1)*aux; //this means that if level 0 has blocksPerDim[0] blocks in the x-direction, level rhs.level will have this many blocks
    int MAX_Y_BLOCKS = (blocksPerDim[1] - 1)*aux; //this means that if level 0 has blocksPerDim[1] blocks in the y-direction, level rhs.level will have this many blocks

    //index is the (i,j) coordinates of a block at the current level 
    const bool isWestBoundary  = (rhs_info.index[0] == 0           ); // don't check for west neighbor
    const bool isEastBoundary  = (rhs_info.index[0] == MAX_X_BLOCKS); // don't check for east neighbor
    const bool isSouthBoundary = (rhs_info.index[1] == 0           ); // don't check for south neighbor
    const bool isNorthBoundary = (rhs_info.index[1] == MAX_Y_BLOCKS); // don't check for north neighbor

    //2.Access the block's neighbors (for the Poisson solve in two dimensions we care about four neighbors in total)
    long long Z_west  = rhs_info.Znei[1-1][1][1];
    long long Z_east  = rhs_info.Znei[1+1][1][1];
    long long Z_south = rhs_info.Znei[1][1-1][1];
    long long Z_north = rhs_info.Znei[1][1+1][1];
    //rhs.Z == rhs.Znei[1][1][1] is true always

    BlockInfo &rhsNei_west  = this->sim.tmp->getBlockInfoAll(rhs_info.level,Z_west );
    BlockInfo &rhsNei_east  = this->sim.tmp->getBlockInfoAll(rhs_info.level,Z_east );
    BlockInfo &rhsNei_south = this->sim.tmp->getBlockInfoAll(rhs_info.level,Z_south);
    BlockInfo &rhsNei_north = this->sim.tmp->getBlockInfoAll(rhs_info.level,Z_north);

    //For later: there's a total of three boolean variables:
    // I.   grid->Tree(rhsNei_west).Exists()
    // II.  grid->Tree(rhsNei_west).CheckCoarser()
    // III. grid->Tree(rhsNei_west).CheckFiner()
    // And only one of them is true

    // Add matrix elements associated to interior cells of a block
    for(int iy=1; iy<BSY-1; iy++)
    for(int ix=1; ix<BSX-1; ix++)
    {
      const int sfc_idx = i*BSX*BSY+iy*BSX+ix;
      const int wn_idx = sfc_idx-1; // west neighbour
      const int en_idx = sfc_idx+1; // east neighbour
      const int sn_idx = i*BSX*BSY+(iy-1)*BSX+ix; // south neighbour 
      const int nn_idx = i*BSX*BSY+(iy+1)*BSX+ix; // north neighbour
      
      // Add diagonal matrix element
      this->cooMatPushBack(-4, sfc_idx, sfc_idx);
      // Add matrix element associated to west cell
      this->cooMatPushBack(1., sfc_idx, wn_idx);
      // Add matrix element associated to east cell
      this->cooMatPushBack(1., sfc_idx, en_idx);
      // Add matrix element associated to south cell
      this->cooMatPushBack(1., sfc_idx, sn_idx);
      // Add matrix element associated to north cell
      this->cooMatPushBack(1., sfc_idx, nn_idx);
    }

    for(int iy=1; iy<BSY-1; iy++)
    {
      // Add matrix elements associated to cells on the western boundary of the block (excl. corners)
      int ix = 0;
      this->edgeBoundaryCell(
          rhs_info, 
          BSX, 
          BSY, 
          ix, 
          iy, 
          isWestBoundary,
          rhsNei_west,
          WestEdge());

      // Add matrix elements associated to cells on the eastern boundary of the block (excl. corners)
      ix = BSX-1;
      this->edgeBoundaryCell(
          rhs_info,
          BSX, 
          BSY,
          ix,
          iy,
          isEastBoundary,
          rhsNei_east,
          EastEdge());
    }

    for(int ix=1; ix<BSX-1; ix++)
    {
      // Add matrix elements associated to cells on the northern boundary of the block (excl. corners)
      int iy = BSY-1;
      this->edgeBoundaryCell(
          rhs_info, 
          BSX, 
          BSY, 
          ix, 
          iy, 
          isNorthBoundary,
          rhsNei_north,
          NorthEdge());

      // Add matrix elements associated to cells on the southern boundary of the block (excl. corners)
      iy = 0;
      this->edgeBoundaryCell(
          rhs_info,
          BSX,
          BSY,
          ix,
          iy,
          isSouthBoundary,
          rhsNei_south,
          SouthEdge());
    }
    {
      // Add matrix elements associated to cells on the north-west corner of the block (excl. corners)
      int ix = 0;
      int iy = BSY-1;
      this->cornerBoundaryCell(
          rhs_info,
          BSX,
          BSY,
          ix,
          iy,
          isWestBoundary,
          rhsNei_west,
          WestEdge(),
          isNorthBoundary,
          rhsNei_north,
          NorthEdge());

      // Add matrix elements associated to cells on the north-east corner of the block (excl. corners)
      ix = BSX-1;
      iy = BSY-1;
      this->cornerBoundaryCell(
          rhs_info,
          BSX,
          BSY,
          ix,
          iy,
          isNorthBoundary,
          rhsNei_north,
          NorthEdge(),
          isEastBoundary,
          rhsNei_east,
          EastEdge());
      
      // Add matrix elements associated to cells on the south-east corner of the block (excl. corners)
      ix = BSX-1;
      iy = 0;
      this->cornerBoundaryCell(
          rhs_info,
          BSX,
          BSY,
          ix,
          iy,
          isEastBoundary,
          rhsNei_east,
          EastEdge(),
          isSouthBoundary,
          rhsNei_south,
          SouthEdge());

      // Add matrix elements associated to cells on the south-west corner of the block (excl. corners)
      ix = 0;
      iy = 0;
      this->cornerBoundaryCell(
          rhs_info,
          BSX,
          BSY,
          ix,
          iy,
          isSouthBoundary,
          rhsNei_south,
          SouthEdge(),
          isWestBoundary,
          rhsNei_west,
          WestEdge());
    }
  }
  // Save params of current linear system
  m_ = N; // rows
  n_ = N; // cols
  nnz_ = this->cooValA_.size(); // non-zero elements
  std::cout << "Rows: " << m_  
            << " cols: " << n_ 
            << " non-zero elements: " << nnz_ << std::endl;

  sim.stopProfiler();
}

void cudaAMRSolver::solve()
{

  std::cout << "--------------------- Calling on cudaAMRSolver.solve() ------------------------ \n";

  this->Get_LS();

  static constexpr int BSX = VectorBlock::sizeX;
  static constexpr int BSY = VectorBlock::sizeY;
  std::vector<cubism::BlockInfo>&  zInfo = sim.pres->getBlocksInfo();
  const int Nblocks = zInfo.size();

  const double max_error = sim.step < 10 ? 0.0 : sim.PoissonTol * sim.uMax_measured / sim.dt;
  const double max_rel_error = sim.step < 10 ? 0.0 : min(1e-2,sim.PoissonTolRel * sim.uMax_measured / sim.dt );
  const int max_restarts = sim.step < 10 ? 100 : sim.maxPoissonRestarts;

//  BiCGSTAB(
//      m_, 
//      n_, 
//      nnz_, 
//      cooValA_.data(), 
//      cooRowA_.data(), 
//      cooColA_.data(), 
//      x_.data(), 
//      b_.data(), 
//      max_error, 
//      max_rel_error,
//      max_restarts);
   pBiCGSTAB(
      m_, 
      n_, 
      nnz_, 
      cooValA_.data(), 
      cooRowA_.data(), 
      cooColA_.data(), 
      x_.data(), 
      b_.data(), 
      BSX * BSY,
      P_inv_.data(),
      max_error, 
      max_rel_error,
      max_restarts);

  //Now that we found the solution, we just substract the mean to get a zero-mean solution. 
  //This can be done because the solver only cares about grad(P) = grad(P-mean(P))
  double avg = 0;
  double avg1 = 0;
  #pragma omp parallel
  {
     #pragma omp for reduction (+:avg,avg1)
     for(int i=0; i< Nblocks; i++)
     {
        ScalarBlock& P  = *(ScalarBlock*) zInfo[i].ptrBlock;
        const double vv = zInfo[i].h*zInfo[i].h;
        for(int iy=0; iy<VectorBlock::sizeY; iy++)
        for(int ix=0; ix<VectorBlock::sizeX; ix++)
        {
            P(ix,iy).s = x_[i*BSX*BSY + iy*BSX + ix];
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
        for(int iy=0; iy<VectorBlock::sizeY; iy++)
        for(int ix=0; ix<VectorBlock::sizeX; ix++)
           P(ix,iy).s -= avg;
     }
  }
}

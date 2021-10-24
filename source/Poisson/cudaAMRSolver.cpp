//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "cudaAMRSolver.h"
#include "bicgstab.h"

using namespace cubism;

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

void inline cudaAMRSolver::cooMatPushBack(
    const double &val, 
    const int &row, 
    const int &col){
  this->cooValA_.push_back(val);
  this->cooRowA_.push_back(row);
  this->cooColA_.push_back(col);
}

// Prepare linear system for uniform grid
void cudaAMRSolver::Get_LS()
{
  std::cout << "[cudaAMRSolver]: Preparing linear system... \n";
  sim.startProfiler("Poisson solver: unifLinsysPrepHost()");

  static constexpr int BSX = VectorBlock::sizeX;
  static constexpr int BSY = VectorBlock::sizeY;

  // Extract number of blocks and cells on grid
  //This returns an array with the blocks that the coarsest possible 
  //mesh would have (i.e. all blocks are at level 0)
  std::array<int, 3> blocksPerDim = sim.pres->getMaxBlocks();

  //Get a vector of all BlockInfos of the grid we're interested in
  std::vector<cubism::BlockInfo>&  RhsInfo = sim.tmp->getBlocksInfo();
  std::vector<cubism::BlockInfo>&  zInfo = sim.pres->getBlocksInfo();
  const size_t Nblocks = RhsInfo.size();
  const size_t N = BSX*BSY*Nblocks;

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

  size_t double_bd = 0;
  size_t north_bd = 0;
  size_t south_bd = 0;
  size_t east_bd = 0;
  size_t west_bd = 0;

  // No 'parallel for' to avoid accidental reorderings of COO elements during push_back
  for(size_t i=0; i< Nblocks; i++)
  {    
    BlockInfo &rhs_info = RhsInfo[i];
    ScalarBlock & __restrict__ rhs  = *(ScalarBlock*) RhsInfo[i].ptrBlock;
    ScalarBlock & __restrict__ p  = *(ScalarBlock*) zInfo[i].ptrBlock;

    // Construct RHS and x_0 vectors for linear system
    #pragma omp parallel for
    for(int iy=0; iy<BSY; iy++)
    for(int ix=0; ix<BSX; ix++)
    {
      const int sfc_idx = i*BSX*BSY+iy*BSX+ix;
      // b_[sfc_idx] = 0.;
      // x_[sfc_idx] = 1.;
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
    if (isWestBoundary) {}//do this
    if (isEastBoundary) {}//do that
    if (isSouthBoundary){}//do sth else
    if (isNorthBoundary){}//do this
    //keep in mind that a block might be on two boundaries (say west and north) at the same time!

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

    // Add matrix elements associated to contributions from north/south on western/eastern boundary of block (excl.corners)
    for(int iy=1; iy<BSY-1; iy++)
    for(int ix=0; ix<BSX; ix+=(BSX-1))
    { // The inner loop executes on ix = [0, BSX-1] (western/eastern boundary) where no interaction with
      // neighbouring blocks takes place
      const int sfc_idx = i*BSX*BSY+iy*BSX+ix;
      const int sn_idx = i*BSX*BSY+(iy-1)*BSX+ix; // south neighbour 
      const int nn_idx = i*BSX*BSY+(iy+1)*BSX+ix; // north neighbour
      // Add matrix element associated to south cell
      this->cooMatPushBack(1., sfc_idx, sn_idx);
      // Add matrix element associated to north cell
      this->cooMatPushBack(1., sfc_idx, nn_idx);
    }

    // Add matrix elements associated to contributions from east/west cells on western boundary of block (excl. corners)
    for(int iy=1; iy<BSY-1; iy++)
    {
      const int ix = 0;
      const int sfc_idx = i*BSX*BSY+iy*BSX+ix;
      const int en_idx = sfc_idx+1; // east neighbour

      // Add matrix element associated to east cell
      this->cooMatPushBack(1., sfc_idx, en_idx);

      if (isWestBoundary){
        this->cooMatPushBack(-3., sfc_idx, sfc_idx);
        west_bd++;
      }
      else{
        this->cooMatPushBack(-4., sfc_idx, sfc_idx);
        if (sim.tmp->Tree(rhsNei_west).Exists())
        {
          //then west neighbor exists and we can safely use rhsNei_west and access the gridpoint-data etc.
          const size_t i_west = rhsNei_west.blockID;
          const int wn_idx = i_west*BSX*BSY+iy*BSX+(BSX-1); // eastmost cell of western block
          this->cooMatPushBack(1., sfc_idx, wn_idx);
        }
        else { throw; }
      }
    }

    // Add matrix elements associated to contributions from east/west cells on eastern boundary of block (excl. corners)
    for(int iy=1; iy<BSY-1; iy++)
    {
      const int ix = BSX-1;
      const int sfc_idx = i*BSX*BSY+iy*BSX+ix;
      const int wn_idx = sfc_idx-1; // west neighbour

      // Add matrix element associated to west cell
      this->cooMatPushBack(1., sfc_idx, wn_idx);

      if (isEastBoundary){
        this->cooMatPushBack(-3., sfc_idx, sfc_idx);
        east_bd++;
      }
      else{
        this->cooMatPushBack(-4., sfc_idx, sfc_idx);
        if (sim.tmp->Tree(rhsNei_east).Exists())
        {
          //then west neighbor exists and we can safely use rhsNei_west and access the gridpoint-data etc.
          const size_t i_east = rhsNei_east.blockID;
          const int en_idx = i_east*BSX*BSY+iy*BSX+0; // westmost cell of eastern block
          this->cooMatPushBack(1., sfc_idx, en_idx);
        }
        else { throw; }
      }
    }

    // Add matrix elements associated to contributions from west/east on southern/northern boundary of block (with corners)
    for(int ix=0; ix<BSX; ix++)
    for(int iy=0; iy<BSY; iy+=(BSY-1))
    { // The inner loop executes on iy = {0, BSY-1} (southern/northern boundary), at ix = [0, BSX-1] interaction
      // with western/eastern block takes place
      const int sfc_idx = i*BSX*BSY+iy*BSX+ix;

      // Add matrix element associated to west cell
      if (ix > 0){
        const int wn_idx = sfc_idx-1; // west neighbour
        this->cooMatPushBack(1., sfc_idx, wn_idx);
      }
      else{
        if (!isWestBoundary){
          if (sim.tmp->Tree(rhsNei_west).Exists()){
            const size_t i_west = rhsNei_west.blockID;
            const int wn_idx = i_west*BSX*BSY+iy*BSX+(BSX-1); // eastmost cell of western block
            this->cooMatPushBack(1., sfc_idx, wn_idx);
          }
          else{ throw; }
        }
        // If western boundary, the diagonal element is modified when north/south contributions are considered
      }
      // Add matrix element associated to east cell
      if (ix < BSX - 1){
        const int en_idx = sfc_idx+1; // east neighbour
        this->cooMatPushBack(1., sfc_idx, en_idx);
      }
      else {
        if (!isEastBoundary){
          if (sim.tmp->Tree(rhsNei_east).Exists()){
            const size_t i_east = rhsNei_east.blockID;
            const int en_idx = i_east*BSX*BSY+iy*BSX+0; // westmost cell of eastern block
            this->cooMatPushBack(1., sfc_idx, en_idx);
          }
          else { throw; }
        }
        // If eastern boundary, the diagonal element is modified when north/south contributions are considered
      }
    }

    // Add matrix elements associated to contributions from north/south cells on southern boundary of block (with corners)
    for(int ix=0; ix<BSX;ix++)
    {
      const int iy = 0;
      const int sfc_idx = i*BSX*BSY+iy*BSX+ix;
      const int nn_idx = i*BSX*BSY+(iy+1)*BSX+ix; // north neighbour

      // Add matrix element associated to north cell
      this->cooMatPushBack(1., sfc_idx, nn_idx);

      if(isSouthBoundary){
        south_bd++;
        if((isWestBoundary && ix == 0) || (isEastBoundary && ix == (BSX-1)))
        { // Two boundary conditions to consider for diagonal element
          this->cooMatPushBack(-2., sfc_idx, sfc_idx);
          double_bd++;
          west_bd += isWestBoundary ? 1 : 0;
          east_bd += isEastBoundary ? 1 : 0;
        }
        else
        { // One boundary condition to consider for diagonal element
          this->cooMatPushBack(-3., sfc_idx, sfc_idx);
        }
      }
      else{
        if((isWestBoundary && ix == 0) || (isEastBoundary && ix == (BSX-1)))
        { // Could still be east/west boundary!
          this->cooMatPushBack(-3., sfc_idx, sfc_idx);
          west_bd += isWestBoundary ? 1 : 0;
          east_bd += isEastBoundary ? 1 : 0;
        }
        else
        { // Otherwise the diagonal element does not change
          this->cooMatPushBack(-4., sfc_idx, sfc_idx);
        }
        if (sim.tmp->Tree(rhsNei_south).Exists())
        {
          //then west neighbor exists and we can safely use rhsNei_west and access the gridpoint-data etc.
          const size_t i_south = rhsNei_south.blockID;
          const int sn_idx = i_south*BSX*BSY+(BSY-1)*BSX+ix; // northmost cell of southern block
          this->cooMatPushBack(1., sfc_idx, sn_idx);
        }
        else { throw; }
      }
    }

    // Add matrix elements associated to contributions from north/south cells on northern boundary of block (with corners)
    for(int ix=0; ix<BSX;ix++)
    {
      const int iy = BSY-1;
      const int sfc_idx = i*BSX*BSY+iy*BSX+ix;
      const int sn_idx = i*BSX*BSY+(iy-1)*BSX+ix; // south neighbour

      // Add matrix element associated to south cell
      this->cooMatPushBack(1., sfc_idx, sn_idx);

      if (isNorthBoundary){
        north_bd++;
        if((isWestBoundary && ix == 0) || (isEastBoundary && ix == (BSX-1)))
        { // Two boundary conditions to consider for diagonal element
          this->cooMatPushBack(-2., sfc_idx, sfc_idx);
          double_bd++;
          west_bd += isWestBoundary ? 1 : 0;
          east_bd += isEastBoundary ? 1 : 0;
        }
        else
        { // One boundary condition to consider for diagonal element
          this->cooMatPushBack(-3., sfc_idx, sfc_idx);
        }
      }
      else{
        if((isWestBoundary && ix == 0) || (isEastBoundary && ix == (BSX-1)))
        { // Could still be east/west boundary!
          this->cooMatPushBack(-3., sfc_idx, sfc_idx);
          west_bd += isWestBoundary ? 1 : 0;
          east_bd += isEastBoundary ? 1 : 0;
        }
        else
        { // Otherwise the diagonal element does not change
          this->cooMatPushBack(-4., sfc_idx, sfc_idx);
        }
        if (sim.tmp->Tree(rhsNei_north).Exists())
        {
          //then west neighbor exists and we can safely use rhsNei_north and access the gridpoint-data etc.
          const size_t i_north = rhsNei_north.blockID;
          const int nn_idx = i_north*BSX*BSY+0*BSX+ix; // southmost cell belonging to north block
          this->cooMatPushBack(1., sfc_idx, nn_idx);
        }
        else { throw; }
      }
    }
  }
  std::cout << "Corner BC's: " << double_bd << std::endl;
  std::cout << "North BC's: " << north_bd << std::endl;
  std::cout << "East BC's: " << east_bd << std::endl;
  std::cout << "South BC's: " << south_bd << std::endl;
  std::cout << "West BC's: " << west_bd << std::endl;
  // Save params of current linear system
  m_ = N; // rows
  n_ = N; // cols
  nnz_ = this->cooValA_.size(); // non-zero elements

  sim.stopProfiler();
}

void cudaAMRSolver::solve()
{

  std::cout << "--------------------- Calling on cudaAMRSolver.solve() ------------------------ \n";

  this->Get_LS();

  const double max_error = sim.step < 10 ? 0.0 : sim.PoissonTol * sim.uMax_measured / sim.dt;
  const double max_rel_error = sim.step < 10 ? 0.0 : min(1e-2,sim.PoissonTolRel * sim.uMax_measured / sim.dt );
  const int max_restarts = sim.step < 10 ? 100 : sim.maxPoissonRestarts;

  BiCGSTAB(
      m_, 
      n_, 
      nnz_, 
      cooValA_.data(), 
      cooRowA_.data(), 
      cooColA_.data(), 
      x_.data(), 
      b_.data(), 
      max_error, 
      max_rel_error,
      max_restarts);

  //Now that we found the solution, we just substract the mean to get a zero-mean solution. 
  //This can be done because the solver only cares about grad(P) = grad(P-mean(P))
  static constexpr int BSX = VectorBlock::sizeX;
  static constexpr int BSY = VectorBlock::sizeY;
  std::vector<cubism::BlockInfo>&  zInfo = sim.pres->getBlocksInfo();
  const size_t Nblocks = zInfo.size();

  double avg = 0;
  double avg1 = 0;
  #pragma omp parallel
  {
     #pragma omp for reduction (+:avg,avg1)
     for(size_t i=0; i< Nblocks; i++)
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
     for(size_t i=0; i< Nblocks; i++)
     {
        ScalarBlock& P  = *(ScalarBlock*) zInfo[i].ptrBlock;
        for(int iy=0; iy<VectorBlock::sizeY; iy++)
        for(int ix=0; ix<VectorBlock::sizeX; ix++)
           P(ix,iy).s -= avg;
     }
  }
}

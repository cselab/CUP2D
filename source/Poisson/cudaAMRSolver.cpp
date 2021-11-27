//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "cudaAMRSolver.h"
#include "bicgstab.h"

using namespace cubism;

enum Compass {North, East, South, West};

class CellIndexer{
  public:
    static int This(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy)
    { return info.blockID*BSX*BSY + iy*BSX + ix; }
    static int WestNeighbour(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int dist = 1)
    { return info.blockID*BSX*BSY + iy*BSX + ix-dist; }
    static int NorthNeighbour(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int dist = 1)
    { return info.blockID*BSX*BSY + (iy+dist)*BSX + ix;}
    static int EastNeighbour(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int dist = 1)
    { return info.blockID*BSX*BSY + iy*BSX + ix+dist; }
    static int SouthNeighbour(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int dist = 1)
    { return info.blockID*BSX*BSY + (iy-dist)*BSX + ix; }

    static int WestmostCell(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int offset = 0)
    { return info.blockID*BSX*BSY + iy*BSX + offset; }
    static int NorthmostCell(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int offset = 0)
    { return info.blockID*BSX*BSY + (BSY-1-offset)*BSX + ix; }
    static int EastmostCell(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int offset = 0)
    { return info.blockID*BSX*BSY + iy*BSX + (BSX-1-offset); }
    static int SouthmostCell(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int offset = 0)
    { return info.blockID*BSX*BSY + offset*BSX + ix; }
};


class NorthEdgeCell : public CellIndexer{
  public:
    static int inblock_n1(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int dist = 1)
    { return EastNeighbour(info, BSY, BSY, ix, iy, dist); }
    static int inblock_n2(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int dist = 1)
    { return SouthNeighbour(info, BSY, BSY, ix, iy, dist); }
    static int inblock_n3(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int dist = 1)
    { return WestNeighbour(info, BSY, BSY, ix, iy, dist); }
    static int neiblock_n(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int offset = 0)
    { return SouthmostCell(info, BSY, BSY, ix, iy, offset); }

    constexpr static Compass EdgeType = {North};
    constexpr static std::array<int,3> Zchild1_idx = {0,0,0};
    constexpr static std::array<int,3> Zchild2_idx = {1,0,0};
};

class EastEdgeCell : public CellIndexer{
  public:
    static int inblock_n1(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int dist = 1)
    { return SouthNeighbour(info, BSY, BSY, ix, iy, dist); }
    static int inblock_n2(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int dist = 1)
    { return WestNeighbour(info, BSY, BSY, ix, iy, dist); }
    static int inblock_n3(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int dist = 1)
    { return NorthNeighbour(info, BSY, BSY, ix, iy, dist); }
    static int neiblock_n(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int offset = 0)
    { return WestmostCell(info, BSY, BSY, ix, iy, offset); }

    constexpr static Compass EdgeType = {East};
    constexpr static std::array<int,3> Zchild1_idx = {0,0,0};
    constexpr static std::array<int,3> Zchild2_idx = {0,1,0};
};

class SouthEdgeCell : public CellIndexer{
  public:
    static int inblock_n1(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int dist = 1)
    { return WestNeighbour(info, BSY, BSY, ix, iy, dist); }
    static int inblock_n2(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int dist = 1)
    { return NorthNeighbour(info, BSY, BSY, ix, iy, dist); }
    static int inblock_n3(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int dist = 1)
    { return EastNeighbour(info, BSY, BSY, ix, iy, dist); }
    static int neiblock_n(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int offset = 0)
    { return NorthmostCell(info, BSY, BSY, ix, iy, offset); }

    constexpr static Compass EdgeType = {South};
    constexpr static std::array<int,3> Zchild1_idx = {0,1,0};
    constexpr static std::array<int,3> Zchild2_idx = {1,1,0};
};

class WestEdgeCell : public CellIndexer{
  public:
    static int inblock_n1(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int dist = 1)
    { return NorthNeighbour(info, BSY, BSY, ix, iy, dist); }
    static int inblock_n2(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int dist = 1)
    { return EastNeighbour(info, BSY, BSY, ix, iy, dist); }
    static int inblock_n3(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int dist = 1)
    { return SouthNeighbour(info, BSY, BSY, ix, iy, dist); }
    static int neiblock_n(const BlockInfo &info, const int &BSX, const int &BSY, const int &ix, const int &iy, const int offset = 0)
    { return EastmostCell(info, BSY, BSY, ix, iy, offset); }

    constexpr static Compass EdgeType = {West};
    constexpr static std::array<int,3> Zchild1_idx = {1,0,0};
    constexpr static std::array<int,3> Zchild2_idx = {1,1,0};
};

static long long getZchild(const BlockInfo& rhsNei, const std::array<int,3> &Zchild_idx)
{
  return rhsNei.Zchild[Zchild_idx[0]][Zchild_idx[1]][Zchild_idx[2]];
}

class InnerUpperTaylor {
  public:
    void operator()(
        const double &sign,
        const int &centre_idx,
        const int &bottom_idx,
        const int &top_idx,
        std::map<int,double> &row_map)
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

class InnerLowerTaylor {
  public:
    void operator()(
        const double& sign,
        const int &centre_idx,
        const int &bottom_idx,
        const int &top_idx,
        std::map<int,double> &row_map)
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

class CornerUpperTaylor {
  public:
    void operator()(
        const double& sign,
        const int &centre_idx,
        const int &offset1_idx,
        const int &offset2_idx,
        std::map<int,double> &row_map)
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

class CornerLowerTaylor {
  public:
    void operator()(
        const double& sign,
        const int &centre_idx,
        const int &offset1_idx,
        const int &offset2_idx,
        std::map<int,double> &row_map)
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
        F taylor)
    {
      taylor(sign, coarse_centre_idx, coarse_offset1_idx, coarse_offset2_idx, row_map);
      row_map[fine_close_idx] += sign * p_fine_close;
      row_map[fine_far_idx] += sign * p_fine_far;
    } 
  private:
    static constexpr double p_fine_close = 2./3.;
    static constexpr double p_fine_far = -1./5.;
};

double cudaAMRSolver::getA_local(int I1,int I2) //matrix for Poisson's equation on a uniform grid
{
   static constexpr int BSX = VectorBlock::sizeX;
   static constexpr int BSY = VectorBlock::sizeY;
   int j1 = I1 / BSX;
   int i1 = I1 % BSX;
   int j2 = I2 / BSY;
   int i2 = I2 % BSY;
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
  for (int i(0); i<N ; i++)
  {
    L[i].resize(i+1);
    L_inv[i].resize(i+1);
    // L_inv will act as right block in GJ algorithm, init as identity
    for (int j(0); j<=i; j++){
      L_inv[i][j] = (i == j) ? 1. : 0.;
    }
  }

  // compute the Cholesky decomposition of the preconditioner with Cholesky-Crout
  for (int i(0); i<N ; i++)
  {
    double s1 = 0;
    for (int k(0); k<=i-1; k++)
      s1 += L[i][k]*L[i][k];
    L[i][i] = sqrt(getA_local(i,i) - s1);
    for (int j(i+1); j<N; j++)
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
  for (int br(0); br<N; br++)
  { // 'br' - base row in which all columns up to L_lb[br][br] are already zero
    const double bsf = 1. / L[br][br];
    for (int c(0); c<=br; c++)
      L_inv[br][c] *= bsf;

    for (int wr(br+1); wr<N; wr++)
    { // 'wr' - working row where elements below L_lb[br][br] will be set to zero
      const double wsf = L[wr][br];
      for (int c(0); c<=br; c++)
        L_inv[wr][c] -= (wsf * L_inv[br][c]);
    }
  }

  // P_inv_ holds inverse preconditionner in row major order!
  P_inv_.resize(N * N);
  for (int i(0); i<N; i++)
  for (int j(0); j<N; j++)
  {
    double aux = 0.;
    for (int k(0); k<N; k++) // P_inv_ = (L^T)^{-1} L^{-1}
      aux += (i <= k && j <=k) ? L_inv[k][i] * L_inv[k][j] : 0.;

    P_inv_[i*N+j] = aux;
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

void cudaAMRSolver::cooMatPushBackRow(
    const int &row_idx,
    std::map<int,double> &row_map)
{
  double aux = 0.;
  for (const auto &[col_idx, val] : row_map)
  {
    this->cooMatPushBack(val, row_idx, col_idx);
    aux+=val;
  }
  if (std::abs(aux)>1e-10){
    std::string errorMessage = "Row: " + std::to_string(row_idx) 
                             + ", sum: " + std::to_string(aux);
    throw std::runtime_error(errorMessage);
  }

}

template<class EdgeHelper >
void cudaAMRSolver::neiBlockElement(
  BlockInfo &rhs_info,
  const int &BSX,
  const int &BSY,
  const int &ix,
  const int &iy,
  BlockInfo &rhsNei,
  EdgeHelper helper,
  std::vector< std::map<int,double> > &row_map)
{
  const int sfc_idx = CellIndexer::This(rhs_info, BSX, BSY, ix, iy);
  PolyInterpolation interpolator;

  if (this->sim.tmp->Tree(rhsNei).Exists())
  { 
    const int n_idx = helper.neiblock_n(rhsNei, BSX, BSY, ix, iy);

    // Map flux associated to out-of-block edges at the same level of refinement
    row_map[sfc_idx][n_idx] += 1.; row_map[sfc_idx][sfc_idx] += -1.;
  }
  else if (this->sim.tmp->Tree(rhsNei).CheckCoarser())
  {
    BlockInfo &rhsNei_c = this->sim.tmp->getBlockInfoAll(rhs_info.level - 1 , rhsNei.Zparent);
    if (this->sim.tmp->Tree(rhsNei_c).Exists())
    {
      const bool NorthSouthEdge = helper.EdgeType == North || helper.EdgeType == South;
      const bool EastWestEdge = helper.EdgeType == East || helper.EdgeType == West;
      int ix_c = ix / 2;
      int iy_c = iy / 2;
      if (NorthSouthEdge && (rhs_info.index[0] % 2 == 1))
      { // North/South edge leftward fine block 
        ix_c += BSX / 2;
      }
      if (EastWestEdge && (rhs_info.index[1] % 2 == 1))
      { // East/West edge top fine block
        iy_c += BSY / 2;
      }

      // Map flux associated to interpolation at interface
      const int coarse_centre_idx = helper.neiblock_n(rhsNei_c, BSX, BSY,ix_c, iy_c);
      const int fine_far_idx = helper.inblock_n2(rhs_info, BSX, BSY, ix, iy); // index of cell on opposite inblock edge
      if ( (ix_c == 0 && NorthSouthEdge) || (iy_c == 0 && EastWestEdge) )
      {
        const int coarse_offset1_idx = helper.neiblock_n(rhsNei_c, BSX, BSY, ix_c+1,iy_c+1);
        const int coarse_offset2_idx = helper.neiblock_n(rhsNei_c, BSX, BSY, ix_c+2,iy_c+2);

        if ( (ix == 0 && NorthSouthEdge) || (iy == 0 && EastWestEdge) )
        {
          interpolator(1., coarse_centre_idx, coarse_offset1_idx, coarse_offset2_idx, sfc_idx, fine_far_idx, row_map[sfc_idx], CornerLowerTaylor());
          interpolator(-1., coarse_centre_idx, coarse_offset1_idx, coarse_offset2_idx, sfc_idx, fine_far_idx, row_map[coarse_centre_idx], CornerLowerTaylor());
          // Map negative diagonal contribution to flux
          row_map[sfc_idx][sfc_idx] += -1.;
          row_map[coarse_centre_idx][sfc_idx] += 1.;
        }
        else if ( (ix == 1 && NorthSouthEdge) || (iy == 1 && EastWestEdge) )
        {
          interpolator(1., coarse_centre_idx, coarse_offset1_idx, coarse_offset2_idx, sfc_idx, fine_far_idx, row_map[sfc_idx], CornerUpperTaylor());
          interpolator(-1., coarse_centre_idx, coarse_offset1_idx, coarse_offset2_idx, sfc_idx, fine_far_idx, row_map[coarse_centre_idx], CornerUpperTaylor());
          // Map negative diagonal contribution to flux
          row_map[sfc_idx][sfc_idx] += -1.;
          row_map[coarse_centre_idx][sfc_idx] += 1.;
        }
        else { throw std::runtime_error("Interface error 1"); }

      }
      else if ( (ix_c == (BSX-1) && NorthSouthEdge) || (iy_c == (BSY-1) && EastWestEdge) )
      {
        const int coarse_offset1_idx = helper.neiblock_n(rhsNei_c, BSX, BSY, ix_c-1,iy_c-1);
        const int coarse_offset2_idx = helper.neiblock_n(rhsNei_c, BSX, BSY, ix_c-2,iy_c-2);

        if ( (ix == BSX-2 && NorthSouthEdge) || (iy == BSY-2 && EastWestEdge) )
        {
          interpolator(1., coarse_centre_idx, coarse_offset1_idx, coarse_offset2_idx, sfc_idx, fine_far_idx, row_map[sfc_idx], CornerLowerTaylor());
          interpolator(-1., coarse_centre_idx, coarse_offset1_idx, coarse_offset2_idx, sfc_idx, fine_far_idx, row_map[coarse_centre_idx], CornerLowerTaylor());
          // Map negative diagonal contribution to flux
          row_map[sfc_idx][sfc_idx] += -1.;
          row_map[coarse_centre_idx][sfc_idx] += 1.;
        }
        else if ( (ix == BSX-1 && NorthSouthEdge) || (iy == BSY-1 && EastWestEdge) )
        {
          interpolator(1., coarse_centre_idx, coarse_offset1_idx, coarse_offset2_idx, sfc_idx, fine_far_idx, row_map[sfc_idx], CornerUpperTaylor());
          interpolator(-1., coarse_centre_idx, coarse_offset1_idx, coarse_offset2_idx, sfc_idx, fine_far_idx, row_map[coarse_centre_idx], CornerUpperTaylor());
          // Map negative diagonal contribution to flux
          row_map[sfc_idx][sfc_idx] += -1.;
          row_map[coarse_centre_idx][sfc_idx] += 1.;
        }
        else { throw std::runtime_error("Interface error 2"); }
      }
      else
      {
        const int coarse_offset1_idx = helper.neiblock_n(rhsNei_c, BSX, BSY, ix_c-1,iy_c-1); // bottom
        const int coarse_offset2_idx = helper.neiblock_n(rhsNei_c, BSX, BSY, ix_c+1,iy_c+1); // top

        if ( (ix % 2 == 0 && NorthSouthEdge) || (iy % 2 == 0 && EastWestEdge) )
        {
          interpolator(1., coarse_centre_idx, coarse_offset1_idx, coarse_offset2_idx, sfc_idx, fine_far_idx, row_map[sfc_idx], InnerLowerTaylor());
          interpolator(-1., coarse_centre_idx, coarse_offset1_idx, coarse_offset2_idx, sfc_idx, fine_far_idx, row_map[coarse_centre_idx], InnerLowerTaylor());
          // Map negative diagonal contribution to flux
          row_map[sfc_idx][sfc_idx] += -1.;
          row_map[coarse_centre_idx][sfc_idx] += 1.;
        }
        else if ( (ix % 2 == 1 && NorthSouthEdge) || (iy % 2 == 1 && EastWestEdge) )
        {
          interpolator(1., coarse_centre_idx, coarse_offset1_idx, coarse_offset2_idx, sfc_idx, fine_far_idx, row_map[sfc_idx], InnerUpperTaylor());
          interpolator(-1., coarse_centre_idx, coarse_offset1_idx, coarse_offset2_idx, sfc_idx, fine_far_idx, row_map[coarse_centre_idx], InnerUpperTaylor());
          // Map negative diagonal contribution to flux
          row_map[sfc_idx][sfc_idx] += -1.;
          row_map[coarse_centre_idx][sfc_idx] += 1.;
        }
        else { throw std::runtime_error("Interface error 3"); }
      }
    }
    else { throw std::runtime_error("Runtime2"); }
  }
  else if (this->sim.tmp->Tree(rhsNei).CheckFiner())
  {
//    const bool NorthSouthEdge = helper.EdgeType == North || helper.EdgeType == South;
//    const bool EastWestEdge = helper.EdgeType == East || helper.EdgeType == West;
//
//    /* Determine which fine block the current coarse edge neighbours.
//       It is assumed that members 'Zchild_1' and 'Zchild_2' respect this order:
//       Zchild[0][0][0] is (2i  ,2j  ,2k  )
//       Zchild[1][0][0] is (2i+1,2j  ,2k  )
//       Zchild[0][1][0] is (2i  ,2j+1,2k  )
//       Zchild[1][1][0] is (2i+1,2j+1,2k  ) */
//
//    long long rhsNei_Zchild;
//    if (NorthSouthEdge)
//      rhsNei_Zchild = ix < BSX / 2 ? getZchild(rhsNei, helper.Zchild1_idx) : getZchild(rhsNei, helper.Zchild2_idx);
//    else if (EastWestEdge)
//      rhsNei_Zchild = iy < BSY / 2 ? getZchild(rhsNei, helper.Zchild1_idx) : getZchild(rhsNei, helper.Zchild2_idx);
//    BlockInfo &rhsNei_f = this->sim.tmp->getBlockInfoAll(rhs_info.level + 1, rhsNei_Zchild);
//
//    if (this->sim.tmp->Tree(rhsNei_f).Exists())
//    { 
//      const int ix_f = (ix % (BSX/2)) * 2;
//      const int iy_f = (iy % (BSY/2)) * 2;
//
//      int coarse_offset1_idx;
//      int coarse_offset2_idx;
//      int fine_close_idx;
//      int fine_far_idx;
//
//      if ( (ix == 0 && iy == 0) || (ix == 0 && iy == BSY-1) || (ix == BSX-1 && iy == 0) || (ix == BSX-1 && iy == BSY-1) )
//      {
//        if ( (ix == 0) && NorthSouthEdge)
//        {
//          coarse_offset1_idx = CellIndexer::EastNeighbour(rhs_info, BSX, BSY, ix, iy, 1);
//          coarse_offset2_idx = CellIndexer::EastNeighbour(rhs_info, BSX, BSY, ix, iy, 2);
//        }
//        else if ( (iy == 0) && EastWestEdge)
//        {
//          coarse_offset1_idx = CellIndexer::NorthNeighbour(rhs_info, BSX, BSY, ix, iy, 1);
//          coarse_offset2_idx = CellIndexer::NorthNeighbour(rhs_info, BSX, BSY, ix, iy, 2);
//        }
//        else if ( (ix == BSX-1) && NorthSouthEdge)
//        {
//          coarse_offset1_idx = CellIndexer::WestNeighbour(rhs_info, BSX, BSY, ix, iy, 1);
//          coarse_offset2_idx = CellIndexer::WestNeighbour(rhs_info, BSX, BSY, ix, iy, 2);
//        }
//        else if ( (iy == BSY-1) && EastWestEdge)
//        {
//          coarse_offset1_idx = CellIndexer::SouthNeighbour(rhs_info, BSX, BSY, ix, iy, 1);
//          coarse_offset2_idx = CellIndexer::SouthNeighbour(rhs_info, BSX, BSY, ix, iy, 2);
//        }
//        else { throw std::runtime_error("Interface error 4"); }
//
//        // Add flux at left/lower corner interface
//        fine_close_idx = helper.neiblock_n(rhsNei_f, BSX, BSY, ix_f, iy_f, 0);
//        fine_far_idx = helper.neiblock_n(rhsNei_f, BSX, BSY, ix_f, iy_f, 1);
//        row_map[sfc_idx][fine_close_idx] += 1.;
//        interpolator(-1., sfc_idx, coarse_offset1_idx, coarse_offset2_idx, fine_close_idx, fine_far_idx, row_map[sfc_idx], CornerLowerTaylor());
//        row_map[fine_close_idx][fine_close_idx] += -1.;
//        interpolator(1., sfc_idx, coarse_offset1_idx, coarse_offset2_idx, fine_close_idx, fine_far_idx, row_map[fine_close_idx], CornerLowerTaylor());
//
//        // Add flux at right/higher corner interface
//        fine_close_idx = helper.neiblock_n(rhsNei_f, BSX, BSY, ix_f+1, iy_f+1, 0);
//        fine_far_idx = helper.neiblock_n(rhsNei_f, BSX, BSY, ix_f+1, iy_f+1, 1);
//        row_map[sfc_idx][fine_close_idx] += 1.;
//        interpolator(-1., sfc_idx, coarse_offset1_idx, coarse_offset2_idx, fine_close_idx, fine_far_idx, row_map[sfc_idx], CornerUpperTaylor());
//        row_map[fine_close_idx][fine_close_idx] += -1.;
//        interpolator(1., sfc_idx, coarse_offset1_idx, coarse_offset2_idx, fine_close_idx, fine_far_idx, row_map[fine_close_idx], CornerUpperTaylor());
//      }
//      else
//      {
//        if (NorthSouthEdge)
//        {
//          coarse_offset1_idx = CellIndexer::WestNeighbour(rhs_info, BSX, BSY, ix, iy);  // bottom
//          coarse_offset2_idx = CellIndexer::EastNeighbour(rhs_info, BSX, BSY, ix, iy);  // top
//        } 
//        else if (EastWestEdge)
//        {
//          coarse_offset1_idx = CellIndexer::SouthNeighbour(rhs_info, BSX, BSY, ix, iy); // bottom
//          coarse_offset2_idx = CellIndexer::NorthNeighbour(rhs_info, BSX, BSY, ix, iy); // top 
//        } 
//
//        // Add flux at left/lower corner interface
//        fine_close_idx = helper.neiblock_n(rhsNei_f, BSX, BSY, ix_f, iy_f, 0);
//        fine_far_idx = helper.neiblock_n(rhsNei_f, BSX, BSY, ix_f, iy_f, 1);
//        row_map[sfc_idx][fine_close_idx] += 1.;
//        interpolator(-1., sfc_idx, coarse_offset1_idx, coarse_offset2_idx, fine_close_idx, fine_far_idx, row_map[sfc_idx], InnerLowerTaylor());
//        row_map[fine_close_idx][fine_close_idx] += -1.;
//        interpolator(1., sfc_idx, coarse_offset1_idx, coarse_offset2_idx, fine_close_idx, fine_far_idx, row_map[fine_close_idx], InnerLowerTaylor());
//
//        // Add flux at right/higher corner interface
//        fine_close_idx = helper.neiblock_n(rhsNei_f, BSX, BSY, ix_f+1, iy_f+1, 0);
//        fine_far_idx = helper.neiblock_n(rhsNei_f, BSX, BSY, ix_f+1, iy_f+1, 1);
//        row_map[sfc_idx][fine_close_idx] += 1.;
//        interpolator(-1., sfc_idx, coarse_offset1_idx, coarse_offset2_idx, fine_close_idx, fine_far_idx, row_map[sfc_idx], InnerUpperTaylor());
//        row_map[fine_close_idx][fine_close_idx] += -1.;
//        interpolator(1., sfc_idx, coarse_offset1_idx, coarse_offset2_idx, fine_close_idx, fine_far_idx, row_map[fine_close_idx], InnerUpperTaylor());
//      }
//    }
//    else { throw std::runtime_error("Runtime4"); }
  }
  else { 
    std::cout << "rhs_info.blockID: " << rhs_info.blockID << std::endl;
    std::cout << "EdgeType: " << helper.EdgeType << std::endl;
    std::cout << "rhs_info.index[0]: " << rhs_info.index[0] << ", rhs_info.index[1]: " << rhs_info.index[1] << std::endl;
    std::cout << "rhs_info.level: " << rhs_info.level << std::endl;
    throw std::runtime_error("Runtime5"); }
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
    EdgeHelper helper,
    std::vector< std::map<int,double> > &row_map)
{
    const int sfc_idx = CellIndexer::This(rhs_info, BSX, BSY, ix, iy);
    const int n1_idx = helper.inblock_n1(rhs_info, BSX, BSY, ix, iy); // in-block neighbour 1
    const int n2_idx = helper.inblock_n2(rhs_info, BSX, BSY, ix, iy); // in-block neighbour 2
    const int n3_idx = helper.inblock_n3(rhs_info, BSX, BSY, ix, iy); // in-block neighbour 3

    // Map fluxes associated to in-block edges
    row_map[sfc_idx][n1_idx] += 1.;
    row_map[sfc_idx][n2_idx] += 1.;
    row_map[sfc_idx][n3_idx] += 1.;
    row_map[sfc_idx][sfc_idx] += -3.;

    if (!isBoundary)
      this->neiBlockElement(rhs_info, BSX, BSY, ix, iy, rhsNei, helper, row_map);
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
    EdgeHelper2 helper2,
    std::vector< std::map<int,double> > &row_map)
{
    const int sfc_idx = CellIndexer::This(rhs_info, BSX, BSY, ix, iy);
    const int n1_idx = helper1.inblock_n2(rhs_info, BSX, BSY, ix, iy); // helper.inblock_n2 is the opposite edge
    const int n2_idx = helper2.inblock_n2(rhs_info, BSX, BSY, ix, iy); // makes corner input order invariant

    // Map fluxes associated to in-block edges
    row_map[sfc_idx][n1_idx] += 1.;
    row_map[sfc_idx][n2_idx] += 1.;
    row_map[sfc_idx][sfc_idx] += -2.;

    if (!isBoundary1)
      this->neiBlockElement(rhs_info, BSX, BSY, ix, iy, rhsNei_1, helper1, row_map);
    if (!isBoundary2)
      this->neiBlockElement(rhs_info, BSX, BSY, ix, iy, rhsNei_2, helper2, row_map);
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
  std::vector< std::map<int,double> > row_map(N);
  for(int i=0; i< Nblocks; i++)
  {    
    BlockInfo &rhs_info = RhsInfo[i];
    ScalarBlock & __restrict__ rhs  = *(ScalarBlock*) RhsInfo[i].ptrBlock;
    ScalarBlock & __restrict__ p  = *(ScalarBlock*) zInfo[i].ptrBlock;

    // Construct RHS and x_0 vectors for linear system
    for(int iy=0; iy<BSY; iy++)
    for(int ix=0; ix<BSX; ix++)
    {
      const int sfc_idx = CellIndexer::This(rhs_info, BSX, BSY, ix, iy);
      //double arr[2];
      //rhs_info.pos(arr,ix,iy);
      //b_[sfc_idx] = 0.;
      //x_[sfc_idx] = arr[0]*arr[0]*arr[0];
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
      const int sfc_idx = CellIndexer::This(rhs_info, BSX, BSY, ix, iy);
      const int wn_idx = CellIndexer::WestNeighbour(rhs_info, BSX, BSY, ix, iy);
      const int en_idx = CellIndexer::EastNeighbour(rhs_info, BSX, BSY, ix, iy);
      const int sn_idx = CellIndexer::SouthNeighbour(rhs_info, BSX, BSY, ix, iy);
      const int nn_idx = CellIndexer::NorthNeighbour(rhs_info, BSX, BSY, ix, iy);
      
      row_map[sfc_idx][wn_idx] += 1.;
      row_map[sfc_idx][en_idx] += 1.;
      row_map[sfc_idx][sn_idx] += 1.;
      row_map[sfc_idx][nn_idx] += 1.;
      row_map[sfc_idx][sfc_idx] += -4.;
    }

    for(int ix=1; ix<BSX-1; ix++)
    { // Add matrix elements associated to cells on the northern boundary of the block (excl. corners)
      int iy = BSY-1;
      this->edgeBoundaryCell(rhs_info, BSX, BSY, ix, iy, isNorthBoundary, rhsNei_north, NorthEdgeCell(), row_map);

      // Add matrix elements associated to cells on the southern boundary of the block (excl. corners)
      iy = 0;
      this->edgeBoundaryCell(rhs_info, BSX, BSY, ix, iy, isSouthBoundary, rhsNei_south, SouthEdgeCell(), row_map);
    }

    for(int iy=1; iy<BSY-1; iy++)
    { // Add matrix elements associated to cells on the western boundary of the block (excl. corners)
      int ix = 0;
      this->edgeBoundaryCell(rhs_info, BSX, BSY, ix, iy, isWestBoundary, rhsNei_west, WestEdgeCell(), row_map);

      // Add matrix elements associated to cells on the eastern boundary of the block (excl. corners)
      ix = BSX-1;
      this->edgeBoundaryCell(rhs_info, BSX, BSY, ix, iy, isEastBoundary, rhsNei_east, EastEdgeCell(), row_map);
    }
    // Add matrix elements associated to cells on the north-west corner of the block (excl. corners)
    int ix = 0;
    int iy = BSY-1;
    this->cornerBoundaryCell(
        rhs_info, BSX, BSY, ix, iy, 
        isWestBoundary, rhsNei_west, WestEdgeCell(), 
        isNorthBoundary, rhsNei_north, NorthEdgeCell(), row_map);

    // Add matrix elements associated to cells on the north-east corner of the block (excl. corners)
    ix = BSX-1;
    iy = BSY-1;
    this->cornerBoundaryCell(
        rhs_info, BSX, BSY, ix, iy, 
        isNorthBoundary, rhsNei_north, NorthEdgeCell(), 
        isEastBoundary, rhsNei_east, EastEdgeCell(), row_map);
    
    // Add matrix elements associated to cells on the south-east corner of the block (excl. corners)
    ix = BSX-1;
    iy = 0;
    this->cornerBoundaryCell(
        rhs_info, BSX, BSY, ix, iy, 
        isEastBoundary, rhsNei_east, EastEdgeCell(), 
        isSouthBoundary, rhsNei_south, SouthEdgeCell(), row_map);

    // Add matrix elements associated to cells on the south-west corner of the block (excl. corners)
    ix = 0;
    iy = 0;
    this->cornerBoundaryCell(
        rhs_info, BSX, BSY, ix, iy, 
        isSouthBoundary, rhsNei_south, SouthEdgeCell(), 
        isWestBoundary, rhsNei_west, WestEdgeCell(), row_map);
  }

  for (int i(0); i < N; i++)
    this->cooMatPushBackRow(i, row_map[i]);

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
           P(ix,iy).s += -avg;
     }
  }
}

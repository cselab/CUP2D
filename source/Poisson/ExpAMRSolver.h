//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Operator.h"
#include "Cubism/FluxCorrection.h"
#include "Base.h"
#include "LocalSpMatDnVec.h"

class ExpAMRSolver : public PoissonSolver
{
  /*
  Method used to solve Poisson's equation: https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
  */
public:
  std::string getName() {
    // ExpAMRSolver == AMRSolver for explicit linear system
    return "ExpAMRSolver";
  }
  // Constructor and destructor
  ExpAMRSolver(SimulationData& s);
  ~ExpAMRSolver() = default;

  //main function used to solve Poisson's equation
  void solve(
      const ScalarGrid *input, 
      ScalarGrid * const output);

protected:
  //this struct contains information such as the currect timestep size, fluid properties and many others
  SimulationData& sim; 

  int rank_;
  MPI_Comm m_comm_;
  int comm_size_;

  static constexpr int BSX_ = VectorBlock::sizeX;
  static constexpr int BSY_ = VectorBlock::sizeY;

  //This returns element K_{I1,I2}. It is used when we invert K
  double getA_local(int I1,int I2);

  // Method to add off-diagonal matrix element associated to cell in 'rhsNei' block
  template<class EdgeIndexer >
  void makeFlux(
      const cubism::BlockInfo &rhs_info,
      const int &ix,
      const int &iy,
      const cubism::BlockInfo &rhsNei,
      const EdgeIndexer &helper,
      SpRowInfo &row) const;

  // Method to construct matrix row for cell on edge of block
  template<class EdgeIndexer>
  void makeEdgeCellRow( // excluding corners
      const cubism::BlockInfo &rhs_info,
      const int &ix,
      const int &iy,
      const bool &isBoundary,
      const cubism::BlockInfo &rhsNei,
      const EdgeIndexer &helper);

  // Method to construct matrix row for cell on corner of block
  template<class EdgeIndexer1, class EdgeIndexer2>
  void makeCornerCellRow(
      const cubism::BlockInfo &rhs_info,
      const int &ix,
      const int &iy,
      const bool &isBoundary1,
      const cubism::BlockInfo &rhsNei_1,
      const EdgeIndexer1 &helper1, 
      const bool &isBoundary2,
      const cubism::BlockInfo &rhsNei_2,
      const EdgeIndexer2 &helper2);

  // Method to compute A and b for the current mesh
  void getMat(); // update LHS and RHS after refinement
  void getVec(); // update initial guess and RHS vecs only

  // Distributed linear system which uses local indexing
  std::unique_ptr<LocalSpMatDnVec> LocalLS_;

  std::vector<long long> Nblocks_xcumsum_;
  std::vector<long long> Nrows_xcumsum_;

  // Edge descriptors to allow algorithmic access to cell indices regardless of edge type
  class CellIndexer{
    public:
      CellIndexer(SimulationData &s, std::vector<long long> &Nblocks_xcumsum) : sim(s), Nblocks_xcumsum_(Nblocks_xcumsum) {}
      ~CellIndexer() = default;

      long long This(const cubism::BlockInfo &info, const int &ix, const int &iy) const
      { return (info.blockID + Nblocks_xcumsum_[sim.tmp->Tree(info).rank()])*BLEN + (long long)(iy*BSX + ix); }
      long long WestNeighbour(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return (info.blockID + Nblocks_xcumsum_[sim.tmp->Tree(info).rank()])*BLEN + (long long)(iy*BSX + ix-dist); }
      long long NorthNeighbour(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return (info.blockID + Nblocks_xcumsum_[sim.tmp->Tree(info).rank()])*BLEN + (long long)((iy+dist)*BSX + ix);}
      long long EastNeighbour(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return (info.blockID + Nblocks_xcumsum_[sim.tmp->Tree(info).rank()])*BLEN + (long long)(iy*BSX + ix+dist); }
      long long SouthNeighbour(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return (info.blockID + Nblocks_xcumsum_[sim.tmp->Tree(info).rank()])*BLEN + (long long)((iy-dist)*BSX + ix); }

      long long WestmostCell(const cubism::BlockInfo &info, const int &ix, const int &iy, const int offset = 0) const
      { return (info.blockID + Nblocks_xcumsum_[sim.tmp->Tree(info).rank()])*BLEN + (long long)(iy*BSX + offset); }
      long long NorthmostCell(const cubism::BlockInfo &info, const int &ix, const int &iy, const int offset = 0) const
      { return (info.blockID + Nblocks_xcumsum_[sim.tmp->Tree(info).rank()])*BLEN + (long long)((BSY-1-offset)*BSX + ix); }
      long long EastmostCell(const cubism::BlockInfo &info, const int &ix, const int &iy, const int offset = 0) const
      { return (info.blockID + Nblocks_xcumsum_[sim.tmp->Tree(info).rank()])*BLEN + (long long)(iy*BSX + (BSX-1-offset)); }
      long long SouthmostCell(const cubism::BlockInfo &info, const int &ix, const int &iy, const int offset = 0) const
      { return (info.blockID + Nblocks_xcumsum_[sim.tmp->Tree(info).rank()])*BLEN + (long long)(offset*BSX + ix); }

      SimulationData &sim;
      std::vector<long long> &Nblocks_xcumsum_;
      static constexpr int BSX = VectorBlock::sizeX;
      static constexpr int BSY = VectorBlock::sizeY;
      static constexpr long long BLEN = BSX * BSY;
  };

  class NorthEdgeIndexer : public CellIndexer{
    public:
      NorthEdgeIndexer(SimulationData &s, std::vector<long long> &Nblocks_xcumsum) : CellIndexer(s, Nblocks_xcumsum) {}

      long long inblock_n1(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return EastNeighbour(info, ix, iy, dist); }
      long long inblock_n2(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return SouthNeighbour(info, ix, iy, dist); }
      long long inblock_n3(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return WestNeighbour(info, ix, iy, dist); }
      long long neiblock_n(const cubism::BlockInfo &nei_info, const int &ix, const int &iy, const int offset = 0) const
      { return SouthmostCell(nei_info, ix, iy, offset); }

      long long forward(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return EastNeighbour(info, ix, iy, dist); }
      long long backward(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return WestNeighbour(info, ix, iy, dist); }

      static bool back_corner(const int &ix, const int &iy)
      { return ix == 0 || ix == BSX / 2; }
      static bool front_corner(const int &ix, const int &iy)
      { return ix == BSX - 1 || ix == (BSX / 2 - 1); }
      static bool mod(const int &ix, const int &iy)
      { return ix % 2 == 0; }

      static int ix_c(const cubism::BlockInfo &info, const int &ix, const int &iy)
      { return (info.index[0] % 2 == 1) ? (ix/2 + BSX/2) : (ix/2); }
      static int iy_c(const cubism::BlockInfo &info, const int &ix, const int &iy)
      { return -1; } // in correct execution, this should not participate anywhere

      static long long Zchild(const cubism::BlockInfo &nei_info, const int &ix, const int &iy)
      {return ix < BSX/2 ? nei_info.Zchild[0][0][0] : nei_info.Zchild[1][0][0];}
  };

  class EastEdgeIndexer : public CellIndexer{
    public:
      EastEdgeIndexer(SimulationData &s, std::vector<long long> &Nblocks_xcumsum) : CellIndexer(s, Nblocks_xcumsum) {}

      long long inblock_n1(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return SouthNeighbour(info, ix, iy, dist); }
      long long inblock_n2(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return WestNeighbour(info, ix, iy, dist); }
      long long inblock_n3(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return NorthNeighbour(info, ix, iy, dist); }
      long long neiblock_n(const cubism::BlockInfo &nei_info, const int &ix, const int &iy, const int offset = 0) const
      { return WestmostCell(nei_info, ix, iy, offset); }

      long long forward(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return NorthNeighbour(info, ix, iy, dist); }
      long long backward(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return SouthNeighbour(info, ix, iy, dist); }

      static bool back_corner(const int &ix, const int &iy)
      { return iy == 0 || iy == BSY / 2; }
      static bool front_corner(const int &ix, const int &iy)
      { return iy == BSY - 1 || iy == (BSY / 2 - 1); }
      static bool mod(const int &ix, const int &iy)
      { return iy % 2 == 0; }

      static int ix_c(const cubism::BlockInfo &info, const int &ix, const int &iy)
      { return -1; }
      static int iy_c(const cubism::BlockInfo &info, const int &ix, const int &iy)
      { return (info.index[1] % 2 == 1) ? (iy/2 + BSY/2) : (iy/2); }

      static long long Zchild(const cubism::BlockInfo &nei_info, const int &ix, const int &iy)
      {return iy < BSY/2 ? nei_info.Zchild[0][0][0] : nei_info.Zchild[0][1][0];}
  };

  class SouthEdgeIndexer : public CellIndexer{
    public:
      SouthEdgeIndexer(SimulationData &s, std::vector<long long> &Nblocks_xcumsum) : CellIndexer(s, Nblocks_xcumsum) {}

      long long inblock_n1(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return WestNeighbour(info, ix, iy, dist); }
      long long inblock_n2(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return NorthNeighbour(info, ix, iy, dist); }
      long long inblock_n3(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return EastNeighbour(info, ix, iy, dist); }
      long long neiblock_n(const cubism::BlockInfo &nei_info, const int &ix, const int &iy, const int offset = 0) const
      { return NorthmostCell(nei_info, ix, iy, offset); }

      long long forward(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return EastNeighbour(info, ix, iy, dist); }
      long long backward(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return WestNeighbour(info, ix, iy, dist); }

      static bool back_corner(const int &ix, const int &iy)
      { return ix == 0 || ix == BSX / 2; }
      static bool front_corner(const int &ix, const int &iy)
      { return ix == BSX - 1 || ix == (BSX / 2 - 1); }
      static bool mod(const int &ix, const int &iy)
      { return ix % 2 == 0; }

      static int ix_c(const cubism::BlockInfo &info, const int &ix, const int &iy)
      { return (info.index[0] % 2 == 1) ? (ix/2 + BSX/2) : (ix/2); }
      static int iy_c(const cubism::BlockInfo &info, const int &ix, const int &iy)
      { return -1; }

      static long long Zchild(const cubism::BlockInfo &nei_info, const int &ix, const int &iy)
      {return ix < BSX/2 ? nei_info.Zchild[0][1][0] : nei_info.Zchild[1][1][0];}
  };

  class WestEdgeIndexer : public CellIndexer{
    public:
      WestEdgeIndexer(SimulationData &s, std::vector<long long> &Nblocks_xcumsum) : CellIndexer(s, Nblocks_xcumsum) {}

      long long inblock_n1(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return NorthNeighbour(info, ix, iy, dist); }
      long long inblock_n2(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return EastNeighbour(info, ix, iy, dist); }
      long long inblock_n3(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return SouthNeighbour(info, ix, iy, dist); }
      long long neiblock_n(const cubism::BlockInfo &nei_info, const int &ix, const int &iy, const int offset = 0) const
      { return EastmostCell(nei_info, ix, iy, offset); }

      long long forward(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return NorthNeighbour(info, ix, iy, dist); }
      long long backward(const cubism::BlockInfo &info, const int &ix, const int &iy, const int dist = 1) const
      { return SouthNeighbour(info, ix, iy, dist); }

      static bool back_corner(const int &ix, const int &iy)
      { return iy == 0 || iy == BSY / 2; }
      static bool front_corner(const int &ix, const int &iy)
      { return iy == BSY - 1 || iy == (BSY / 2 - 1); }
      static bool mod(const int &ix, const int &iy)
      { return iy % 2 == 0; }

      static int ix_c(const cubism::BlockInfo &info, const int &ix, const int &iy)
      { return -1; }
      static int iy_c(const cubism::BlockInfo &info, const int &ix, const int &iy)
      { return (info.index[1] % 2 == 1) ? (iy/2 + BSY/2) : (iy/2); }

      static long long Zchild(const cubism::BlockInfo &nei_info, const int &ix, const int &iy)
      {return iy < BSY/2 ? nei_info.Zchild[1][0][0] : nei_info.Zchild[1][1][0];}
  };

  // Edge descriptors for use in the class
  NorthEdgeIndexer NorthCell;
  EastEdgeIndexer  EastCell;
  SouthEdgeIndexer SouthCell;
  WestEdgeIndexer  WestCell;
};

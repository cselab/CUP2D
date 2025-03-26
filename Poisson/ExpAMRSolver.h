#pragma once
#include "../Operator.h"
#include "Base.h"
#include "Cubism/FluxCorrection.h"
#include "LocalSpMatDnVec.h"
class ExpAMRSolver : public PoissonSolver {
public:
  std::string getName() { return "ExpAMRSolver"; }
  ExpAMRSolver(SimulationData &s);
  ~ExpAMRSolver() = default;
  void solve(const ScalarGrid *input, ScalarGrid *const output);

protected:
  SimulationData &sim;
  int rank_;
  MPI_Comm m_comm_;
  int comm_size_;
  static constexpr int BSX_ = VectorBlock::sizeX;
  static constexpr int BSY_ = VectorBlock::sizeY;
  static constexpr int BLEN_ = BSX_ * BSY_;
  double getA_local(int I1, int I2);
  class EdgeCellIndexer;
  void makeFlux(const cubism::BlockInfo &rhs_info, const int ix, const int iy,
                const cubism::BlockInfo &rhsNei, const EdgeCellIndexer &indexer,
                SpRowInfo &row) const;
  void getMat();
  void getVec();
  std::unique_ptr<LocalSpMatDnVec> LocalLS_;
  std::vector<long long> Nblocks_xcumsum_;
  std::vector<long long> Nrows_xcumsum_;
  class CellIndexer {
  public:
    CellIndexer(const ExpAMRSolver &pSolver) : ps(pSolver) {}
    ~CellIndexer() = default;
    long long This(const cubism::BlockInfo &info, const int ix,
                   const int iy) const {
      return blockOffset(info) + (long long)(iy * BSX_ + ix);
    }
    static bool validXm(const int ix, const int iy) { return ix > 0; }
    static bool validXp(const int ix, const int iy) { return ix < BSX_ - 1; }
    static bool validYm(const int ix, const int iy) { return iy > 0; }
    static bool validYp(const int ix, const int iy) { return iy < BSY_ - 1; }
    long long Xmin(const cubism::BlockInfo &info, const int ix, const int iy,
                   const int offset = 0) const {
      return blockOffset(info) + (long long)(iy * BSX_ + offset);
    }
    long long Xmax(const cubism::BlockInfo &info, const int ix, const int iy,
                   const int offset = 0) const {
      return blockOffset(info) + (long long)(iy * BSX_ + (BSX_ - 1 - offset));
    }
    long long Ymin(const cubism::BlockInfo &info, const int ix, const int iy,
                   const int offset = 0) const {
      return blockOffset(info) + (long long)(offset * BSX_ + ix);
    }
    long long Ymax(const cubism::BlockInfo &info, const int ix, const int iy,
                   const int offset = 0) const {
      return blockOffset(info) + (long long)((BSY_ - 1 - offset) * BSX_ + ix);
    }

  protected:
    long long blockOffset(const cubism::BlockInfo &info) const {
      return (info.blockID +
              ps.Nblocks_xcumsum_[ps.sim.tmp->Tree(info).rank()]) *
             BLEN_;
    }
    static int ix_f(const int ix) { return (ix % (BSX_ / 2)) * 2; }
    static int iy_f(const int iy) { return (iy % (BSY_ / 2)) * 2; }
    const ExpAMRSolver &ps;
  };
  class EdgeCellIndexer : public CellIndexer {
  public:
    EdgeCellIndexer(const ExpAMRSolver &pSolver) : CellIndexer(pSolver) {}
    virtual long long neiUnif(const cubism::BlockInfo &nei_info, const int ix,
                              const int iy) const = 0;
    virtual long long neiInward(const cubism::BlockInfo &info, const int ix,
                                const int iy) const = 0;
    virtual double taylorSign(const int ix, const int iy) const = 0;
    virtual int ix_c(const cubism::BlockInfo &info, const int ix) const {
      return info.index[0] % 2 == 0 ? ix / 2 : ix / 2 + BSX_ / 2;
    }
    virtual int iy_c(const cubism::BlockInfo &info, const int iy) const {
      return info.index[1] % 2 == 0 ? iy / 2 : iy / 2 + BSY_ / 2;
    }
    virtual long long neiFine1(const cubism::BlockInfo &nei_info, const int ix,
                               const int iy, const int offset = 0) const = 0;
    virtual long long neiFine2(const cubism::BlockInfo &nei_info, const int ix,
                               const int iy, const int offset = 0) const = 0;
    virtual bool isBD(const int ix, const int iy) const = 0;
    virtual bool isFD(const int ix, const int iy) const = 0;
    virtual long long Nei(const cubism::BlockInfo &info, const int ix,
                          const int iy, const int dist) const = 0;
    virtual long long Zchild(const cubism::BlockInfo &nei_info, const int ix,
                             const int iy) const = 0;
  };
  class XbaseIndexer : public EdgeCellIndexer {
  public:
    XbaseIndexer(const ExpAMRSolver &pSolver) : EdgeCellIndexer(pSolver) {}
    double taylorSign(const int ix, const int iy) const override {
      return iy % 2 == 0 ? -1. : 1.;
    }
    bool isBD(const int ix, const int iy) const override {
      return iy == BSY_ - 1 || iy == BSY_ / 2 - 1;
    }
    bool isFD(const int ix, const int iy) const override {
      return iy == 0 || iy == BSY_ / 2;
    }
    long long Nei(const cubism::BlockInfo &info, const int ix, const int iy,
                  const int dist) const override {
      return This(info, ix, iy + dist);
    }
  };
  class XminIndexer : public XbaseIndexer {
  public:
    XminIndexer(const ExpAMRSolver &pSolver) : XbaseIndexer(pSolver) {}
    long long neiUnif(const cubism::BlockInfo &nei_info, const int ix,
                      const int iy) const override {
      return Xmax(nei_info, ix, iy);
    }
    long long neiInward(const cubism::BlockInfo &info, const int ix,
                        const int iy) const override {
      return This(info, ix + 1, iy);
    }
    int ix_c(const cubism::BlockInfo &info, const int ix) const override {
      return BSX_ - 1;
    }
    long long neiFine1(const cubism::BlockInfo &nei_info, const int ix,
                       const int iy, const int offset = 0) const override {
      return Xmax(nei_info, ix_f(ix), iy_f(iy), offset);
    }
    long long neiFine2(const cubism::BlockInfo &nei_info, const int ix,
                       const int iy, const int offset = 0) const override {
      return Xmax(nei_info, ix_f(ix), iy_f(iy) + 1, offset);
    }
    long long Zchild(const cubism::BlockInfo &nei_info, const int ix,
                     const int iy) const override {
      return nei_info.Zchild[1][int(iy >= BSY_ / 2)][0];
    }
  };
  class XmaxIndexer : public XbaseIndexer {
  public:
    XmaxIndexer(const ExpAMRSolver &pSolver) : XbaseIndexer(pSolver) {}
    long long neiUnif(const cubism::BlockInfo &nei_info, const int ix,
                      const int iy) const override {
      return Xmin(nei_info, ix, iy);
    }
    long long neiInward(const cubism::BlockInfo &info, const int ix,
                        const int iy) const override {
      return This(info, ix - 1, iy);
    }
    int ix_c(const cubism::BlockInfo &info, const int ix) const override {
      return 0;
    }
    long long neiFine1(const cubism::BlockInfo &nei_info, const int ix,
                       const int iy, const int offset = 0) const override {
      return Xmin(nei_info, ix_f(ix), iy_f(iy), offset);
    }
    long long neiFine2(const cubism::BlockInfo &nei_info, const int ix,
                       const int iy, const int offset = 0) const override {
      return Xmin(nei_info, ix_f(ix), iy_f(iy) + 1, offset);
    }
    long long Zchild(const cubism::BlockInfo &nei_info, const int ix,
                     const int iy) const override {
      return nei_info.Zchild[0][int(iy >= BSY_ / 2)][0];
    }
  };
  class YbaseIndexer : public EdgeCellIndexer {
  public:
    YbaseIndexer(const ExpAMRSolver &pSolver) : EdgeCellIndexer(pSolver) {}
    double taylorSign(const int ix, const int iy) const override {
      return ix % 2 == 0 ? -1. : 1.;
    }
    bool isBD(const int ix, const int iy) const override {
      return ix == BSX_ - 1 || ix == BSX_ / 2 - 1;
    }
    bool isFD(const int ix, const int iy) const override {
      return ix == 0 || ix == BSX_ / 2;
    }
    long long Nei(const cubism::BlockInfo &info, const int ix, const int iy,
                  const int dist) const override {
      return This(info, ix + dist, iy);
    }
  };
  class YminIndexer : public YbaseIndexer {
  public:
    YminIndexer(const ExpAMRSolver &pSolver) : YbaseIndexer(pSolver) {}
    long long neiUnif(const cubism::BlockInfo &nei_info, const int ix,
                      const int iy) const override {
      return Ymax(nei_info, ix, iy);
    }
    long long neiInward(const cubism::BlockInfo &info, const int ix,
                        const int iy) const override {
      return This(info, ix, iy + 1);
    }
    int iy_c(const cubism::BlockInfo &info, const int iy) const override {
      return BSY_ - 1;
    }
    long long neiFine1(const cubism::BlockInfo &nei_info, const int ix,
                       const int iy, const int offset = 0) const override {
      return Ymax(nei_info, ix_f(ix), iy_f(iy), offset);
    }
    long long neiFine2(const cubism::BlockInfo &nei_info, const int ix,
                       const int iy, const int offset = 0) const override {
      return Ymax(nei_info, ix_f(ix) + 1, iy_f(iy), offset);
    }
    long long Zchild(const cubism::BlockInfo &nei_info, const int ix,
                     const int iy) const override {
      return nei_info.Zchild[int(ix >= BSX_ / 2)][1][0];
    }
  };
  class YmaxIndexer : public YbaseIndexer {
  public:
    YmaxIndexer(const ExpAMRSolver &pSolver) : YbaseIndexer(pSolver) {}
    long long neiUnif(const cubism::BlockInfo &nei_info, const int ix,
                      const int iy) const override {
      return Ymin(nei_info, ix, iy);
    }
    long long neiInward(const cubism::BlockInfo &info, const int ix,
                        const int iy) const override {
      return This(info, ix, iy - 1);
    }
    int iy_c(const cubism::BlockInfo &info, const int iy) const override {
      return 0;
    }
    long long neiFine1(const cubism::BlockInfo &nei_info, const int ix,
                       const int iy, const int offset = 0) const override {
      return Ymin(nei_info, ix_f(ix), iy_f(iy), offset);
    }
    long long neiFine2(const cubism::BlockInfo &nei_info, const int ix,
                       const int iy, const int offset = 0) const override {
      return Ymin(nei_info, ix_f(ix) + 1, iy_f(iy), offset);
    }
    long long Zchild(const cubism::BlockInfo &nei_info, const int ix,
                     const int iy) const override {
      return nei_info.Zchild[int(ix >= BSX_ / 2)][0][0];
    }
  };
  CellIndexer GenericCell;
  XminIndexer XminCell;
  XmaxIndexer XmaxCell;
  YminIndexer YminCell;
  YmaxIndexer YmaxCell;
  std::array<const EdgeCellIndexer *, 4> edgeIndexers;
  std::array<std::pair<long long, double>, 3> D1(const cubism::BlockInfo &info,
                                                 const EdgeCellIndexer &indexer,
                                                 const int ix,
                                                 const int iy) const {
    if (indexer.isBD(ix, iy))
      return {{{indexer.Nei(info, ix, iy, -2), 1. / 8.},
               {indexer.Nei(info, ix, iy, -1), -1. / 2.},
               {indexer.This(info, ix, iy), 3. / 8.}}};
    else if (indexer.isFD(ix, iy))
      return {{{indexer.Nei(info, ix, iy, 2), -1. / 8.},
               {indexer.Nei(info, ix, iy, 1), 1. / 2.},
               {indexer.This(info, ix, iy), -3. / 8.}}};
    return {{{indexer.Nei(info, ix, iy, -1), -1. / 8.},
             {indexer.Nei(info, ix, iy, 1), 1. / 8.},
             {indexer.This(info, ix, iy), 0.}}};
  }
  std::array<std::pair<long long, double>, 3> D2(const cubism::BlockInfo &info,
                                                 const EdgeCellIndexer &indexer,
                                                 const int ix,
                                                 const int iy) const {
    if (indexer.isBD(ix, iy))
      return {{{indexer.Nei(info, ix, iy, -2), 1. / 32.},
               {indexer.Nei(info, ix, iy, -1), -1. / 16.},
               {indexer.This(info, ix, iy), 1. / 32.}}};
    else if (indexer.isFD(ix, iy))
      return {{{indexer.Nei(info, ix, iy, 2), 1. / 32.},
               {indexer.Nei(info, ix, iy, 1), -1. / 16.},
               {indexer.This(info, ix, iy), 1. / 32.}}};
    return {{{indexer.Nei(info, ix, iy, -1), 1. / 32.},
             {indexer.Nei(info, ix, iy, 1), 1. / 32.},
             {indexer.This(info, ix, iy), -1. / 16.}}};
  }
  void interpolate(const cubism::BlockInfo &info_c, const int ix_c,
                   const int iy_c, const cubism::BlockInfo &info_f,
                   const long long fine_close_idx, const long long fine_far_idx,
                   const double signI, const double signT,
                   const EdgeCellIndexer &indexer, SpRowInfo &row) const;
};

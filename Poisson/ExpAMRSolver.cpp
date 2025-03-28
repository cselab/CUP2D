#include <mpi.h>
#include "config.h"
#include "ExpAMRSolver.h"
using namespace cubism;
double ExpAMRSolver::getA_local(int I1, int I2) {
  int j1 = I1 / BSX_;
  int i1 = I1 % BSX_;
  int j2 = I2 / BSX_;
  int i2 = I2 % BSX_;
  if (i1 == i2 && j1 == j2)
    return 4.0;
  else if (abs(i1 - i2) + abs(j1 - j2) == 1)
    return -1.0;
  else
    return 0.0;
}
ExpAMRSolver::ExpAMRSolver(SimulationData &s)
    : sim(s), m_comm_(sim.comm), GenericCell(*this), XminCell(*this),
      XmaxCell(*this), YminCell(*this), YmaxCell(*this),
      edgeIndexers{&XminCell, &XmaxCell, &YminCell, &YmaxCell} {
  MPI_Comm_rank(m_comm_, &rank_);
  MPI_Comm_size(m_comm_, &comm_size_);
  Nblocks_xcumsum_.resize(comm_size_ + 1);
  Nrows_xcumsum_.resize(comm_size_ + 1);
  std::vector<std::vector<double>> L;
  std::vector<std::vector<double>> L_inv;
  L.resize(BLEN_);
  L_inv.resize(BLEN_);
  for (int i(0); i < BLEN_; i++) {
    L[i].resize(i + 1);
    L_inv[i].resize(i + 1);
    for (int j(0); j <= i; j++) {
      L_inv[i][j] = (i == j) ? 1. : 0.;
    }
  }
  for (int i(0); i < BLEN_; i++) {
    double s1 = 0;
    for (int k(0); k <= i - 1; k++)
      s1 += L[i][k] * L[i][k];
    L[i][i] = sqrt(getA_local(i, i) - s1);
    for (int j(i + 1); j < BLEN_; j++) {
      double s2 = 0;
      for (int k(0); k <= i - 1; k++)
        s2 += L[i][k] * L[j][k];
      L[j][i] = (getA_local(j, i) - s2) / L[i][i];
    }
  }
  for (int br(0); br < BLEN_; br++) {
    const double bsf = 1. / L[br][br];
    for (int c(0); c <= br; c++)
      L_inv[br][c] *= bsf;
    for (int wr(br + 1); wr < BLEN_; wr++) {
      const double wsf = L[wr][br];
      for (int c(0); c <= br; c++)
        L_inv[wr][c] -= (wsf * L_inv[br][c]);
    }
  }
  std::vector<double> P_inv(BLEN_ * BLEN_);
  for (int i(0); i < BLEN_; i++)
    for (int j(0); j < BLEN_; j++) {
      double aux = 0.;
      for (int k(0); k < BLEN_; k++)
        aux += (i <= k && j <= k) ? L_inv[k][i] * L_inv[k][j] : 0.;
      P_inv[i * BLEN_ + j] = -aux;
    }
  LocalLS_ = std::make_unique<LocalSpMatDnVec>(m_comm_, BSX_ * BSY_,
                                               sim.bMeanConstraint, P_inv);
}
void ExpAMRSolver::interpolate(const BlockInfo &info_c, const int ix_c,
                               const int iy_c, const BlockInfo &info_f,
                               const long long fine_close_idx,
                               const long long fine_far_idx,
                               const double signInt, const double signTaylor,
                               const EdgeCellIndexer &indexer,
                               SpRowInfo &row) const {
  const int rank_c = sim.tmp->Tree(info_c).rank();
  const int rank_f = sim.tmp->Tree(info_f).rank();
  row.mapColVal(rank_f, fine_close_idx, signInt * 2. / 3.);
  row.mapColVal(rank_f, fine_far_idx, -signInt * 1. / 5.);
  const double tf = signInt * 8. / 15.;
  row.mapColVal(rank_c, indexer.This(info_c, ix_c, iy_c), tf);
  std::array<std::pair<long long, double>, 3> D;
  D = D1(info_c, indexer, ix_c, iy_c);
  for (int i(0); i < 3; i++)
    row.mapColVal(rank_c, D[i].first, signTaylor * tf * D[i].second);
  D = D2(info_c, indexer, ix_c, iy_c);
  for (int i(0); i < 3; i++)
    row.mapColVal(rank_c, D[i].first, tf * D[i].second);
}
void ExpAMRSolver::makeFlux(const BlockInfo &rhs_info, const int ix,
                            const int iy, const BlockInfo &rhsNei,
                            const EdgeCellIndexer &indexer,
                            SpRowInfo &row) const {
  const long long sfc_idx = indexer.This(rhs_info, ix, iy);
  if (this->sim.tmp->Tree(rhsNei).Exists()) {
    const int nei_rank = sim.tmp->Tree(rhsNei).rank();
    const long long nei_idx = indexer.neiUnif(rhsNei, ix, iy);
    row.mapColVal(nei_rank, nei_idx, 1.);
    row.mapColVal(sfc_idx, -1.);
  } else if (this->sim.tmp->Tree(rhsNei).CheckCoarser()) {
    const BlockInfo &rhsNei_c =
        this->sim.tmp->getBlockInfoAll(rhs_info.level - 1, rhsNei.Zparent);
    const int ix_c = indexer.ix_c(rhs_info, ix);
    const int iy_c = indexer.iy_c(rhs_info, iy);
    const long long inward_idx = indexer.neiInward(rhs_info, ix, iy);
    const double signTaylor = indexer.taylorSign(ix, iy);
    interpolate(rhsNei_c, ix_c, iy_c, rhs_info, sfc_idx, inward_idx, 1.,
                signTaylor, indexer, row);
    row.mapColVal(sfc_idx, -1.);
  } else if (this->sim.tmp->Tree(rhsNei).CheckFiner()) {
    const BlockInfo &rhsNei_f = this->sim.tmp->getBlockInfoAll(
        rhs_info.level + 1, indexer.Zchild(rhsNei, ix, iy));
    const int nei_rank = this->sim.tmp->Tree(rhsNei_f).rank();
    long long fine_close_idx = indexer.neiFine1(rhsNei_f, ix, iy, 0);
    long long fine_far_idx = indexer.neiFine1(rhsNei_f, ix, iy, 1);
    row.mapColVal(nei_rank, fine_close_idx, 1.);
    interpolate(rhs_info, ix, iy, rhsNei_f, fine_close_idx, fine_far_idx, -1.,
                -1., indexer, row);
    fine_close_idx = indexer.neiFine2(rhsNei_f, ix, iy, 0);
    fine_far_idx = indexer.neiFine2(rhsNei_f, ix, iy, 1);
    row.mapColVal(nei_rank, fine_close_idx, 1.);
    interpolate(rhs_info, ix, iy, rhsNei_f, fine_close_idx, fine_far_idx, -1.,
                1., indexer, row);
  } else {
    throw std::runtime_error(
        "Neighbour doesn't exist, isn't coarser, nor finer...");
  }
}
void ExpAMRSolver::getMat() {
  std::array<int, 3> blocksPerDim = sim.pres->getMaxBlocks();
  sim.tmp->UpdateBlockInfoAll_States(true);
  std::vector<cubism::BlockInfo> &RhsInfo = sim.tmp->getBlocksInfo();
  const int Nblocks = RhsInfo.size();
  const int N = BSX_ * BSY_ * Nblocks;
  LocalLS_->reserve(N);
  const long long Nblocks_long = Nblocks;
  MPI_Allgather(&Nblocks_long, 1, MPI_LONG_LONG, Nblocks_xcumsum_.data(), 1,
                MPI_LONG_LONG, m_comm_);
  for (int i(Nblocks_xcumsum_.size() - 1); i > 0; i--) {
    Nblocks_xcumsum_[i] = Nblocks_xcumsum_[i - 1];
  }
  Nblocks_xcumsum_[0] = 0;
  Nrows_xcumsum_[0] = 0;
  for (size_t i(1); i < Nblocks_xcumsum_.size(); i++) {
    Nblocks_xcumsum_[i] += Nblocks_xcumsum_[i - 1];
    Nrows_xcumsum_[i] = BLEN_ * Nblocks_xcumsum_[i];
  }
  for (int i = 0; i < Nblocks; i++) {
    const BlockInfo &rhs_info = RhsInfo[i];
    const int aux = 1 << rhs_info.level;
    const int MAX_X_BLOCKS = blocksPerDim[0] * aux - 1;
    const int MAX_Y_BLOCKS = blocksPerDim[1] * aux - 1;
    std::array<bool, 4> isBoundary;
    isBoundary[0] = (rhs_info.index[0] == 0);
    isBoundary[1] = (rhs_info.index[0] == MAX_X_BLOCKS);
    isBoundary[2] = (rhs_info.index[1] == 0);
    isBoundary[3] = (rhs_info.index[1] == MAX_Y_BLOCKS);
    std::array<bool, 2> isPeriodic;
    isPeriodic[0] = (cubismBCX == periodic);
    isPeriodic[1] = (cubismBCY == periodic);
    std::array<long long, 4> Z;
    Z[0] = rhs_info.Znei[1 - 1][1][1];
    Z[1] = rhs_info.Znei[1 + 1][1][1];
    Z[2] = rhs_info.Znei[1][1 - 1][1];
    Z[3] = rhs_info.Znei[1][1 + 1][1];
    std::array<const BlockInfo *, 4> rhsNei;
    rhsNei[0] = &(this->sim.tmp->getBlockInfoAll(rhs_info.level, Z[0]));
    rhsNei[1] = &(this->sim.tmp->getBlockInfoAll(rhs_info.level, Z[1]));
    rhsNei[2] = &(this->sim.tmp->getBlockInfoAll(rhs_info.level, Z[2]));
    rhsNei[3] = &(this->sim.tmp->getBlockInfoAll(rhs_info.level, Z[3]));
    if (sim.bMeanConstraint && rhs_info.index[0] == 0 &&
        rhs_info.index[1] == 0 && rhs_info.index[2] == 0)
      LocalLS_->set_bMeanRow(GenericCell.This(rhs_info, 0, 0) -
                             Nrows_xcumsum_[rank_]);
    for (int iy = 0; iy < BSY_; iy++)
      for (int ix = 0; ix < BSX_; ix++) {
        const long long sfc_idx = GenericCell.This(rhs_info, ix, iy);
        if ((ix > 0 && ix < BSX_ - 1) && (iy > 0 && iy < BSY_ - 1)) {
          LocalLS_->cooPushBackVal(1, sfc_idx,
                                   GenericCell.This(rhs_info, ix, iy - 1));
          LocalLS_->cooPushBackVal(1, sfc_idx,
                                   GenericCell.This(rhs_info, ix - 1, iy));
          LocalLS_->cooPushBackVal(-4, sfc_idx, sfc_idx);
          LocalLS_->cooPushBackVal(1, sfc_idx,
                                   GenericCell.This(rhs_info, ix + 1, iy));
          LocalLS_->cooPushBackVal(1, sfc_idx,
                                   GenericCell.This(rhs_info, ix, iy + 1));
        } else {
          std::array<bool, 4> validNei;
          validNei[0] = GenericCell.validXm(ix, iy);
          validNei[1] = GenericCell.validXp(ix, iy);
          validNei[2] = GenericCell.validYm(ix, iy);
          validNei[3] = GenericCell.validYp(ix, iy);
          std::array<long long, 4> idxNei;
          idxNei[0] = GenericCell.This(rhs_info, ix - 1, iy);
          idxNei[1] = GenericCell.This(rhs_info, ix + 1, iy);
          idxNei[2] = GenericCell.This(rhs_info, ix, iy - 1);
          idxNei[3] = GenericCell.This(rhs_info, ix, iy + 1);
          SpRowInfo row(sim.tmp->Tree(rhs_info).rank(), sfc_idx, 8);
          for (int j(0); j < 4; j++) {
            if (validNei[j]) {
              row.mapColVal(idxNei[j], 1);
              row.mapColVal(sfc_idx, -1);
            } else if (!isBoundary[j] || (isBoundary[j] && isPeriodic[j / 2]))
              this->makeFlux(rhs_info, ix, iy, *rhsNei[j], *edgeIndexers[j],
                             row);
          }
          LocalLS_->cooPushBackRow(row);
        }
      }
  }
  LocalLS_->make(Nrows_xcumsum_);
}
void ExpAMRSolver::getVec() {
  std::vector<cubism::BlockInfo> &RhsInfo = sim.tmp->getBlocksInfo();
  std::vector<cubism::BlockInfo> &zInfo = sim.pres->getBlocksInfo();
  const int Nblocks = RhsInfo.size();
  std::vector<double> &x = LocalLS_->get_x();
  std::vector<double> &b = LocalLS_->get_b();
  std::vector<double> &h2 = LocalLS_->get_h2();
  const long long shift = -Nrows_xcumsum_[rank_];
#pragma omp parallel for
  for (int i = 0; i < Nblocks; i++) {
    const BlockInfo &rhs_info = RhsInfo[i];
    const ScalarBlock &__restrict__ rhs = *(ScalarBlock *)RhsInfo[i].ptrBlock;
    const ScalarBlock &__restrict__ p = *(ScalarBlock *)zInfo[i].ptrBlock;
    h2[i] = RhsInfo[i].h * RhsInfo[i].h;
    for (int iy = 0; iy < BSY_; iy++)
      for (int ix = 0; ix < BSX_; ix++) {
        const long long sfc_loc = GenericCell.This(rhs_info, ix, iy) + shift;
        if (sim.bMeanConstraint && rhs_info.index[0] == 0 &&
            rhs_info.index[1] == 0 && rhs_info.index[2] == 0 && ix == 0 &&
            iy == 0)
          b[sfc_loc] = 0.;
        else
          b[sfc_loc] = rhs(ix, iy).s;
        x[sfc_loc] = p(ix, iy).s;
      }
  }
}
void ExpAMRSolver::solve(const ScalarGrid *input, ScalarGrid *const output) {
  if (rank_ == 0) {
    if (sim.verbose)
      std::cout << "--------------------- Calling on ExpAMRSolver.solve() "
                   "------------------------\n";
    else
      std::cout << '\n';
  }
  const double max_error = this->sim.step < 10 ? 0.0 : sim.PoissonTol;
  const double max_rel_error = this->sim.step < 10 ? 0.0 : sim.PoissonTolRel;
  const int max_restarts = this->sim.step < 10 ? 100 : sim.maxPoissonRestarts;
  if (sim.pres->UpdateFluxCorrection) {
    sim.pres->UpdateFluxCorrection = false;
    this->getMat();
    this->getVec();
    LocalLS_->solveWithUpdate(max_error, max_rel_error, max_restarts);
  } else {
    this->getVec();
    LocalLS_->solveNoUpdate(max_error, max_rel_error, max_restarts);
  }
  std::vector<cubism::BlockInfo> &zInfo = sim.pres->getBlocksInfo();
  const int Nblocks = zInfo.size();
  const std::vector<double> &x = LocalLS_->get_x();
  double avg = 0;
  double avg1 = 0;
#pragma omp parallel for reduction(+ : avg, avg1)
  for (int i = 0; i < Nblocks; i++) {
    ScalarBlock &P = *(ScalarBlock *)zInfo[i].ptrBlock;
    const double vv = zInfo[i].h * zInfo[i].h;
    for (int iy = 0; iy < BSY_; iy++)
      for (int ix = 0; ix < BSX_; ix++) {
        P(ix, iy).s = x[i * BSX_ * BSY_ + iy * BSX_ + ix];
        avg += P(ix, iy).s * vv;
        avg1 += vv;
      }
  }
  double quantities[2] = {avg, avg1};
  MPI_Allreduce(MPI_IN_PLACE, &quantities, 2, MPI_DOUBLE, MPI_SUM, m_comm_);
  avg = quantities[0];
  avg1 = quantities[1];
  avg = avg / avg1;
#pragma omp parallel for
  for (int i = 0; i < Nblocks; i++) {
    ScalarBlock &P = *(ScalarBlock *)zInfo[i].ptrBlock;
    for (int iy = 0; iy < BSY_; iy++)
      for (int ix = 0; ix < BSX_; ix++)
        P(ix, iy).s += -avg;
  }
}

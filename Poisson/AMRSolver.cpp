#include "config.h"
#include "AMRSolver.h"
using namespace cubism;
void AMRSolver::getZ(Real *input, BlockInfo &zInfo) {
  const int BSX = VectorBlock::sizeX;
  const int BSY = VectorBlock::sizeY;
  const int N = BSX * BSY;
  {
    const int bpdx = sim.chi->getMaxBlocks()[0];
    const int bpdy = sim.chi->getMaxBlocks()[1];
    const int tid = omp_get_thread_num();
    {
      const Real c11 = input[(BSX - 1) + BSX * (BSY - 1)];
      const Real c10 = input[(BSX - 1) + BSX * (0)];
      const Real c01 = input[(0) + BSX * (BSY - 1)];
      const Real c00 = input[(0) + BSX * (0)];
      bool plus_x = zInfo.index[0] > bpdx * (1 << zInfo.level) / 2 - 1;
      bool plus_y = zInfo.index[1] > bpdy * (1 << zInfo.level) / 2 - 1;
      if (std::fabs(std::fabs(c11 + c10) - std::fabs(c01 + c00)) > 1e-14)
        plus_x = std::fabs(c11 + c10) > std::fabs(c01 + c00);
      if (std::fabs(std::fabs(c11 + c01) - std::fabs(c10 + c00)) > 1e-14)
        plus_y = std::fabs(c11 + c01) > std::fabs(c10 + c00);
      const int base_x = plus_x ? 0 : BSX - 1;
      const int base_y = plus_y ? 0 : BSY - 1;
      const int sign_x = plus_x ? 1 : -1;
      const int sign_y = plus_y ? 1 : -1;
      for (int I = 0; I < N; I++) {
        Real rhs = 0.0;
        for (size_t jj = 0; jj < L_row[tid][I].size(); jj++) {
          const int J = L_row[tid][I][jj].first;
          const int iy = base_y + sign_y * J / BSX;
          const int ix = base_x + sign_x * J % BSX;
          rhs += L_row[tid][I][jj].second * input[iy * BSX + ix];
        }
        const int iy = base_y + sign_y * I / BSX;
        const int ix = base_x + sign_x * I % BSX;
        input[iy * BSX + ix] = (input[iy * BSX + ix] - rhs) * Ld[tid][I];
      }
      for (int I = N - 1; I >= 0; I--) {
        Real rhs = 0.0;
        for (size_t jj = 0; jj < L_col[tid][I].size(); jj++) {
          const int J = L_col[tid][I][jj].first;
          const int iy = base_y + sign_y * J / BSX;
          const int ix = base_x + sign_x * J % BSX;
          rhs += -L_col[tid][I][jj].second * input[iy * BSX + ix];
        }
        const int iy = base_y + sign_y * I / BSX;
        const int ix = base_x + sign_x * I % BSX;
        input[iy * BSX + ix] = (rhs - input[iy * BSX + ix]) * Ld[tid][I];
      }
    }
  }
}
Real AMRSolver::getA_local(const int I1, const int I2) {
  const int BSX = VectorBlock::sizeX;
  const int j1 = I1 / BSX;
  const int i1 = I1 % BSX;
  const int j2 = I2 / BSX;
  const int i2 = I2 % BSX;
  if (i1 == i2 && j1 == j2) {
    return 4.0;
  } else if (abs(i1 - i2) + abs(j1 - j2) == 1) {
    return -1.0;
  } else {
    return 0.0;
  }
}
AMRSolver::AMRSolver(SimulationData &ss) : sim(ss), Get_LHS(ss) {
  const int BSX = VectorBlock::sizeX;
  const int BSY = VectorBlock::sizeY;
  const int N = BSX * BSY;
  std::vector<std::vector<Real>> L(N);
  for (int i = 0; i < N; i++) {
    L[i].resize(i + 1);
  }
  for (int i = 0; i < N; i++) {
    Real s1 = 0;
    for (int k = 0; k <= i - 1; k++)
      s1 += L[i][k] * L[i][k];
    L[i][i] = sqrt(getA_local(i, i) - s1);
    for (int j = i + 1; j < N; j++) {
      Real s2 = 0;
      for (int k = 0; k <= i - 1; k++)
        s2 += L[i][k] * L[j][k];
      L[j][i] = (getA_local(j, i) - s2) / L[i][i];
    }
  }
  L_row.resize(omp_get_max_threads());
  L_col.resize(omp_get_max_threads());
  Ld.resize(omp_get_max_threads());
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    L_row[tid].resize(N);
    L_col[tid].resize(N);
    for (int i = 0; i < N; i++) {
      Ld[tid].push_back(1.0 / L[i][i]);
      for (int j = 0; j < i; j++) {
        if (abs(L[i][j]) > 1e-10)
          L_row[tid][i].push_back({j, L[i][j]});
      }
    }
    for (int j = 0; j < N; j++)
      for (int i = j + 1; i < N; i++) {
        if (abs(L[i][j]) > 1e-10)
          L_col[tid][j].push_back({i, L[i][j]});
      }
  }
}
void AMRSolver::solve(const ScalarGrid *input, ScalarGrid *const output) {
  if (input != sim.tmp || output != sim.pres)
    throw std::invalid_argument(
        "AMRSolver hardcoded to sim.tmp and sim.pres for now");
  const auto &AxInfo = input->getBlocksInfo();
  const auto &zInfo = output->getBlocksInfo();
  const size_t Nblocks = zInfo.size();
  const int BSX = VectorBlock::sizeX;
  const int BSY = VectorBlock::sizeY;
  const size_t N = BSX * BSY * Nblocks;
  const Real eps = 1e-100;
  const Real max_error = sim.step < 10 ? 0.0 : sim.PoissonTol;
  const Real max_rel_error = sim.step < 10 ? 0.0 : sim.PoissonTolRel;
  const int max_restarts = sim.step < 10 ? 100 : sim.maxPoissonRestarts;
  bool serious_breakdown = false;
  bool useXopt = false;
  int restarts = 0;
  Real min_norm = 1e50;
  Real norm_1 = 0.0;
  Real norm_2 = 0.0;
  const MPI_Comm m_comm = sim.chi->getWorldComm();
  const bool verbose = sim.rank == 0 && !sim.muteAll;
  phat.resize(N);
  rhat.resize(N);
  shat.resize(N);
  what.resize(N);
  zhat.resize(N);
  qhat.resize(N);
  s.resize(N);
  w.resize(N);
  z.resize(N);
  t.resize(N);
  v.resize(N);
  q.resize(N);
  r.resize(N);
  y.resize(N);
  x.resize(N);
  r0.resize(N);
  b.resize(N);
  x_opt.resize(N);
#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++) {
    ScalarBlock &__restrict__ rhs = *(ScalarBlock *)AxInfo[i].ptrBlock;
    const ScalarBlock &__restrict__ zz = *(ScalarBlock *)zInfo[i].ptrBlock;
    if (sim.bMeanConstraint == 1)
      if (isCorner(AxInfo[i]))
        rhs(0, 0).s = 0.0;
    for (int iy = 0; iy < BSY; iy++)
      for (int ix = 0; ix < BSX; ix++) {
        const int j = i * BSX * BSY + iy * BSX + ix;
        b[j] = rhs(ix, iy).s;
        r[j] = rhs(ix, iy).s;
        x[j] = zz(ix, iy).s;
      }
  }
  _lhs(x, r0);
#pragma omp parallel for
  for (size_t i = 0; i < N; i++) {
    r0[i] = r[i] - r0[i];
    r[i] = r0[i];
  }
  _preconditioner(r0, rhat);
  _lhs(rhat, w);
  _preconditioner(w, what);
  _lhs(what, t);
  Real alpha = 0.0;
  Real norm = 0.0;
  Real beta = 0.0;
  Real omega = 0.0;
  Real r0r_prev;
  {
    Real temp0 = 0.0;
    Real temp1 = 0.0;
#pragma omp parallel for reduction(+ : temp0, temp1, norm)
    for (size_t j = 0; j < N; j++) {
      temp0 += r0[j] * r0[j];
      temp1 += r0[j] * w[j];
      norm += r0[j] * r0[j];
    }
    Real temporary[3] = {temp0, temp1, norm};
    MPI_Allreduce(MPI_IN_PLACE, temporary, 3, MPI_Real, MPI_SUM, m_comm);
    alpha = temporary[0] / (temporary[1] + eps);
    r0r_prev = temporary[0];
    norm = std::sqrt(temporary[2]);
    if (verbose)
      std::cout << "[Poisson solver]: initial error norm:" << norm << "\n";
  }
  const Real init_norm = norm;
  int k;
  for (k = 0; k < sim.maxPoissonIterations; k++) {
    Real qy = 0.0;
    Real yy = 0.0;
    if (k % 50 != 0) {
#pragma omp parallel for reduction(+ : qy, yy)
      for (size_t j = 0; j < N; j++) {
        phat[j] = rhat[j] + beta * (phat[j] - omega * shat[j]);
        s[j] = w[j] + beta * (s[j] - omega * z[j]);
        shat[j] = what[j] + beta * (shat[j] - omega * zhat[j]);
        z[j] = t[j] + beta * (z[j] - omega * v[j]);
        q[j] = r[j] - alpha * s[j];
        qhat[j] = rhat[j] - alpha * shat[j];
        y[j] = w[j] - alpha * z[j];
        qy += q[j] * y[j];
        yy += y[j] * y[j];
      }
    } else {
#pragma omp parallel for
      for (size_t j = 0; j < N; j++) {
        phat[j] = rhat[j] + beta * (phat[j] - omega * shat[j]);
      }
      _lhs(phat, s);
      _preconditioner(s, shat);
      _lhs(shat, z);
#pragma omp parallel for reduction(+ : qy, yy)
      for (size_t j = 0; j < N; j++) {
        q[j] = r[j] - alpha * s[j];
        qhat[j] = rhat[j] - alpha * shat[j];
        y[j] = w[j] - alpha * z[j];
        qy += q[j] * y[j];
        yy += y[j] * y[j];
      }
    }
    MPI_Request request;
    Real quantities[7];
    quantities[0] = qy;
    quantities[1] = yy;
    MPI_Iallreduce(MPI_IN_PLACE, &quantities, 2, MPI_Real, MPI_SUM, m_comm,
                   &request);
    _preconditioner(z, zhat);
    _lhs(zhat, v);
    MPI_Waitall(1, &request, MPI_STATUSES_IGNORE);
    qy = quantities[0];
    yy = quantities[1];
    omega = qy / (yy + eps);
    Real r0r = 0.0;
    Real r0w = 0.0;
    Real r0s = 0.0;
    Real r0z = 0.0;
    norm = 0.0;
    norm_1 = 0.0;
    norm_2 = 0.0;
    if (k % 50 != 0) {
#pragma omp parallel for reduction(+ : r0r, r0w, r0s, r0z, norm_1, norm_2, norm)
      for (size_t j = 0; j < N; j++) {
        x[j] = x[j] + alpha * phat[j] + omega * qhat[j];
        r[j] = q[j] - omega * y[j];
        rhat[j] = qhat[j] - omega * (what[j] - alpha * zhat[j]);
        w[j] = y[j] - omega * (t[j] - alpha * v[j]);
        r0r += r0[j] * r[j];
        r0w += r0[j] * w[j];
        r0s += r0[j] * s[j];
        r0z += r0[j] * z[j];
        norm += r[j] * r[j];
        norm_1 += r[j] * r[j];
        norm_2 += r0[j] * r0[j];
      }
    } else {
#pragma omp parallel for
      for (size_t j = 0; j < N; j++) {
        x[j] = x[j] + alpha * phat[j] + omega * qhat[j];
      }
      _lhs(x, r);
#pragma omp parallel for
      for (size_t j = 0; j < N; j++) {
        r[j] = b[j] - r[j];
      }
      _preconditioner(r, rhat);
      _lhs(rhat, w);
#pragma omp parallel for reduction(+ : r0r, r0w, r0s, r0z, norm_1, norm_2, norm)
      for (size_t j = 0; j < N; j++) {
        r0r += r0[j] * r[j];
        r0w += r0[j] * w[j];
        r0s += r0[j] * s[j];
        r0z += r0[j] * z[j];
        norm += r[j] * r[j];
        norm_1 += r[j] * r[j];
        norm_2 += r0[j] * r0[j];
      }
    }
    quantities[0] = r0r;
    quantities[1] = r0w;
    quantities[2] = r0s;
    quantities[3] = r0z;
    quantities[4] = norm_1;
    quantities[5] = norm_2;
    quantities[6] = norm;
    MPI_Iallreduce(MPI_IN_PLACE, &quantities, 7, MPI_Real, MPI_SUM, m_comm,
                   &request);
    _preconditioner(w, what);
    _lhs(what, t);
    MPI_Waitall(1, &request, MPI_STATUSES_IGNORE);
    r0r = quantities[0];
    r0w = quantities[1];
    r0s = quantities[2];
    r0z = quantities[3];
    norm_1 = quantities[4];
    norm_2 = quantities[5];
    norm = std::sqrt(quantities[6]);
    beta = alpha / (omega + eps) * r0r / (r0r_prev + eps);
    alpha = r0r / (r0w + beta * r0s - beta * omega * r0z);
    Real alphat = 1.0 / (omega + eps) + r0w / (r0r + eps) -
                  beta * omega * r0z / (r0r + eps);
    alphat = 1.0 / (alphat + eps);
    if (std::fabs(alphat) < 10 * std::fabs(alpha))
      alpha = alphat;
    r0r_prev = r0r;
    serious_breakdown = r0r * r0r < 1e-16 * norm_1 * norm_2;
    if (serious_breakdown && restarts < max_restarts) {
      restarts++;
      if (verbose)
        std::cout << "  [Poisson solver]: Restart at iteration: " << k
                  << " norm: " << norm << std::endl;
#pragma omp parallel for
      for (size_t i = 0; i < N; i++)
        r0[i] = r[i];
      _preconditioner(r0, rhat);
      _lhs(rhat, w);
      alpha = 0.0;
      Real temp0 = 0.0;
      Real temp1 = 0.0;
#pragma omp parallel for reduction(+ : temp0, temp1)
      for (size_t j = 0; j < N; j++) {
        temp0 += r0[j] * r0[j];
        temp1 += r0[j] * w[j];
      }
      MPI_Request request2;
      Real temporary[2] = {temp0, temp1};
      MPI_Iallreduce(MPI_IN_PLACE, temporary, 2, MPI_Real, MPI_SUM, m_comm,
                     &request2);
      _preconditioner(w, what);
      _lhs(what, t);
      MPI_Waitall(1, &request2, MPI_STATUSES_IGNORE);
      alpha = temporary[0] / (temporary[1] + eps);
      r0r_prev = temporary[0];
      beta = 0.0;
      omega = 0.0;
    }
    if (norm < min_norm) {
      useXopt = true;
      min_norm = norm;
#pragma omp parallel for
      for (size_t i = 0; i < N; i++)
        x_opt[i] = x[i];
    }
    if (norm < max_error || norm / (init_norm + eps) < max_rel_error) {
      if (verbose)
        std::cout << "  [Poisson solver]: Converged after " << k
                  << " iterations.\n";
      break;
    }
  }
  if (verbose) {
    std::cout << " Error norm (relative) = " << min_norm << "/" << max_error
              << std::endl;
  }
  Real *solution = useXopt ? x_opt.data() : x.data();
#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++) {
    ScalarBlock &P = *(ScalarBlock *)zInfo[i].ptrBlock;
    for (int iy = 0; iy < BSY; iy++)
      for (int ix = 0; ix < BSX; ix++) {
        P(ix, iy).s = solution[i * BSX * BSY + iy * BSX + ix];
      }
  }
}

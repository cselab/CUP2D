// Based on Guido Novati's FFTW Poisson solver.
#include "FFTW.h"

#if CUP2D_FFTW
#include "../SimulationData.h"
#include <Cubism/Grid.hh>
#include <fftw3.h>

template <typename T, typename Plan>
static bool fftwInit(int numThreads, int MX, int MY, Plan *fwd,
                     Plan *bwd, T **buf)
{
  if constexpr (std::is_same_v<T, float>) {
    if (fftwf_init_threads() == 0)
      return false;
    fftwf_plan_with_nthreads(numThreads);
    *buf = fftwf_alloc_real(MY * MX);
    *fwd = fftwf_plan_r2r_2d(MY, MX, *buf, *buf, FFTW_REDFT10, FFTW_REDFT10, FFTW_MEASURE);
    *bwd = fftwf_plan_r2r_2d(MY, MX, *buf, *buf, FFTW_REDFT01, FFTW_REDFT01, FFTW_MEASURE);
  } else if constexpr (std::is_same_v<T, double>) {
    if (fftw_init_threads() == 0)
      return false;
    fftw_plan_with_nthreads(numThreads);
    *buf = fftw_alloc_real(MY * MX);
    *fwd = fftw_plan_r2r_2d(MY, MX, *buf, *buf, FFTW_REDFT10, FFTW_REDFT10, FFTW_MEASURE);
    *bwd = fftw_plan_r2r_2d(MY, MX, *buf, *buf, FFTW_REDFT01, FFTW_REDFT01, FFTW_MEASURE);
  } else if constexpr (std::is_same_v<T, long double>) {
    if (fftwl_init_threads() == 0)
      return false;
    fftwl_plan_with_nthreads(numThreads);
    *buf = fftwl_alloc_real(MY * MX);
    *fwd = fftwl_plan_r2r_2d(MY, MX, *buf, *buf, FFTW_REDFT10, FFTW_REDFT10, FFTW_MEASURE);
    *bwd = fftwl_plan_r2r_2d(MY, MX, *buf, *buf, FFTW_REDFT01, FFTW_REDFT01, FFTW_MEASURE);
  } else {
    throw std::runtime_error("type not supported by FFTW");
  }
  return true;
}

template <typename T, typename Plan>
static void fftwExecute(Plan plan)
{
  if constexpr (std::is_same_v<T, float>)
    fftwf_execute(plan);
  else if constexpr (std::is_same_v<T, double>)
    fftw_execute(plan);
  else if constexpr (std::is_same_v<T, long double>)
    fftwl_execute(plan);
  else
    throw std::runtime_error("unsupported type");
}

template <typename T, typename Plan>
static void fftwCleanup(Plan fwd, Plan bwd, Real *buffer)
{
  if constexpr (std::is_same_v<Real, float>) {
    fftwf_cleanup_threads();
    fftwf_destroy_plan(fwd);
    fftwf_destroy_plan(bwd);
    fftwf_free(buffer);
  } else if constexpr (std::is_same_v<Real, double>) {
    fftw_cleanup_threads();
    fftw_destroy_plan(fwd);
    fftw_destroy_plan(bwd);
    fftw_free(buffer);
  } else if constexpr (std::is_same_v<Real, long double>) {
    fftwl_cleanup_threads();
    fftwl_destroy_plan(fwd);
    fftwl_destroy_plan(bwd);
    fftwl_free(buffer);
  } else {
    throw std::runtime_error("unsupported type");
  }
}

namespace {

template <typename T> struct FFTWPlan { using type = fftw_plan; };
template <> struct FFTWPlan<float> { using type = fftwf_plan; };
template <> struct FFTWPlan<long double> { using type = fftwl_plan; };

struct FFTWCommon
{
  FFTWCommon(SimulationData& _s);

  SimulationData& s;
  FFTWPlan<Real>::type fwd{};
  FFTWPlan<Real>::type bwd{};
};

}  // anonymous namespace

FFTWCommon::FFTWCommon(SimulationData& _s) : s{_s}
{
  if (s.levelMax > 1) {
    throw std::runtime_error("FFT solver only works for uniform grids "
                             "(levelMax == numLevels == 1)");
  }

  int size;
  MPI_Comm_size(s.comm, &size);
  if (size > 1) {
    // Because of the space-filling curve. It could easily perform 2, 4 and
    // similar number of ranks, as long as each rank has a rectangular
    // subdomain of same size.
    throw std::runtime_error("FFT solver supports only 1 rank");
  }
}

class FFTWDirichletImpl : public FFTWCommon
{
public:
  FFTWDirichletImpl(SimulationData& s, Real tol) : FFTWCommon{s}, tol_{tol}
  {
    const auto cells = s.vel->getMaxMostRefinedCells();
    MX_ = (int)cells[0];
    MY_ = (int)cells[1];
    assert(cells[2] == 1);

    cosCoefX_ = new(std::align_val_t{64}) Real[MX_];
    cosCoefY_ = new(std::align_val_t{64}) Real[MY_];
    if ((uintptr_t)cosCoefX_ % 64 != 0 || (uintptr_t)cosCoefY_ % 64 != 0)
      throw std::runtime_error("aligned new[] failed");
    for (int i = 0; i < MX_; ++i)
      cosCoefX_[i] = std::cos(2 * M_PI / MX_ * i);
    for (int j = 0; j < MY_; ++j)
      cosCoefY_[j] = std::cos(2 * M_PI / MY_ * j);

    const int numThreads = omp_get_max_threads();
    if (!fftwInit(numThreads, MX_, MY_, &fwd, &bwd, &buffer_))
      throw std::runtime_error("fftw_init_threads() failed");
  }

  ~FFTWDirichletImpl()
  {
    fftwCleanup<Real>(fwd, bwd, buffer_);

    // https://stackoverflow.com/questions/53922209/how-to-invoke-aligned-new-delete-properly
    operator delete(cosCoefY_, std::align_val_t{64});
    operator delete(cosCoefX_, std::align_val_t{64});
  }

  void solve(const ScalarGrid *input, ScalarGrid * const output)
  {
    static_assert(sizeof(ScalarElement) == sizeof(Real));
    input->copyToUniformNoInterpolation((ScalarElement *)buffer_);

    fftwExecute<Real>(fwd);
    _solve();
    fftwExecute<Real>(bwd);

    output->copyFromMatrix((const ScalarElement *)buffer_);
  }

  // BALANCE TWO PROBLEMS:
  // - if only grid consistent odd DOF and even DOF do not 'talk' to each others
  // - if only spectral then nont really div free
  // COMPROMISE: define a tolerance that balances two effects
  void _solve() const
  {
    const Real waveFactX = M_PI / MX_;
    const Real waveFactY = M_PI / MY_;
    const Real tol = tol_;
    const Real norm_factor = 0.25 / (MX_ * MY_);
    Real * __restrict__ const inOut = buffer_;

    const Real * const cosCoefX = (const Real *)__builtin_assume_aligned(cosCoefX_, 64);
    const Real * const cosCoefY = (const Real *)__builtin_assume_aligned(cosCoefY_, 64);

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < MY_; ++j)
    for (int i = 0; i < MX_; ++i) {
      const Real rkx = (i + (Real).5) * waveFactX;
      const Real rky = (j + (Real).5) * waveFactY;
      const Real denomFD = 1 - 0.5 * (cosCoefX[i] + cosCoefY[j]);
      const Real denomSP = rkx * rkx + rky * rky;
      inOut[j * MX_ + i] *= -norm_factor / ((1 - tol) * denomFD + tol * denomSP);
    }
    inOut[0] = 0;
  }

private:
  int MX_;
  int MY_;
  Real tol_;
  Real *cosCoefX_ = nullptr;
  Real *cosCoefY_ = nullptr;
  Real *buffer_ = nullptr;
};

FFTWDirichlet::FFTWDirichlet(SimulationData& s, Real tol) :
  impl_{std::make_unique<FFTWDirichletImpl>(s, tol)}
{ }

FFTWDirichlet::~FFTWDirichlet() = default;

void FFTWDirichlet::solve(const ScalarGrid *input, ScalarGrid *output)
{
  impl_->solve(input, output);
}

#else   // before: CUP2D_FFTW, after: !CUP2D_FFTW

struct FFTWDirichletImpl { };

FFTWDirichlet::FFTWDirichlet(SimulationData& s, Real tol)
{
  throw std::runtime_error("CubismUP2D compiled without FFTW support");
}

FFTWDirichlet::~FFTWDirichlet() = default;

void FFTWDirichlet::solve(const ScalarGrid *, ScalarGrid *) { }

#endif  // !CUP2D_FFTW

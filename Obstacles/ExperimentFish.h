

#pragma once
#include "Fish.h"
#include "FishData.h"
#include "FishUtilities.h"

class ExperimentFish : public Fish {
  const Real timeStart;

public:
  ExperimentFish(SimulationData &s, cubism::ArgumentParser &p, Real C[2]);

  void updatePosition(Real dt) override;
  void updateVelocity(Real dt) override;
};

class ExperimentDataFish : public FishData {

  std::vector<std::vector<Real>> midlineData;
  std::vector<std::vector<Real>> centerOfMassData;

  const Real timeStart, dtDataset;
  Real tLast = 0.0, tNext = 0.0;
  size_t idxLast = 0, idxNext = 0;

  Schedulers::ParameterSchedulerVector<6> midlineScheduler;

public:
  Real u = 0.0, v = 0.0, omega = 0.0;

  ExperimentDataFish(Real L, Real _h, std::string path, Real _timeStart,
                     Real _dtDataset)
      : FishData(L, _h), timeStart(_timeStart), dtDataset(_dtDataset) {
    midlineData = loadFile(path + "/dummyCenterline.txt");
    centerOfMassData = loadFile(path + "/dummyCoM.txt");
    _computeWidth();
    writeMidline2File(0, "initialCheck");
  }

  std::vector<std::vector<Real>> loadFile(const std::string path);
  void computeMidline(const Real time, const Real dt) override;

  Real _width(const Real s, const Real L) override {
    const Real sb = .04 * length, st = .95 * length, wt = .01 * length,
               wh = .04 * length;
    if (s < 0 or s > L)
      return 0;
    return (s < sb
                ? std::sqrt(2 * wh * s - s * s)
                : (s < st ? wh - (wh - wt) * std::pow((s - sb) / (st - sb), 1)
                          : (wt * (L - s) / (L - st))));
  }
};

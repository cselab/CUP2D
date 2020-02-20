//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "Naca.h"
#include "FishLibrary.h"
#include "FishUtilities.h"

using namespace cubism;

class NacaData : public FishData
{
  const Real tRatio;
 public:
  NacaData(Real L, Real _h, Real _tRatio)
  : FishData(L, 0, 0, _h, 0), tRatio(_tRatio) { _computeWidth(); }

  void computeMidline(const Real time, const Real dt) override;
  Real _width(const Real s, const Real L) override
  {
    const Real a =  0.2969;
    const Real b = -0.1260;
    const Real c = -0.3516;
    const Real d =  0.2843;
    const Real e = -0.1036; // instead of -0.1015 to ensure closing end
    const Real t = tRatio*L; //NACA00{tRatio}

    if(s<0 or s>L) return 0;
    const Real p = s/L;
    return 5*t* (a*std::sqrt(p) +b*p +c*p*p +d*p*p*p + e*p*p*p*p);
  }
};

void NacaData::computeMidline(const Real t, const Real dt)
{
  rX[0] = rY[0] = vX[0] = vY[0] = 0;

  #pragma omp parallel for schedule(static)
  for(int i=1; i<Nm; ++i) {
    rY[i] = vX[i] = vY[i] = 0;
    rX[i] = rX[i-1] + std::fabs(rS[i]-rS[i-1]);
  }

  #pragma omp parallel for schedule(static)
  for(int i=1; i<Nm; ++i) {
    const Real dy = rY[i]-rY[i-1], ds = rS[i]-rS[i-1];
    const Real dx = std::sqrt(ds*ds-dy*dy);
    assert(dx>0);
    const Real dVy = vY[i]-vY[i-1];
    const Real dVx = - dy/dx * dVy; // ds^2 = dx^2+dy^2 -> ddx = -dy/dx*ddy

    rX[i] = dx;
    vX[i] = dVx;
    norX[ i-1] = -dy/ds;
    norY[ i-1] =  dx/ds;
    vNorX[i-1] = -dVy/ds;
    vNorY[i-1] =  dVx/ds;
  }

  for(int i=1; i<Nm; ++i) { rX[i] += rX[i-1]; vX[i] += vX[i-1]; }

  norX[Nm-1] = norX[Nm-2];
  norY[Nm-1] = norY[Nm-2];
  vNorX[Nm-1] = vNorX[Nm-2];
  vNorY[Nm-1] = vNorY[Nm-2];
}

Naca::Naca(SimulationData&s, ArgumentParser&p, double C[2])
  : Fish(s,p,C), Apitch(p("-Apitch").asDouble(0.0)), Fpitch(p("-Fpitch").asDouble(0.0)), time(0.0)  {
  const Real tRatio = p("-tRatio").asDouble(0.12);
  myFish = new NacaData(length, sim.getH(), tRatio);
  printf("NacaFoil Nm=%d L=%f t=%f A=%f w=%f\n",myFish->Nm, length, tRatio, Apitch, Fpitch);
}

void Naca::create(const std::vector<BlockInfo>& vInfo) {
  Fish::create(vInfo);
}

void Naca::resetAll() {
  Fish::resetAll();
}

void Naca::updateVelocity(double dt)
{
  Shape::updateVelocity(dt);

  // x velocity can be either fixed from the start (then we follow the obst op
  // pattern) or self propelled, here we do not touch it.
  time += dt;
  const Real arga = 2*M_PI*Fpitch*time;
  omega = 2*M_PI*Fpitch*Apitch*std::cos(arga); // last term is [(center of mass - rotation axis)*L]^2 according to parallel axis theorem
  u = (0.399421-0.106667)*length*std::cos(arga)*omega;
  v = -(0.399421-0.106667)*length*std::sin(arga)*omega;
}

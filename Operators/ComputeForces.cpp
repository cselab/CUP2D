#include "ComputeForces.h"
#include "../Shape.h"
#include <fstream>
#include <sstream>
#include <string>
using UDEFMAT = Real[VectorBlock::sizeY][VectorBlock::sizeX][2];
struct KernelComputeForces {
  const int big = 5;
  const int small = -4;
  KernelComputeForces(const SimulationData &s) : sim(s) {}
  const SimulationData &sim;
  cubism::StencilInfo stencil{small, small, 0, big, big, 1, true, {0, 1}};
  cubism::StencilInfo stencil2{small, small, 0, big, big, 1, true, {0}};
  const int bigg = ScalarBlock::sizeX + big - 1;
  const int stencil_start[3] = {small, small, small},
            stencil_end[3] = {big, big, big};
  const Real c0 = -137. / 60.;
  const Real c1 = 5.;
  const Real c2 = -5.;
  const Real c3 = 10. / 3.;
  const Real c4 = -5. / 4.;
  const Real c5 = 1. / 5.;
  inline bool inrange(const int i) const { return (i >= small && i < bigg); }
  const std::vector<cubism::BlockInfo> &presInfo = sim.pres->getBlocksInfo();
  void operator()(VectorLab &lab, ScalarLab &chi, const cubism::BlockInfo &info,
                  const cubism::BlockInfo &info2) const {
    VectorLab &V = lab;
    ScalarBlock &__restrict__ P =
        *(ScalarBlock *)presInfo[info.blockID].ptrBlock;
    for (const auto &_shape : sim.shapes) {
      const Shape *const shape = _shape.get();
      const std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
      const Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];
      const Real vel_norm =
          std::sqrt(shape->u * shape->u + shape->v * shape->v);
      const Real vel_unit[2] = {
          vel_norm > 0 ? (Real)shape->u / vel_norm : (Real)0,
          vel_norm > 0 ? (Real)shape->v / vel_norm : (Real)0};
      const Real NUoH = sim.nu / info.h;
      ObstacleBlock *const O = OBLOCK[info.blockID];
      if (O == nullptr)
        continue;
      assert(O->filled);
      for (size_t k = 0; k < O->n_surfPoints; ++k) {
        const int ix = O->surface[k]->ix, iy = O->surface[k]->iy;
        const std::array<Real, 2> p = info.pos<Real>(ix, iy);
        const Real normX = O->surface[k]->dchidx;
        const Real normY = O->surface[k]->dchidy;
        const Real norm = 1.0 / std::sqrt(normX * normX + normY * normY);
        const Real dx = normX * norm;
        const Real dy = normY * norm;
        Real DuDx;
        Real DuDy;
        Real DvDx;
        Real DvDy;
        {
          int x = ix;
          int y = iy;
          for (int kk = 0; kk < 5; kk++) {
            const int dxi = round(kk * dx);
            const int dyi = round(kk * dy);
            if (ix + dxi + 1 >= ScalarBlock::sizeX + big - 1 ||
                ix + dxi - 1 < small)
              continue;
            if (iy + dyi + 1 >= ScalarBlock::sizeY + big - 1 ||
                iy + dyi - 1 < small)
              continue;
            x = ix + dxi;
            y = iy + dyi;
            if (chi(x, y).s < 0.01)
              break;
          }
          const auto &l = lab;
          const int sx = normX > 0 ? +1 : -1;
          const int sy = normY > 0 ? +1 : -1;
          VectorElement dveldx;
          if (inrange(x + 5 * sx))
            dveldx = sx * (c0 * l(x, y) + c1 * l(x + sx, y) +
                           c2 * l(x + 2 * sx, y) + c3 * l(x + 3 * sx, y) +
                           c4 * l(x + 4 * sx, y) + c5 * l(x + 5 * sx, y));
          else if (inrange(x + 2 * sx))
            dveldx = sx * (-1.5 * l(x, y) + 2.0 * l(x + sx, y) -
                           0.5 * l(x + 2 * sx, y));
          else
            dveldx = sx * (l(x + sx, y) - l(x, y));
          VectorElement dveldy;
          if (inrange(y + 5 * sy))
            dveldy = sy * (c0 * l(x, y) + c1 * l(x, y + sy) +
                           c2 * l(x, y + 2 * sy) + c3 * l(x, y + 3 * sy) +
                           c4 * l(x, y + 4 * sy) + c5 * l(x, y + 5 * sy));
          else if (inrange(y + 2 * sy))
            dveldy = sy * (-1.5 * l(x, y) + 2.0 * l(x, y + sy) -
                           0.5 * l(x, y + 2 * sy));
          else
            dveldy = sx * (l(x, y + sy) - l(x, y));
          const VectorElement dveldx2 =
              l(x - 1, y) - 2.0 * l(x, y) + l(x + 1, y);
          const VectorElement dveldy2 =
              l(x, y - 1) - 2.0 * l(x, y) + l(x, y + 1);
          VectorElement dveldxdy;
          if (inrange(x + 2 * sx) && inrange(y + 2 * sy))
            dveldxdy =
                sx * sy *
                (-0.5 * (-1.5 * l(x + 2 * sx, y) + 2 * l(x + 2 * sx, y + sy) -
                         0.5 * l(x + 2 * sx, y + 2 * sy)) +
                 2 * (-1.5 * l(x + sx, y) + 2 * l(x + sx, y + sy) -
                      0.5 * l(x + sx, y + 2 * sy)) -
                 1.5 * (-1.5 * l(x, y) + 2 * l(x, y + sy) -
                        0.5 * l(x, y + 2 * sy)));
          else
            dveldxdy = sx * sy * (l(x + sx, y + sy) - l(x + sx, y)) -
                       (l(x, y + sy) - l(x, y));
          DuDx =
              dveldx.u[0] + dveldx2.u[0] * (ix - x) + dveldxdy.u[0] * (iy - y);
          DvDx =
              dveldx.u[1] + dveldx2.u[1] * (ix - x) + dveldxdy.u[1] * (iy - y);
          DuDy =
              dveldy.u[0] + dveldy2.u[0] * (iy - y) + dveldxdy.u[0] * (ix - x);
          DvDy =
              dveldy.u[1] + dveldy2.u[1] * (iy - y) + dveldxdy.u[1] * (ix - x);
        }
        const Real fXV = NUoH * DuDx * normX + NUoH * DuDy * normY,
                   fXP = -P(ix, iy).s * normX;
        const Real fYV = NUoH * DvDx * normX + NUoH * DvDy * normY,
                   fYP = -P(ix, iy).s * normY;
        const Real fXT = fXV + fXP, fYT = fYV + fYP;
        O->x_s[k] = p[0];
        O->y_s[k] = p[1];
        O->p_s[k] = P(ix, iy).s;
        O->u_s[k] = V(ix, iy).u[0];
        O->v_s[k] = V(ix, iy).u[1];
        O->nx_s[k] = dx;
        O->ny_s[k] = dy;
        O->omega_s[k] = (DvDx - DuDy) / info.h;
        O->uDef_s[k] = O->udef[iy][ix][0];
        O->vDef_s[k] = O->udef[iy][ix][1];
        O->fX_s[k] = -P(ix, iy).s * dx + NUoH * DuDx * dx + NUoH * DuDy * dy;
        O->fY_s[k] = -P(ix, iy).s * dy + NUoH * DvDx * dx + NUoH * DvDy * dy;
        O->fXv_s[k] = NUoH * DuDx * dx + NUoH * DuDy * dy;
        O->fYv_s[k] = NUoH * DvDx * dx + NUoH * DvDy * dy;
        O->perimeter += std::sqrt(normX * normX + normY * normY);
        O->circulation += normX * O->v_s[k] - normY * O->u_s[k];
        O->forcex += fXT;
        O->forcey += fYT;
        O->forcex_V += fXV;
        O->forcey_V += fYV;
        O->forcex_P += fXP;
        O->forcey_P += fYP;
        O->torque += (p[0] - Cx) * fYT - (p[1] - Cy) * fXT;
        O->torque_P += (p[0] - Cx) * fYP - (p[1] - Cy) * fXP;
        O->torque_V += (p[0] - Cx) * fYV - (p[1] - Cy) * fXV;
        const Real forcePar = fXT * vel_unit[0] + fYT * vel_unit[1];
        O->thrust += .5 * (forcePar + std::fabs(forcePar));
        O->drag -= .5 * (forcePar - std::fabs(forcePar));
        const Real forcePerp = fXT * vel_unit[1] - fYT * vel_unit[0];
        O->lift += forcePerp;
        const Real powOut = fXT * O->u_s[k] + fYT * O->v_s[k];
        const Real powDef = fXT * O->uDef_s[k] + fYT * O->vDef_s[k];
        O->Pout += powOut;
        O->defPower += powDef;
        O->PoutBnd += std::min((Real)0, powOut);
        O->defPowerBnd += std::min((Real)0, powDef);
      }
      O->PoutNew = O->forcex * shape->u + O->forcey * shape->v;
    }
  }
};
void ComputeForces::operator()(const Real dt) {
  sim.startProfiler("ComputeForces");
  KernelComputeForces K(sim);
  cubism::compute<KernelComputeForces, VectorGrid, VectorLab, ScalarGrid,
                  ScalarLab>(K, *sim.vel, *sim.chi);
  for (const auto &shape : sim.shapes)
    shape->computeForces();
  sim.stopProfiler();
}
ComputeForces::ComputeForces(SimulationData &s) : Operator(s) {}

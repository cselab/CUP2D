#pragma once
#include "Definitions.h"
using CHI_MAT = Real[_BS_][_BS_];
using UDEFMAT = Real[_BS_][_BS_][2];
struct surface_data {
  const int ix, iy;
  const Real dchidx, dchidy, delta;
  surface_data(const int _ix, const int _iy, const Real Xdx, const Real Xdy,
               const Real D)
      : ix(_ix), iy(_iy), dchidx(Xdx), dchidy(Xdy), delta(D) {}
};
struct ObstacleBlock {
  static const int sizeX = _BS_;
  static const int sizeY = _BS_;
  Real chi[sizeY][sizeX];
  Real dist[sizeY][sizeX];
  Real udef[sizeY][sizeX][2];
  size_t n_surfPoints = 0;
  bool filled = false;
  std::vector<surface_data *> surface;
  Real *x_s = nullptr;
  Real *y_s = nullptr;
  Real *p_s = nullptr;
  Real *u_s = nullptr;
  Real *v_s = nullptr;
  Real *nx_s = nullptr;
  Real *ny_s = nullptr;
  Real *omega_s = nullptr;
  Real *uDef_s = nullptr;
  Real *vDef_s = nullptr;
  Real *fX_s = nullptr;
  Real *fY_s = nullptr;
  Real *fXv_s = nullptr;
  Real *fYv_s = nullptr;
  Real perimeter = 0, forcex = 0, forcey = 0, forcex_P = 0, forcey_P = 0;
  Real forcex_V = 0, forcey_V = 0, torque = 0, torque_P = 0, torque_V = 0;
  Real drag = 0, thrust = 0, lift = 0, Pout = 0, PoutNew = 0, PoutBnd = 0,
       defPower = 0, defPowerBnd = 0;
  Real circulation = 0;
  Real COM_x = 0;
  Real COM_y = 0;
  Real Mass = 0;
  ObstacleBlock() {
    clear();
    surface.reserve(4 * _BS_);
  }
  ~ObstacleBlock() { clear_surface(); }
  void clear_surface() {
    filled = false;
    n_surfPoints = 0;
    perimeter = forcex = forcey = forcex_P = forcey_P = 0;
    forcex_V = forcey_V = torque = torque_P = torque_V = drag = thrust = lift =
        0;
    Pout = PoutBnd = defPower = defPowerBnd = circulation = 0;
    for (auto &trash : surface) {
      if (trash == nullptr)
        continue;
      delete trash;
      trash = nullptr;
    }
    surface.clear();
    if (x_s not_eq nullptr) {
      free(x_s);
      x_s = nullptr;
    }
    if (y_s not_eq nullptr) {
      free(y_s);
      y_s = nullptr;
    }
    if (p_s not_eq nullptr) {
      free(p_s);
      p_s = nullptr;
    }
    if (u_s not_eq nullptr) {
      free(u_s);
      u_s = nullptr;
    }
    if (v_s not_eq nullptr) {
      free(v_s);
      v_s = nullptr;
    }
    if (nx_s not_eq nullptr) {
      free(nx_s);
      nx_s = nullptr;
    }
    if (ny_s not_eq nullptr) {
      free(ny_s);
      ny_s = nullptr;
    }
    if (omega_s not_eq nullptr) {
      free(omega_s);
      omega_s = nullptr;
    }
    if (uDef_s not_eq nullptr) {
      free(uDef_s);
      uDef_s = nullptr;
    }
    if (vDef_s not_eq nullptr) {
      free(vDef_s);
      vDef_s = nullptr;
    }
    if (fX_s not_eq nullptr) {
      free(fX_s);
      fX_s = nullptr;
    }
    if (fY_s not_eq nullptr) {
      free(fY_s);
      fY_s = nullptr;
    }
    if (fXv_s not_eq nullptr) {
      free(fXv_s);
      fXv_s = nullptr;
    }
    if (fYv_s not_eq nullptr) {
      free(fYv_s);
      fYv_s = nullptr;
    }
  }
  void clear() {
    clear_surface();
    std::fill(dist[0], dist[0] + sizeX * sizeY, -1);
    std::fill(chi[0], chi[0] + sizeX * sizeY, 0);
    memset(udef, 0, sizeof(Real) * sizeX * sizeY * 2);
  }
  void write(const int ix, const int iy, const Real delta, const Real gradUX,
             const Real gradUY) {
    assert(!filled);
    if (delta > 0) {
      n_surfPoints++;
      const Real dchidx = -delta * gradUX, dchidy = -delta * gradUY;
      surface.push_back(new surface_data(ix, iy, dchidx, dchidy, delta));
    }
  }
  void allocate_surface() {
    filled = true;
    assert(surface.size() == n_surfPoints);
    x_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    y_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    p_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    u_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    v_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    nx_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    ny_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    omega_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    uDef_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    vDef_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    fX_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    fY_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    fXv_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    fYv_s = (Real *)calloc(n_surfPoints, sizeof(Real));
  }
  void fill_stringstream(std::stringstream &s) {
    for (size_t i = 0; i < n_surfPoints; i++)
      s << x_s[i] << ", " << y_s[i] << ", " << p_s[i] << ", " << u_s[i] << ", "
        << v_s[i] << ", " << nx_s[i] << ", " << ny_s[i] << ", " << omega_s[i]
        << ", " << uDef_s[i] << ", " << vDef_s[i] << ", " << fX_s[i] << ", "
        << fY_s[i] << ", " << fXv_s[i] << ", " << fYv_s[i] << "\n";
  }
};

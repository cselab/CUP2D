//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Definitions.h"
#include <math.h>

struct IF2D_Frenet2D
{
  static void solve( const unsigned Nm, const Real*const rS,
    const Real*const curv, const Real*const curv_dt,
    Real*const rX, Real*const rY, Real*const vX, Real*const vY,
    Real*const norX, Real*const norY, Real*const vNorX, Real*const vNorY )
  {
    // initial conditions
    rX[0] = 0.0;
    rY[0] = 0.0;
    norX[0] = 0.0;
    norY[0] = 1.0;
    Real ksiX = 1.0;
    Real ksiY = 0.0;
    // velocity variables
    vX[0] = 0.0;
    vY[0] = 0.0;
    vNorX[0] = 0.0;
    vNorY[0] = 0.0;
    Real vKsiX = 0.0;
    Real vKsiY = 0.0;

    for(unsigned i=1; i<Nm; i++) {
      // compute derivatives positions
      const Real dksiX = curv[i-1]*norX[i-1];
      const Real dksiY = curv[i-1]*norY[i-1];
      const Real dnuX = -curv[i-1]*ksiX;
      const Real dnuY = -curv[i-1]*ksiY;
      // compute derivatives velocity
      const Real dvKsiX = curv_dt[i-1]*norX[i-1] + curv[i-1]*vNorX[i-1];
      const Real dvKsiY = curv_dt[i-1]*norY[i-1] + curv[i-1]*vNorY[i-1];
      const Real dvNuX = -curv_dt[i-1]*ksiX - curv[i-1]*vKsiX;
      const Real dvNuY = -curv_dt[i-1]*ksiY - curv[i-1]*vKsiY;
      // compute current ds
      const Real ds = rS[i] - rS[i-1];
      // update
      rX[i] = rX[i-1] + ds*ksiX;
      rY[i] = rY[i-1] + ds*ksiY;
      norX[i] = norX[i-1] + ds*dnuX;
      norY[i] = norY[i-1] + ds*dnuY;
      ksiX += ds * dksiX;
      ksiY += ds * dksiY;
      // update velocities
      vX[i] = vX[i-1] + ds*vKsiX;
      vY[i] = vY[i-1] + ds*vKsiY;
      vNorX[i] = vNorX[i-1] + ds*dvNuX;
      vNorY[i] = vNorY[i-1] + ds*dvNuY;
      vKsiX += ds * dvKsiX;
      vKsiY += ds * dvKsiY;
      // normalize unit vectors
      const Real d1 = ksiX*ksiX + ksiY*ksiY;
      const Real d2 = norX[i]*norX[i] + norY[i]*norY[i];
      if(d1>std::numeric_limits<Real>::epsilon()) {
        const Real normfac = 1/std::sqrt(d1);
        ksiX*=normfac;
        ksiY*=normfac;
      }
      if(d2>std::numeric_limits<Real>::epsilon()) {
        const Real normfac = 1/std::sqrt(d2);
        norX[i]*=normfac;
        norY[i]*=normfac;
      }
    }
  }
};

class IF2D_Interpolation1D
{
 public:

  static void naturalCubicSpline(const Real*x, const Real*y,
    const unsigned n, const Real*xx, Real*yy, const unsigned nn) {
      return naturalCubicSpline(x,y,n,xx,yy,nn,0);
  }

  static void naturalCubicSpline(const Real*x, const Real*y, const unsigned n,
    const Real*xx, Real*yy, const unsigned nn, const Real offset)
  {
    std::vector<Real> y2(n), u(n-1);

    y2[0] = 0;
    u[0] = 0;
    for(unsigned i=1; i<n-1; i++) {
      const Real sig = (x[i]-x[i-1])/(x[i+1]-x[i-1]);
      const Real p = sig*y2[i-1] +2;
      y2[i] = (sig-1)/p;
      u[i] = (y[i+1]-y[i])/(x[i+1]-x[i])-(y[i]-y[i-1])/(x[i]-x[i-1]);
      u[i] = (6*u[i]/(x[i+1]-x[i-1])-sig*u[i-1])/p;
    }

    const Real qn = 0;
    const Real un = 0;
    y2[n-1] = (un-qn*u[n-2])/(qn*y2[n-2] +1);

    for(unsigned k=n-2; k>0; k--) y2[k] = y2[k]*y2[k+1] +u[k];

    //#pragma omp parallel for schedule(static)
    for(unsigned j=0; j<nn; j++) {
      unsigned int klo = 0;
      unsigned int khi = n-1;
      unsigned int k = 0;
      while(khi-klo>1) {
        k=(khi+klo)>>1;
        if( x[k]>(xx[j]+offset)) khi=k;
        else                     klo=k;
      }

      const Real h = x[khi] - x[klo];
      if(h<=0.0) {
        std::cout<<"Interpolation points must be distinct!"<<std::endl; abort();
      }
      const Real a = (x[khi]-(xx[j]+offset))/h;
      const Real b = ((xx[j]+offset)-x[klo])/h;
      yy[j] = a*y[klo]+b*y[khi]+((a*a*a-a)*y2[klo]+(b*b*b-b)*y2[khi])*(h*h)/6;
    }
  }

  static void cubicInterpolation(const Real x0, const Real x1, const Real x,
    const Real y0,const Real y1,const Real dy0,const Real dy1, Real&y, Real&dy)
  {
    const Real xrel = (x-x0);
    const Real deltax = (x1-x0);

    const Real a = (dy0+dy1)/(deltax*deltax) - 2*(y1-y0)/(deltax*deltax*deltax);
    const Real b = (-2*dy0-dy1)/deltax + 3*(y1-y0)/(deltax*deltax);
    const Real c = dy0;
    const Real d = y0;

    y = a*xrel*xrel*xrel + b*xrel*xrel + c*xrel + d;
    dy = 3*a*xrel*xrel + 2*b*xrel + c;
  }

  static void cubicInterpolation(const Real x0, const Real x1, const Real x,
    const Real y0, const Real y1, Real & y, Real & dy) {
    return cubicInterpolation(x0,x1,x,y0,y1,0,0,y,dy); // 0 slope at end points
  }

  static void linearInterpolation(const Real x0, const Real x1, const Real x,
    const Real y0,const Real y1, Real&y, Real&dy)
  {
    y = (y1 - y0) / (x1 - x0) * (x - x0) + y0;
    dy = (y1 - y0) / (x1 - x0);
  }
};

namespace Schedulers
{
template<int Npoints>
struct ParameterScheduler
{
  std::array<Real, Npoints>  parameters_t0; // parameters at t0
  std::array<Real, Npoints>  parameters_t1; // parameters at t1
  std::array<Real, Npoints> dparameters_t0; // derivative at t0
  Real t0, t1; // t0 and t1

  void save(std::string filename) {
    std::ofstream savestream;
    savestream.setf(std::ios::scientific);
    savestream.precision(std::numeric_limits<Real>::digits10 + 1);
    savestream.open(filename);

    savestream << t0 << "\t" << t1 << std::endl;
    for(int i=0;i<Npoints;++i)
      savestream << parameters_t0[i]  << "\t"
                 << parameters_t1[i]  << "\t"
                 << dparameters_t0[i] << std::endl;
    savestream.close();
  }

  void restart(std::string filename)
  {
    std::ifstream restartstream;
    restartstream.open(filename);
    restartstream >> t0 >> t1;
    for(int i=0;i<Npoints;++i)
    restartstream >> parameters_t0[i] >> parameters_t1[i] >> dparameters_t0[i];
    restartstream.close();
  }
  virtual void resetAll()
  {
    parameters_t0 = std::array<Real, Npoints>();
    parameters_t1 = std::array<Real, Npoints>();
    dparameters_t0 = std::array<Real, Npoints>();
    t0 = -1;
    t1 =  0;
  }

  ParameterScheduler()
  {
    t0=-1; t1=0;
    parameters_t0 = std::array<Real, Npoints>();
    parameters_t1 = std::array<Real, Npoints>();
    dparameters_t0 = std::array<Real, Npoints>();
  }
  virtual ~ParameterScheduler() {}

  void transition(const Real t, const Real tstart, const Real tend,
      const std::array<Real, Npoints> parameters_tend,
      const bool UseCurrentDerivative = false)
  {
    if(t<tstart or t>tend) return; // this transition is out of scope
    //if(tstart<t0) return; // this transition is not relevant: we are doing a next one already

    // we transition from whatever state we are in to a new state
    // the start point is where we are now: lets find out
    std::array<Real, Npoints> parameters;
    std::array<Real, Npoints> dparameters;
    gimmeValues(tstart,parameters,dparameters);

    // fill my members
    t0 = tstart;
    t1 = tend;
    parameters_t0 = parameters;
    parameters_t1 = parameters_tend;
    dparameters_t0 = UseCurrentDerivative ? dparameters : std::array<Real, Npoints>();
  }

  void transition(const Real t, const Real tstart, const Real tend,
      const std::array<Real, Npoints> parameters_tstart,
      const std::array<Real, Npoints> parameters_tend)
  {
    if(t<tstart or t>tend) return; // this transition is out of scope
    if(tstart<t0) return; // this transition is not relevant: we are doing a next one already

    // fill my members
    t0 = tstart;
    t1 = tend;
    parameters_t0 = parameters_tstart;
    parameters_t1 = parameters_tend;
  }

  void gimmeValues(const Real t, std::array<Real, Npoints>& parameters, std::array<Real, Npoints>& dparameters)
  {
    // look at the different cases
    if(t<t0 or t0<0) { // no transition, we are in state 0
      parameters = parameters_t0;
      dparameters = std::array<Real, Npoints>();
    } else if(t>t1) { // no transition, we are in state 1
      parameters = parameters_t1;
      dparameters = std::array<Real, Npoints>();
    } else { // we are within transition: interpolate
      for(int i=0;i<Npoints;++i)
        IF2D_Interpolation1D::cubicInterpolation(t0,t1,t,parameters_t0[i],parameters_t1[i],dparameters_t0[i],0.0,parameters[i],dparameters[i]);
    }
  }

  void gimmeValuesLinear(const Real t, std::array<Real, Npoints>& parameters, std::array<Real, Npoints>& dparameters)
  {
    // look at the different cases
    if(t<t0 or t0<0) { // no transition, we are in state 0
      parameters = parameters_t0;
      dparameters = std::array<Real, Npoints>();
    } else if(t>t1) { // no transition, we are in state 1
      parameters = parameters_t1;
      dparameters = std::array<Real, Npoints>();
    } else { // we are within transition: interpolate
      for(int i=0;i<Npoints;++i)
        IF2D_Interpolation1D::linearInterpolation(t0,t1,t,parameters_t0[i],parameters_t1[i],parameters[i],dparameters[i]);
    }
  }

  void gimmeValues(const Real t, std::array<Real, Npoints>& parameters)
  {
    std::array<Real, Npoints> dparameters_whocares; // no derivative info
    return gimmeValues(t,parameters,dparameters_whocares);
  }
};

struct ParameterSchedulerScalar : ParameterScheduler<1>
{
  void transition(const Real t, const Real tstart, const Real tend,
    const Real parameter_tend, const bool keepSlope = false) {
    const std::array<Real, 1> myParameter = {parameter_tend};
    return
      ParameterScheduler<1>::transition(t,tstart,tend,myParameter,keepSlope);
  }

  void transition(const Real t, const Real tstart, const Real tend,
                  const Real parameter_tstart, const Real parameter_tend)
  {
    const std::array<Real, 1> myParameterStart = {parameter_tstart};
    const std::array<Real, 1> myParameterEnd = {parameter_tend};
    return ParameterScheduler<1>::transition(t,tstart,tend,myParameterStart,myParameterEnd);
  }

  void gimmeValues(const Real t, Real & parameter, Real & dparameter)
  {
    std::array<Real, 1> myParameter, mydParameter;
    ParameterScheduler<1>::gimmeValues(t, myParameter, mydParameter);
    parameter = myParameter[0];
    dparameter = mydParameter[0];
  }

  void gimmeValues(const Real t, Real & parameter)
  {
    std::array<Real, 1> myParameter;
    ParameterScheduler<1>::gimmeValues(t, myParameter);
    parameter = myParameter[0];
  }
};

template<int Npoints>
struct ParameterSchedulerVector : ParameterScheduler<Npoints>
{
  void gimmeValues(const Real t, const std::array<Real, Npoints>& positions,
    const int Nfine, const Real*const positions_fine,
    Real*const parameters_fine, Real * const dparameters_fine) {
    // we interpolate in space the start and end point
    Real* parameters_t0_fine  = new Real[Nfine];
    Real* parameters_t1_fine  = new Real[Nfine];
    Real* dparameters_t0_fine = new Real[Nfine];

    IF2D_Interpolation1D::naturalCubicSpline(positions.data(),
      this->parameters_t0.data(), Npoints, positions_fine, parameters_t0_fine,
      Nfine);
    IF2D_Interpolation1D::naturalCubicSpline(positions.data(),
      this->parameters_t1.data(), Npoints, positions_fine, parameters_t1_fine,
      Nfine);
    IF2D_Interpolation1D::naturalCubicSpline(positions.data(),
      this->dparameters_t0.data(),Npoints, positions_fine, dparameters_t0_fine,
      Nfine);

    // look at the different cases
    if(t<this->t0 or this->t0<0) { // no transition, we are in state 0
      memcpy (parameters_fine, parameters_t0_fine, Nfine*sizeof(Real) );
      memset (dparameters_fine, 0, Nfine*sizeof(Real) );
    } else if(t>this->t1) { // no transition, we are in state 1
      memcpy (parameters_fine, parameters_t1_fine, Nfine*sizeof(Real) );
      memset (dparameters_fine, 0, Nfine*sizeof(Real) );
    } else {
      // we are within transition: interpolate in time for each point of the fine discretization
      //#pragma omp parallel for schedule(static)
      for(int i=0;i<Nfine;++i)
        IF2D_Interpolation1D::cubicInterpolation(this->t0, this->t1, t,
          parameters_t0_fine[i], parameters_t1_fine[i], dparameters_t0_fine[i],
          0, parameters_fine[i], dparameters_fine[i]);
    }
    delete [] parameters_t0_fine;
    delete [] parameters_t1_fine;
    delete [] dparameters_t0_fine;
  }

  void gimmeValues(const Real t, std::array<Real, Npoints>& parameters) {
    ParameterScheduler<Npoints>::gimmeValues(t, parameters);
  }

  void gimmeValues(const Real t, std::array<Real, Npoints> & parameters, std::array<Real, Npoints> & dparameters) {
    ParameterScheduler<Npoints>::gimmeValues(t, parameters, dparameters);
  }
};

template<int Npoints>
struct ParameterSchedulerLearnWave : ParameterScheduler<Npoints>
{
  template<typename T>
  void gimmeValues(const Real t, const Real Twave, const Real Length,
    const std::array<Real, Npoints> & positions, const int Nfine,
    const T*const positions_fine, T*const parameters_fine, Real*const dparameters_fine)
  {
    const Real _1oL = 1./Length;
    const Real _1oT = 1./Twave;
    // the fish goes through (as function of t and s) a wave function that describes the curvature
    //#pragma omp parallel for schedule(static)
    for(int i=0;i<Nfine;++i) {
      const Real c = positions_fine[i]*_1oL - (t - this->t0)*_1oT; //traveling wave coord
      bool bCheck = true;

      if (c < positions[0]) { // Are you before latest wave node?
        IF2D_Interpolation1D::cubicInterpolation(
          c, positions[0], c,
          this->parameters_t0[0], this->parameters_t0[0],
          parameters_fine[i], dparameters_fine[i]);
        bCheck = false;
      }
      else if (c > positions[Npoints-1]) {// Are you after oldest wave node?
          IF2D_Interpolation1D::cubicInterpolation(
          positions[Npoints-1], c, c,
          this->parameters_t0[Npoints-1], this->parameters_t0[Npoints-1],
          parameters_fine[i], dparameters_fine[i]);
        bCheck = false;
      } else {
        for (int j=1; j<Npoints; ++j) { // Check at which point of the travelling wave we are
          if (( c >= positions[j-1] ) && ( c <= positions[j] )) {
            IF2D_Interpolation1D::cubicInterpolation(
              positions[j-1], positions[j], c,
              this->parameters_t0[j-1], this->parameters_t0[j],
              parameters_fine[i], dparameters_fine[i]);
            dparameters_fine[i] = -dparameters_fine[i]*_1oT; // df/dc * dc/dt
            bCheck = false;
          }
        }
      }
      if (bCheck) { std::cout << "Ciaone2!" << std::endl; abort(); }
    }
  }

  void Turn(const Real b, const Real t_turn) // each decision adds a node at the beginning of the wave (left, right, straight) and pops last node
  {
    this->t0 = t_turn;

    for(int i=Npoints-1; i>1; --i)
        this->parameters_t0[i] = this->parameters_t0[i-2];
    this->parameters_t0[1] = b;
    this->parameters_t0[0] = 0;
  }
};

/*********************** NEURO-KINEMATIC FISH *******************************/

class Synapse
{
public:
    Real g = 0;
    Real dg = 0;
    const Real tau1 = 0.006 / 0.044;
    const Real tau2 = 0.008 / 0.044;
    Real prevTime = 0.0;
    std::vector<Real> activationTimes;
    std::vector<Real> activationAmplitudes;
public:
    void reset() {
        g = 0.0;
        dg = 0.0;
        prevTime = 0.0;
        activationTimes.clear();
        activationAmplitudes.clear();
    }
    void advance(const Real t) {
//        printf("[Synapse][advance]\n");
        dg = 0;
        Real dt = t - prevTime;
//        printf("[Synapse][advance] activationTimes.size() %ld\n", activationTimes.size());
        for (size_t i=0;i<activationTimes.size();i++) {
            const Real deltaT = t - activationTimes.at(i);
//            printf("[Synapse][advance] deltaT %f\n", deltaT);
            const Real dBiExp = -1 / tau2 * std::exp(-deltaT / tau2) + 1 / tau1 * std::exp(-deltaT / tau1);
//            printf("[Synapse][advance] dBiExp %f\n", dBiExp);
            dg += activationAmplitudes.at(i) * dBiExp;
//            printf("[Synapse][advance] dg %f\n", dg);
        }
        g += dg * dt;
        prevTime = t;
        forget(t);
//        printf("[Synapse][advance][end]\n");
    }
    void excite(const Real t, const Real amp) {
//        printf("[Synapse][excite]\n");
        activationTimes.push_back(t);
        activationAmplitudes.push_back(amp);
//        printf("[Synapse][excite][end]\n");
    }
    void forget(const Real t)
    {
//        printf("[Synapse][forget]\n");
        if (activationTimes.size() != 0) {
//            printf("[Synapse][forget] Number of activated synapses %ld\n", activationTimes.size());
//            printf("[Synapse][forget] t: %f, activationTime0: %f\n", t, activationTimes.at(0));
//            printf("[Synapse][forget] tau1tau2: %f\n", tau1+tau2);
            if (t - activationTimes.at(0) > tau1 + tau2) {
//                printf("Forgetting an activation. Current activation size is %ld\n", activationTimes.size());
                activationTimes.erase(activationTimes.begin());
                activationAmplitudes.erase(activationAmplitudes.begin());
            }
        }
//        printf("[Synapse][forget][end]\n");
    }
    Real value()
    {
        return g;
    }
    Real speed()
    {
        return dg;
    }
};

template<int Npoints>
class Oscillation
{
public:
    Real d = 0.0;
    Real t0 = 0.0;
    Real prev_fmod = 0.0;
    std::vector<Real> signal = std::vector<Real>(Npoints, 0.0);
    std::vector<Real> signal_out = std::vector<Real>(Npoints, 0.0);
public:
    void reset()
    {
        d = 0.0;
        t0 = 0.0;
        prev_fmod = 0.0;
        signal.clear();
        signal_out.clear();
    }
    void modify(const Real t0_in, const Real f_in, const Real d_in) {
//        printf("[Oscillation][modify]\n");
        d = d_in;
        t0 = t0_in;
        prev_fmod = 0;

        signal = std::vector<Real>(Npoints, 0.0);
        signal.at(0) = f_in;
        signal.at(static_cast<int>(std::ceil(static_cast<float>(Npoints + 1)/2.0) - 1.0)) = -f_in;
        signal_out = signal;
//        printf("[Oscillation][modify][end]\n");
    }
    void advance(const Real t)
    {
//        printf("[Oscillation][advance]\n");
        if (fmod(t - t0, d) < prev_fmod && t>t0) {
            signal.insert(signal.begin(), signal.back());
            signal.pop_back();
            signal_out = signal;
        } else if (t == t0) {
            signal_out = signal;
        } else {
            signal_out = std::vector<Real>(Npoints, 0.0);
        }
        prev_fmod = fmod(t - t0, d);
//        printf("[Oscillation][advance][end]\n");
    }
};

template<int Npoints>
struct ParameterSchedulerNeuroKinematic : ParameterScheduler<Npoints>
{
    Real prevTime = 0.0;
    int numActiveSpikes = 0;
    const Real tau1 = 0.006 / 0.044; //1 ms
    const Real tau2 = 0.008 / 0.044; //6 ms (AMPA)

    std::array<Real, Npoints> neuroSignal_t_coarse = std::array<Real, Npoints>();
    std::array<Real, Npoints> timeActivated_coarse = std::array<Real, Npoints>(); // array of time each synapse has been activated for
    std::array<Real, Npoints> muscSignal_t_coarse = std::array<Real, Npoints>();
    std::array<Real, Npoints> dMuscSignal_t_coarse = std::array<Real, Npoints>();
    std::vector<std::array<Real, Npoints>> neuroSignalVec_coarse;
    std::vector<std::array<Real, Npoints>> timeActivatedVec_coarse;
    std::vector<std::array<Real, Npoints>> muscSignalVec_coarse;
    std::vector<std::array<Real, Npoints>> dMuscSignalVec_coarse;
    std::vector<Real> amplitudeVec;


    virtual void resetAll()
    {
        prevTime = 0.0;
        numActiveSpikes = 0;
        neuroSignal_t_coarse = std::array<Real, Npoints>();
        timeActivated_coarse = std::array<Real, Npoints>();
        muscSignal_t_coarse = std::array<Real, Npoints>();
        dMuscSignal_t_coarse = std::array<Real, Npoints>();
        neuroSignalVec_coarse.clear();
        timeActivatedVec_coarse.clear();
        muscSignalVec_coarse.clear();
        dMuscSignalVec_coarse.clear();
        amplitudeVec.clear();
    }

    template<typename T>
    void gimmeValues(const Real t, const Real Length,
                     const std::array<Real, Npoints> & positions, const int Nfine,
                     const T*const positions_fine, T*const muscSignal_t_fine, Real*const dMuscSignal_t_fine,
                     Real*const spatialDerivativeMuscSignal, Real*const spatialDerivativeDMuscSignal)
    {
        // Advance arrays
        if (numActiveSpikes > 0) {
            this->dMuscSignal_t_coarse = std::array<Real, Npoints>();

            // Delete spikes that are no longer relevant
            for (int i=0;i<numActiveSpikes;i++) {
                const Real relaxationTime = (Npoints + 1) * tau1 + tau2;
                const Real activeSpikeTime = t-timeActivatedVec_coarse.at(i).at(0);
                if (activeSpikeTime >= relaxationTime) {
                    numActiveSpikes -= 1;
                    this->neuroSignalVec_coarse.erase(neuroSignalVec_coarse.begin() + i);
                    this->timeActivatedVec_coarse.erase(timeActivatedVec_coarse.begin() + i);
                    this->muscSignalVec_coarse.erase(muscSignalVec_coarse.begin() + i);
                    this->dMuscSignalVec_coarse.erase(dMuscSignalVec_coarse.begin() + i);
                }
            }

            advanceCoarseArrays(t);

            // Set previous time for next gimmeValues call
            this->prevTime = t;

            // Construct spine with cubic spline
            IF2D_Interpolation1D::naturalCubicSpline(positions.data(),
                                                     this->muscSignal_t_coarse.data(), Npoints, positions_fine,
                                                     muscSignal_t_fine, Nfine);
            IF2D_Interpolation1D::naturalCubicSpline(positions.data(),
                                                     this->dMuscSignal_t_coarse.data(), Npoints, positions_fine,
                                                     dMuscSignal_t_fine, Nfine);
        }
    }


    void advanceCoarseArrays(const Real time_current) {
//        printf("[numActiveSpikes][%d]\n", numActiveSpikes);
        const Real delta_t = time_current - this->prevTime;
        for (int i = 0; i < numActiveSpikes; i++) {
            for (int j = 0; j < Npoints; j++) {
                const Real deltaT = time_current - this->timeActivatedVec_coarse.at(i).at(j);
                if (deltaT >= 0) {
//                    printf("[i=%d][j=%d]\n", i, j);
                    // Activate current node but don't switch off previous one.
                    if (j > 0) {
                        this->neuroSignalVec_coarse.at(i).at(j) = this->neuroSignalVec_coarse.at(i)[j - 1];
                    }
                    // Begin the muscle response at the new node.
                    const Real dBiExp = -1 / this->tau2 * std::exp(-deltaT / this->tau2) +
                                          1 / this->tau1 * std::exp(-deltaT / this->tau1);

                    this->dMuscSignalVec_coarse.at(i).at(j) = this->neuroSignalVec_coarse.at(i).at(j) * dBiExp;

                    // Increment the overall muscle signal and write the overall derivative
                    this->dMuscSignal_t_coarse.at(j) += this->dMuscSignalVec_coarse.at(i).at(j);
                    this->muscSignal_t_coarse.at(j) += delta_t * this->dMuscSignalVec_coarse.at(i).at(j);
                }
            }
        }
    }

    // Deal with residual signal from previous firing time action (you can increment the signal with itself)
    void Spike(const Real t_spike, const Real aCmd, const Real dCmd, const Real deltaTFireCmd)
    {
        this->t0 = t_spike;
        this->prevTime = t_spike;
        this->numActiveSpikes += 1;
        this->neuroSignalVec_coarse.push_back(std::array<Real, Npoints>());
        this->timeActivatedVec_coarse.push_back(std::array<Real, Npoints>());
        this->muscSignalVec_coarse.push_back(std::array<Real, Npoints>());
        this->dMuscSignalVec_coarse.push_back(std::array<Real, Npoints>());

        for(int j=0; j < Npoints; j++){
            this->timeActivatedVec_coarse.at(numActiveSpikes-1).at(j) = this->t0 + j*dCmd;
        }
        // Activate the 0th node
        this->neuroSignalVec_coarse.at(numActiveSpikes-1).at(0) = aCmd;
    }
};

template<int Npoints>
struct ParameterSchedulerNeuroKinematicObject : ParameterScheduler<Npoints>
{
    std::array<Synapse, Npoints> synapses;
    Oscillation<Npoints> oscillation;

    std::array<Real, Npoints> muscle_value = std::array<Real, Npoints>();
    std::array<Real, Npoints> muscle_speed = std::array<Real, Npoints>();

    virtual void resetAll()
    {
        for (int i=0;i<Npoints;i++){
            synapses.at(i).reset();
        }
        oscillation.reset();
    }

    template<typename T>
    void gimmeValues(const Real t, const Real Length,
                     const std::array<Real, Npoints> & positions, const int Nfine,
                     const T*const positions_fine, T*const muscle_value_fine, Real*const muscle_speed_fine)
    {
        advance(t);

        // Construct spine with cubic spline
        IF2D_Interpolation1D::naturalCubicSpline(positions.data(),
                                                 this->muscle_value.data(), Npoints, positions_fine,
                                                 muscle_value_fine, Nfine);
        IF2D_Interpolation1D::naturalCubicSpline(positions.data(),
                                                 this->muscle_speed.data(), Npoints, positions_fine,
                                                 muscle_speed_fine, Nfine);
    }


    void advance(const Real t)
    {
        oscillation.advance(t);
        for (int i=0; i<Npoints; i++) {
//            printf("[Scheduler][advance]\n");
            const Real oscAmp = oscillation.signal_out.at(i);
            printf("[Scheduler][advance] signal_i %f\n", (double)oscillation.signal.at(i));
//            printf("[Scheduler][advance] oscAmp_i %f\n", oscAmp);
            if (oscAmp != 0) {synapses.at(i).excite(t, oscAmp);}
            synapses.at(i).advance(t);
            muscle_value.at(i) = synapses.at(i).value();
            muscle_speed.at(i) = synapses.at(i).speed();

            if (i==0) {printf("[Scheduler][advance] muscle_value_0 %f\n", (double)muscle_value.at(0));}
//            if (i==0) {printf("[Scheduler][advance] synapse_0 amplitude %f\n", synapses.at(0).activationAmplitudes.at(0));}
            if (i==0) {printf("[Scheduler][advance] synapse_0 numActivations %ld\n", synapses.at(0).activationAmplitudes.size());}
            if (i==10) {printf("[Scheduler][advance] muscle_value_10 %f\n", (double)muscle_value.at(10));}
//            if (i==9) {printf("[Scheduler][advance] synapse_9 amplitude %f\n", synapses.at(9).activationAmplitudes.at(0));}
            if (i==10) {printf("[Scheduler][advance] synapse_10 numActivations %ld\n", synapses.at(10).activationAmplitudes.size());}

//            printf("[Scheduler][advance] muscle_value_i %f\n", muscle_value.at(i));
//            printf("[Scheduler][advance] muscle_speed_i %f\n", muscle_speed.at(i));
//            printf("[Scheduler][advance][end]\n");
        }
    }

    void Spike(const Real t_spike, const Real aCmd, const Real dCmd, const Real deltaTFireCmd)
    {
        oscillation.modify(t_spike, aCmd, dCmd);
//        synapses.at(0).excite(t_spike, aCmd);
//        synapses.at(static_cast<int>(std::ceil(static_cast<float>(Npoints + 1)/2.0) - 1.0)).excite(t_spike, -aCmd);
//        printf("Activated synapse 0 and synapse %d", static_cast<int>(std::ceil(static_cast<float>(Npoints + 1)/2.0) - 1.0));
    }
};
}

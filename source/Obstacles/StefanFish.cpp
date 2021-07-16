//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "StefanFish.h"
#include "FishData.h"
#include <sstream>

using namespace cubism;

void StefanFish::resetAll() {
  CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  if(cFish == nullptr) { printf("Someone touched my fish\n"); abort(); }
  cFish->resetAll();
  Fish::resetAll();
}

StefanFish::StefanFish(SimulationData&s, ArgumentParser&p, double C[2]):
 Fish(s,p,C), bCorrectTrajectory(p("-pid").asInt(0)),
 bCorrectPosition(p("-pidpos").asInt(0))
{
 #if 0
  // parse tau
  tau = parser("-tau").asDouble(1.0);
  // parse curvature controlpoint values
  curvature_values[0] = parser("-k1").asDouble(0.82014);
  curvature_values[1] = parser("-k2").asDouble(1.46515);
  curvature_values[2] = parser("-k3").asDouble(2.57136);
  curvature_values[3] = parser("-k4").asDouble(3.75425);
  curvature_values[4] = parser("-k5").asDouble(5.09147);
  curvature_values[5] = parser("-k6").asDouble(5.70449);
  // if nonzero && Learnfreq<0 your fish is gonna keep turning
  baseline_values[0] = parser("-b1").asDouble(0.0);
  baseline_values[1] = parser("-b2").asDouble(0.0);
  baseline_values[2] = parser("-b3").asDouble(0.0);
  baseline_values[3] = parser("-b4").asDouble(0.0);
  baseline_values[4] = parser("-b5").asDouble(0.0);
  baseline_values[5] = parser("-b6").asDouble(0.0);
  // curvature points are distributed by default but can be overridden
  curvature_points[0] = parser("-pk1").asDouble(0.00)*length;
  curvature_points[1] = parser("-pk2").asDouble(0.15)*length;
  curvature_points[2] = parser("-pk3").asDouble(0.40)*length;
  curvature_points[3] = parser("-pk4").asDouble(0.65)*length;
  curvature_points[4] = parser("-pk5").asDouble(0.90)*length;
  curvature_points[5] = parser("-pk6").asDouble(1.00)*length;
  baseline_points[0] = parser("-pb1").asDouble(curvature_points[0]/length)*length;
  baseline_points[1] = parser("-pb2").asDouble(curvature_points[1]/length)*length;
  baseline_points[2] = parser("-pb3").asDouble(curvature_points[2]/length)*length;
  baseline_points[3] = parser("-pb4").asDouble(curvature_points[3]/length)*length;
  baseline_points[4] = parser("-pb5").asDouble(curvature_points[4]/length)*length;
  baseline_points[5] = parser("-pb6").asDouble(curvature_points[5]/length)*length;
  printf("created IF2D_StefanFish: xpos=%3.3f ypos=%3.3f angle=%3.3f L=%3.3f Tp=%3.3f tau=%3.3f phi=%3.3f\n",position[0],position[1],angle,length,Tperiod,tau,phaseShift);
  printf("curvature points: pk1=%3.3f pk2=%3.3f pk3=%3.3f pk4=%3.3f pk5=%3.3f pk6=%3.3f\n",curvature_points[0],curvature_points[1],curvature_points[2],curvature_points[3],curvature_points[4],curvature_points[5]);
  printf("curvature values (normalized to L=1): k1=%3.3f k2=%3.3f k3=%3.3f k4=%3.3f k5=%3.3f k6=%3.3f\n",curvature_values[0],curvature_values[1],curvature_values[2],curvature_values[3],curvature_values[4],curvature_values[5]);
  printf("baseline points: pb1=%3.3f pb2=%3.3f pb3=%3.3f pb4=%3.3f pb5=%3.3f pb6=%3.3f\n",baseline_points[0],baseline_points[1],baseline_points[2],baseline_points[3],baseline_points[4],baseline_points[5]);
  printf("baseline values (normalized to L=1): b1=%3.3f b2=%3.3f b3=%3.3f b4=%3.3f b5=%3.3f b6=%3.3f\n",baseline_values[0],baseline_values[1],baseline_values[2],baseline_values[3],baseline_values[4],baseline_values[5]);
  // make curvature dimensional for this length
  for(int i=0; i<6; ++i) curvature_values[i]/=length;
 #endif

  const Real ampFac = p("-amplitudeFactor").asDouble(1.0);
  myFish = new CurvatureFish(length, Tperiod, phaseShift, sim.minH, ampFac);
  if( s.verbose ) printf("[CUP2D] - CurvatureFish %d %f %f %f %f %f %f\n",myFish->Nm, length,myFish->dSref,myFish->dSmid,sim.minH, Tperiod, phaseShift);
}

//static inline Real sgn(const Real val) { return (0 < val) - (val < 0); }
void StefanFish::create(const std::vector<BlockInfo>& vInfo)
{
  // If PID controller to keep position or swim straight enabled
  if (bCorrectPosition || bCorrectTrajectory)
  {
    CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
    if(cFish == nullptr) { printf("Someone touched my fish\n"); abort(); }
    const double DT = sim.dt/Tperiod, time = sim.time;
    // Control pos diffs
    const double   xDiff = (centerOfMass[0] - origC[0])/length;
    const double   yDiff = (centerOfMass[1] - origC[1])/length;
    const double angDiff =  orientation     - origAng;
    const double relU = (u + sim.uinfx) / length;
    const double relV = (v + sim.uinfy) / length;
    const double angVel = omega, lastAngVel = cFish->lastAvel;
    // compute ang vel at t - 1/2 dt such that we have a better derivative:
    const double aVelMidP = (angVel + lastAngVel)*Tperiod/2;
    const double aVelDiff = (angVel - lastAngVel)*Tperiod/sim.dt;
    cFish->lastAvel = angVel; // store for next time

    // derivatives of following 2 exponential averages:
    const double velDAavg = (angDiff-cFish->avgDangle)/Tperiod + DT * angVel;
    const double velDYavg = (  yDiff-cFish->avgDeltaY)/Tperiod + DT * relV;
    const double velAVavg = 10*((aVelMidP-cFish->avgAngVel)/Tperiod +DT*aVelDiff);
    // exponential averages
    cFish->avgDangle = (1.0 -DT) * cFish->avgDangle +    DT * angDiff;
    cFish->avgDeltaY = (1.0 -DT) * cFish->avgDeltaY +    DT *   yDiff;
    // faster average:
    cFish->avgAngVel = (1-10*DT) * cFish->avgAngVel + 10*DT *aVelMidP;
    const double avgDangle = cFish->avgDangle, avgDeltaY = cFish->avgDeltaY;

    // integral (averaged) and proportional absolute DY and their derivative
    const double absPy = std::fabs(yDiff), absIy = std::fabs(avgDeltaY);
    const double velAbsPy =     yDiff>0 ? relV     : -relV;
    const double velAbsIy = avgDeltaY>0 ? velDYavg : -velDYavg;
    assert(origAng<2e-16 && "TODO: rotate pos and vel to fish POV to enable \
                             PID to work even for non-zero angles");

    if (bCorrectPosition && sim.dt>0)
    {
      //If angle is positive: positive curvature only if Dy<0 (must go up)
      //If angle is negative: negative curvature only if Dy>0 (must go down)
      const double IangPdy = (avgDangle *     yDiff < 0)? avgDangle * absPy : 0;
      const double PangIdy = (angDiff   * avgDeltaY < 0)? angDiff   * absIy : 0;
      const double IangIdy = (avgDangle * avgDeltaY < 0)? avgDangle * absIy : 0;

      // derivatives multiplied by 0 when term is inactive later:
      const double velIangPdy = velAbsPy * avgDangle + absPy * velDAavg;
      const double velPangIdy = velAbsIy * angDiff   + absIy * angVel;
      const double velIangIdy = velAbsIy * avgDangle + absIy * velDAavg;

      //zero also the derivatives when appropriate
      const double coefIangPdy = avgDangle *     yDiff < 0 ? 1 : 0;
      const double coefPangIdy = angDiff   * avgDeltaY < 0 ? 1 : 0;
      const double coefIangIdy = avgDangle * avgDeltaY < 0 ? 1 : 0;

      const double valIangPdy = coefIangPdy *    IangPdy;
      const double difIangPdy = coefIangPdy * velIangPdy;
      const double valPangIdy = coefPangIdy *    PangIdy;
      const double difPangIdy = coefPangIdy * velPangIdy;
      const double valIangIdy = coefIangIdy *    IangIdy;
      const double difIangIdy = coefIangIdy * velIangIdy;
      const double periodFac = 1.0 - xDiff;
      const double periodVel =     - relU;

      if(not sim.muteAll) {
        std::ofstream filePID;
        std::stringstream ssF;
        ssF<<sim.path2file<<"/PID_"<<obstacleID<<".dat";
        filePID.open(ssF.str().c_str(), std::ios::app);
        filePID<<time<<" "<<valIangPdy<<" "<<difIangPdy
                     <<" "<<valPangIdy<<" "<<difPangIdy
                     <<" "<<valIangIdy<<" "<<difIangIdy
                     <<" "<<periodFac <<" "<<periodVel <<"\n";
      }
      const double totalTerm = valIangPdy + valPangIdy + valIangIdy;
      const double totalDiff = difIangPdy + difPangIdy + difIangIdy;
      cFish->correctTrajectory(totalTerm, totalDiff, sim.time, sim.dt);
      cFish->correctTailPeriod(periodFac, periodVel, sim.time, sim.dt);
    }
    // if absIy<EPS then we have just one fish that the simulation box follows
    // therefore we control the average angle but not the Y disp (which is 0)
    else if (bCorrectTrajectory && sim.dt>0)
    {
      const double avgAngVel = cFish->avgAngVel, absAngVel = std::fabs(avgAngVel);
      const double absAvelDiff = avgAngVel>0? velAVavg : -velAVavg;
      const Real coefInst = angDiff*avgAngVel>0 ? 0.01 : 1, coefAvg = 0.1;
      const Real termInst = angDiff*absAngVel;
      const Real diffInst = angDiff*absAvelDiff + angVel*absAngVel;
      const double totalTerm = coefInst*termInst + coefAvg*avgDangle;
      const double totalDiff = coefInst*diffInst + coefAvg*velDAavg;

      if(not sim.muteAll) {
        std::ofstream filePID;
        std::stringstream ssF;
        ssF<<sim.path2file<<"/PID_"<<obstacleID<<".dat";
        filePID.open(ssF.str().c_str(), std::ios::app);
        filePID<<time<<" "<<coefInst*termInst<<" "<<coefInst*diffInst
                     <<" "<<coefAvg*avgDangle<<" "<<coefAvg*velDAavg<<"\n";
      }
      cFish->correctTrajectory(totalTerm, totalDiff, sim.time, sim.dt);
    }
  }
  Fish::create(vInfo);
}

void StefanFish::act(const Real t_rlAction, const std::vector<double>& a) const
{
  CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  cFish->execute(sim.time, t_rlAction, a);
}

double StefanFish::getLearnTPeriod() const
{
  const CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  return cFish->periodPIDval;
}

double StefanFish::getPhase(const double t) const
{
  const CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  const double T0 = cFish->time0;
  const double Ts = cFish->timeshift;
  const double Tp = cFish->periodPIDval;
  const double arg  = 2*M_PI*((t-T0)/Tp +Ts) + M_PI*phaseShift;
  const double phase = std::fmod(arg, 2*M_PI);
  return (phase<0) ? 2*M_PI + phase : phase;
}

std::vector<double> StefanFish::state() const
{
  const CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  std::vector<double> S(10,0);
  S[0] = ( center[0] - origC[0] )/ length;
  S[1] = ( center[1] - origC[1] )/ length;
  S[2] = getOrientation();
  S[3] = getPhase( sim.time );
  S[4] = getU() * Tperiod / length;
  S[5] = getV() * Tperiod / length;
  S[6] = getW() * Tperiod;
  S[7] = cFish->lastTact;
  S[8] = cFish->lastCurv;
  S[9] = cFish->oldrCurv;

  #ifndef STEFANS_SENSORS_STATE
    return S;
  #else
    S.resize(16);

    // Get velInfo
    const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo();

    // Get fish skin
    const auto &DU = myFish->upperSkin, &DL = myFish->lowerSkin;

    //// Sensor Signal on Front of Fish ////
    ////////////////////////////////////////

    std::array<Real,2> tipShear;

    // get front-point
    const std::array<Real,2> pFront = {DU.xSurf[0], DU.ySurf[0]};

    // get blockId for this point
    const size_t blockIdFront = holdingBlockID(pFront, velInfo);

    // get gridspacing and its inverse in block
    const Real hFront = velInfo[blockIdFront].h_gridpoint, invhFront = 1/hFront;

    // first point of the two skins is the same
    // normal should be almost the same: take the mean
    const Real normXfront = (DU.normXSurf[0] + DL.normXSurf[0]) / 2;
    const Real normYfront = (DU.normYSurf[0] + DL.normYSurf[0]) / 2;

    // compute point for sensor
    const std::array<Real,2> pFrontSens = {pFront[0] + hFront * normXfront,
                                      pFront[1] + hFront * normYfront};

    // get surface velocity and velocity at sensor
    const std::array<Real,2> vSensFront = sensVel(pFrontSens, velInfo), vSkinFront = skinVel(pFront, velInfo);

    tipShear[0] = (vSensFront[0] - vSkinFront[0]) * invhFront;
    tipShear[1] = (vSensFront[1] - vSkinFront[1]) * invhFront;

    //// Sensor Signal on Side of Fish ////
    ///////////////////////////////////////

    std::array<Real,2> lowShear, topShear;

    // get index for sensors on the side of head
    int iHeadSide = 0;
    for(int i=0; i<myFish->Nm-1; ++i)
      if( myFish->rS[i] <= 0.04*length && myFish->rS[i+1] > 0.04*length )
        iHeadSide = i;
    assert(iHeadSide>0);

    for(int a = 0; a<2; ++a)
    {
      // distinguish upper and lower skin
      const auto& D = a==0 ? myFish->upperSkin : myFish->lowerSkin;

      // get point
      const std::array<Real,2> pSide = {D.midX[iHeadSide], D.midY[iHeadSide]};

      // get blockId
      const size_t blockIdSide = holdingBlockID(pSide, velInfo);

      // get gridspacing and its inverse in block
      const Real hSide = velInfo[blockIdSide].h_gridpoint, invhSide = 1/hSide;

      // get normal to surface
      const Real normX = D.normXSurf[iHeadSide], normY = D.normYSurf[iHeadSide];

      // compute point at sensor location
      const std::array<Real,2> pSideSens = {pSide[0] + hSide * normX,
                                        pSide[1] + hSide * normY};

      // get surface velocity and velocity at sensor
      const std::array<Real,2> vSens = sensVel(pSideSens, velInfo), vSkin = skinVel(pSide, velInfo);

      // compute shear
      const Real shearX = (vSens[0] - vSkin[0]) * invhSide;
      const Real shearY = (vSens[1] - vSkin[1]) * invhSide;

      // now figure out how to rotate it along the fish skin for consistency:
      const Real dX = D.xSurf[iHeadSide+1] - D.xSurf[iHeadSide];
      const Real dY = D.ySurf[iHeadSide+1] - D.ySurf[iHeadSide];
      const Real proj = dX * normX - dY * normY;
      const Real tangX = proj>0?  normX : -normX; // s.t. tang points from head
      const Real tangY = proj>0? -normY :  normY; // to tail, normal outward
      (a==0? topShear[0] : lowShear[0]) = shearX * normX + shearY * normY;
      (a==0? topShear[1] : lowShear[1]) = shearX * tangX + shearY * tangY;
    }

    // put non-dimensional results into state into state
    S[10] = tipShear[0] * Tperiod / length;
    S[11] = tipShear[1] * Tperiod / length;
    S[12] = lowShear[0] * Tperiod / length;
    S[13] = lowShear[1] * Tperiod / length;
    S[14] = topShear[0] * Tperiod / length;
    S[15] = topShear[1] * Tperiod / length;
    // printf("shear tip:[%f %f] lower side:[%f %f] upper side:[%f %f]\n", S[10],S[11], S[12],S[13], S[14],S[15]);
    // fflush(0);
    return S;
  #endif
}

std::vector<double> StefanFish::state(Shape*const p) const
{
  const CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  std::vector<double> S(10,0);
  S[0] = ( center[0] - p->center[0] )/ length;
  S[1] = ( center[1] - p->center[1] )/ length;
  S[2] = getOrientation();
  S[3] = getPhase( sim.time );
  S[4] = getU() * Tperiod / length;
  S[5] = getV() * Tperiod / length;
  S[6] = getW() * Tperiod;
  S[7] = cFish->lastTact;
  S[8] = cFish->lastCurv;
  S[9] = cFish->oldrCurv;

  #ifndef STEFANS_SENSORS_STATE
    return S;
  #else
    S.resize(16);

    // Get velInfo
    const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo();

    // Get fish skin
    const auto &DU = myFish->upperSkin, &DL = myFish->lowerSkin;

    //// Sensor Signal on Front of Fish ////
    ////////////////////////////////////////

    std::array<Real,2> tipShear;

    // get front-point
    const std::array<Real,2> pFront = {DU.xSurf[0], DU.ySurf[0]};

    // get blockId for this point
    const size_t blockIdFront = holdingBlockID(pFront, velInfo);

    // get gridspacing and its inverse in block
    const Real hFront = velInfo[blockIdFront].h_gridpoint, invhFront = 1/hFront;

    // first point of the two skins is the same
    // normal should be almost the same: take the mean
    const Real normXfront = (DU.normXSurf[0] + DL.normXSurf[0]) / 2;
    const Real normYfront = (DU.normYSurf[0] + DL.normYSurf[0]) / 2;

    // compute point for sensor
    const std::array<Real,2> pFrontSens = {pFront[0] + hFront * normXfront,
                                      pFront[1] + hFront * normYfront};

    // get surface velocity and velocity at sensor
    const std::array<Real,2> vSensFront = sensVel(pFrontSens, velInfo), vSkinFront = skinVel(pFront, velInfo);

    tipShear[0] = (vSensFront[0] - vSkinFront[0]) * invhFront;
    tipShear[1] = (vSensFront[1] - vSkinFront[1]) * invhFront;

    //// Sensor Signal on Side of Fish ////
    ///////////////////////////////////////

    std::array<Real,2> lowShear, topShear;

    // get index for sensors on the side of head
    int iHeadSide = 0;
    for(int i=0; i<myFish->Nm-1; ++i)
      if( myFish->rS[i] <= 0.04*length && myFish->rS[i+1] > 0.04*length )
        iHeadSide = i;
    assert(iHeadSide>0);

    for(int a = 0; a<2; ++a)
    {
      // distinguish upper and lower skin
      const auto& D = a==0 ? myFish->upperSkin : myFish->lowerSkin;

      // get point
      const std::array<Real,2> pSide = {D.midX[iHeadSide], D.midY[iHeadSide]};

      // get blockId
      const size_t blockIdSide = holdingBlockID(pSide, velInfo);

      // get gridspacing and its inverse in block
      const Real hSide = velInfo[blockIdSide].h_gridpoint, invhSide = 1/hSide;

      // get normal to surface
      const Real normX = D.normXSurf[iHeadSide], normY = D.normYSurf[iHeadSide];

      // compute point at sensor location
      const std::array<Real,2> pSideSens = {pSide[0] + hSide * normX,
                                        pSide[1] + hSide * normY};

      // get surface velocity and velocity at sensor
      const std::array<Real,2> vSens = sensVel(pSideSens, velInfo), vSkin = skinVel(pSide, velInfo);

      // compute shear
      const Real shearX = (vSens[0] - vSkin[0]) * invhSide;
      const Real shearY = (vSens[1] - vSkin[1]) * invhSide;

      // now figure out how to rotate it along the fish skin for consistency:
      const Real dX = D.xSurf[iHeadSide+1] - D.xSurf[iHeadSide];
      const Real dY = D.ySurf[iHeadSide+1] - D.ySurf[iHeadSide];
      const Real proj = dX * normX - dY * normY;
      const Real tangX = proj>0?  normX : -normX; // s.t. tang points from head
      const Real tangY = proj>0? -normY :  normY; // to tail, normal outward
      (a==0? topShear[0] : lowShear[0]) = shearX * normX + shearY * normY;
      (a==0? topShear[1] : lowShear[1]) = shearX * tangX + shearY * tangY;
    }

    // put non-dimensional results into state into state
    S[10] = tipShear[0] * Tperiod / length;
    S[11] = tipShear[1] * Tperiod / length;
    S[12] = lowShear[0] * Tperiod / length;
    S[13] = lowShear[1] * Tperiod / length;
    S[14] = topShear[0] * Tperiod / length;
    S[15] = topShear[1] * Tperiod / length;
    // printf("shear tip:[%f %f] lower side:[%f %f] upper side:[%f %f]\n", S[10],S[11], S[12],S[13], S[14],S[15]);
    // fflush(0);
    return S;
  #endif
}

/* helpers to compute sensor information */

// function that finds block id of block containing pos (x,y)
size_t StefanFish::holdingBlockID(const std::array<Real,2> pos, const std::vector<cubism::BlockInfo>& velInfo) const
{
  for(size_t i=0; i<velInfo.size(); ++i)
  {
    // get gridspacing in block
    const Real h = velInfo[i].h_gridpoint;

    // compute lower left corner of block
    std::array<Real,2> MIN = velInfo[i].pos<Real>(0, 0);
    for(int j=0; j<2; ++j)
      MIN[j] -= 0.5 * h; // pos returns cell centers

    // compute top right corner of block
    std::array<Real,2> MAX = velInfo[i].pos<Real>(VectorBlock::sizeX-1, VectorBlock::sizeY-1);
    for(int j=0; j<2; ++j)
      MAX[j] += 0.5 * h; // pos returns cell centers

    // check whether point is inside block
    if( pos[0] >= MIN[0] && pos[1] >= MIN[1] && pos[0] <= MAX[0] && pos[1] <= MAX[1] )
    {
      // select obstacle block
      if(obstacleBlocks[i] != nullptr ){
        return i;
      }
    }
  }
  printf("ABORT: coordinate (%g,%g) could not be associated to obstacle block\n", pos[0], pos[1]);
  fflush(0); abort();
  return 0;
};

// function that gives indice of point in block
std::array<int, 2> StefanFish::safeIdInBlock(const std::array<Real,2> pos, const std::array<Real,2> org, const Real invh ) const
{
  const int indx = (int) std::round((pos[0] - org[0])*invh);
  const int indy = (int) std::round((pos[1] - org[1])*invh);
  const int ix = std::min( std::max(0, indx), VectorBlock::sizeX-1);
  const int iy = std::min( std::max(0, indy), VectorBlock::sizeY-1);
  return std::array<int, 2>{{ix, iy}};
};

// return fish velocity at a point on the fish skin:
std::array<Real, 2> StefanFish::skinVel(const std::array<Real,2> pSkin, const std::vector<cubism::BlockInfo>& velInfo) const
{
  // get blockId for pSkin
  const size_t blockId = holdingBlockID(pSkin, velInfo);

  // get block
  const auto& skinBinfo = velInfo[blockId];

  // get origin of block
  const std::array<Real,2> oSkin = skinBinfo.pos<Real>(0, 0);

  // get inverse gridspacing on this block
  const Real invh = 1/velInfo[blockId].h_gridpoint;

  // get index of point in block
  const std::array<int,2> iSkin = safeIdInBlock(pSkin, oSkin, invh);

  //printf("skin pos:[%f %f] -> block org:[%f %f] ind:[%d %d]\n",
  //  pSkin[0], pSkin[1], oSkin[0], oSkin[1], iSkin[0], iSkin[1]);

  // get deformation velocity
  const Real* const udef = obstacleBlocks[skinBinfo.blockID]->udef[iSkin[1]][iSkin[0]];

  // compute velocity of skin point
  const Real uSkin = u - omega * (pSkin[1]-centerOfMass[1]) + udef[0];
  const Real vSkin = v + omega * (pSkin[0]-centerOfMass[0]) + udef[1];

  return std::array<Real, 2>{{uSkin, vSkin}};
};

// return flow velocity at point of flow sensor:
std::array<Real, 2> StefanFish::sensVel(const std::array<Real,2> pSens, const std::vector<cubism::BlockInfo>& velInfo) const
{
  // get blockId
  const size_t blockId = holdingBlockID(pSens, velInfo);

  // get block
  const auto& sensBinfo = velInfo[blockId];

  // get origin of block
  const std::array<Real,2> oSens = sensBinfo.pos<Real>(0, 0);

  // get inverse gridspacing in block
  const Real invh = 1/velInfo[blockId].h_gridpoint;

  // get index for sensor
  const std::array<int,2> iSens = safeIdInBlock(pSens, oSens, invh);

  //printf("sensor pos:[%f %f] -> block org:[%f %f] ind:[%d %d]\n",
  //  pSens[0], pSens[1], oSens[0], oSens[1], iSens[0], iSens[1]);

  // get velocity field at point
  const VectorBlock& b = * (const VectorBlock*) sensBinfo.ptrBlock;

  return std::array<Real, 2>{{b(iSens[0], iSens[1]).u[0],
                              b(iSens[0], iSens[1]).u[1]}};
};

//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "StefanFish.h"
#include <sstream>
#include <iomanip>

using namespace cubism;

//bool TperiodPID = false;

void StefanFish::resetAll() {
  CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  if(cFish == nullptr) { printf("Someone touched my fish\n"); abort(); }
  cFish->resetAll();
  Fish::resetAll();
}

void StefanFish::saveRestart( FILE * f ) {
  assert(f != NULL);
  Fish::saveRestart(f);
  CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  std::stringstream ss;
  ss<<std::setfill('0')<<std::setw(7)<<"_"<<obstacleID<<"_";
  cFish->curvatureScheduler.save("curvatureScheduler"+ ss.str() + ".restart");
  cFish->periodScheduler.save("periodScheduler"+ ss.str() + ".restart");
  cFish->rlBendingScheduler.save("rlBendingScheduler"+ ss.str() + ".restart");

  //Save these numbers for PID controller and other stuff. Maybe not all of them are needed
  //but we don't care, it's only a few numbers.
  fprintf(f, "curv_PID_fac: %20.20e\n", cFish->curv_PID_fac);
  fprintf(f, "curv_PID_dif: %20.20e\n", cFish->curv_PID_dif);
  fprintf(f, "avgDeltaY   : %20.20e\n", cFish->avgDeltaY   );
  fprintf(f, "avgDangle   : %20.20e\n", cFish->avgDangle   );
  fprintf(f, "avgAngVel   : %20.20e\n", cFish->avgAngVel   );
  fprintf(f, "lastTact    : %20.20e\n", cFish->lastTact    );
  fprintf(f, "lastCurv    : %20.20e\n", cFish->lastCurv    );
  fprintf(f, "oldrCurv    : %20.20e\n", cFish->oldrCurv    );
  fprintf(f, "periodPIDval: %20.20e\n", cFish->periodPIDval);
  fprintf(f, "periodPIDdif: %20.20e\n", cFish->periodPIDdif);
  fprintf(f, "time0       : %20.20e\n", cFish->time0       );
  fprintf(f, "timeshift   : %20.20e\n", cFish->timeshift   );
  fprintf(f, "lastTime    : %20.20e\n", cFish->lastTime    );
  fprintf(f, "lastAvel    : %20.20e\n", cFish->lastAvel    );
}

void StefanFish::loadRestart( FILE * f ) {
  assert(f != NULL);
  Fish::loadRestart(f);
  CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  std::stringstream ss;
  ss<<std::setfill('0')<<std::setw(7)<<"_"<<obstacleID<<"_";
  cFish->curvatureScheduler.restart("curvatureScheduler"+ ss.str() + ".restart");
  cFish->periodScheduler.restart("periodScheduler"+ ss.str() + ".restart");
  cFish->rlBendingScheduler.restart("rlBendingScheduler"+ ss.str() + ".restart");

  bool ret = true;
  ret = ret && 1==fscanf(f, "curv_PID_fac: %le\n", &cFish->curv_PID_fac);
  ret = ret && 1==fscanf(f, "curv_PID_dif: %le\n", &cFish->curv_PID_dif);
  ret = ret && 1==fscanf(f, "avgDeltaY   : %le\n", &cFish->avgDeltaY   );
  ret = ret && 1==fscanf(f, "avgDangle   : %le\n", &cFish->avgDangle   );
  ret = ret && 1==fscanf(f, "avgAngVel   : %le\n", &cFish->avgAngVel   );
  ret = ret && 1==fscanf(f, "lastTact    : %le\n", &cFish->lastTact    );
  ret = ret && 1==fscanf(f, "lastCurv    : %le\n", &cFish->lastCurv    );
  ret = ret && 1==fscanf(f, "oldrCurv    : %le\n", &cFish->oldrCurv    );
  ret = ret && 1==fscanf(f, "periodPIDval: %le\n", &cFish->periodPIDval);
  ret = ret && 1==fscanf(f, "periodPIDdif: %le\n", &cFish->periodPIDdif);
  ret = ret && 1==fscanf(f, "time0       : %le\n", &cFish->time0       );
  ret = ret && 1==fscanf(f, "timeshift   : %le\n", &cFish->timeshift   );
  ret = ret && 1==fscanf(f, "lastTime    : %le\n", &cFish->lastTime    );
  ret = ret && 1==fscanf(f, "lastAvel    : %le\n", &cFish->lastAvel    );
  if( (not ret) ) {
    printf("Error reading restart file. Aborting...\n");
    fflush(0); abort();
  }
}


StefanFish::StefanFish(SimulationData&s, ArgumentParser&p, Real C[2]):
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
  if( sim.rank == 0 && s.verbose ) printf("[CUP2D] - CurvatureFish %d %f %f %f %f %f %f\n",myFish->Nm, (double)length,(double)myFish->dSref,(double)myFish->dSmid,(double)sim.minH, (double)Tperiod, (double)phaseShift);
}

//static inline Real sgn(const Real val) { return (0 < val) - (val < 0); }
void StefanFish::create(const std::vector<BlockInfo>& vInfo)
{
  // If PID controller to keep position or swim straight enabled
  if (bCorrectPosition || bCorrectTrajectory)
  {
    CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
    if(cFish == nullptr) { printf("Someone touched my fish\n"); abort(); }
    const Real DT = sim.dt/Tperiod;//, time = sim.time;
    // Control pos diffs
    const Real   xDiff = (centerOfMass[0] - origC[0])/length;
    const Real   yDiff = (centerOfMass[1] - origC[1])/length;
    const Real angDiff =  orientation     - origAng;
    const Real relU = (u + sim.uinfx) / length;
    const Real relV = (v + sim.uinfy) / length;
    const Real angVel = omega, lastAngVel = cFish->lastAvel;
    // compute ang vel at t - 1/2 dt such that we have a better derivative:
    const Real aVelMidP = (angVel + lastAngVel)*Tperiod/2;
    const Real aVelDiff = (angVel - lastAngVel)*Tperiod/sim.dt;
    cFish->lastAvel = angVel; // store for next time

    // derivatives of following 2 exponential averages:
    const Real velDAavg = (angDiff-cFish->avgDangle)/Tperiod + DT * angVel;
    const Real velDYavg = (  yDiff-cFish->avgDeltaY)/Tperiod + DT * relV;
    const Real velAVavg = 10*((aVelMidP-cFish->avgAngVel)/Tperiod +DT*aVelDiff);
    // exponential averages
    cFish->avgDangle = (1.0 -DT) * cFish->avgDangle +    DT * angDiff;
    cFish->avgDeltaY = (1.0 -DT) * cFish->avgDeltaY +    DT *   yDiff;
    // faster average:
    cFish->avgAngVel = (1-10*DT) * cFish->avgAngVel + 10*DT *aVelMidP;
    const Real avgDangle = cFish->avgDangle, avgDeltaY = cFish->avgDeltaY;

    // integral (averaged) and proportional absolute DY and their derivative
    const Real absPy = std::fabs(yDiff), absIy = std::fabs(avgDeltaY);
    const Real velAbsPy =     yDiff>0 ? relV     : -relV;
    const Real velAbsIy = avgDeltaY>0 ? velDYavg : -velDYavg;
    assert(origAng<2e-16 && "TODO: rotate pos and vel to fish POV to enable \
                             PID to work even for non-zero angles");

    if (bCorrectPosition && sim.dt>0)
    {
      //If angle is positive: positive curvature only if Dy<0 (must go up)
      //If angle is negative: negative curvature only if Dy>0 (must go down)
      const Real IangPdy = (avgDangle *     yDiff < 0)? avgDangle * absPy : 0;
      const Real PangIdy = (angDiff   * avgDeltaY < 0)? angDiff   * absIy : 0;
      const Real IangIdy = (avgDangle * avgDeltaY < 0)? avgDangle * absIy : 0;

      // derivatives multiplied by 0 when term is inactive later:
      const Real velIangPdy = velAbsPy * avgDangle + absPy * velDAavg;
      const Real velPangIdy = velAbsIy * angDiff   + absIy * angVel;
      const Real velIangIdy = velAbsIy * avgDangle + absIy * velDAavg;

      //zero also the derivatives when appropriate
      const Real coefIangPdy = avgDangle *     yDiff < 0 ? 1 : 0;
      const Real coefPangIdy = angDiff   * avgDeltaY < 0 ? 1 : 0;
      const Real coefIangIdy = avgDangle * avgDeltaY < 0 ? 1 : 0;

      const Real valIangPdy = coefIangPdy *    IangPdy;
      const Real difIangPdy = coefIangPdy * velIangPdy;
      const Real valPangIdy = coefPangIdy *    PangIdy;
      const Real difPangIdy = coefPangIdy * velPangIdy;
      const Real valIangIdy = coefIangIdy *    IangIdy;
      const Real difIangIdy = coefIangIdy * velIangIdy;
      const Real periodFac = 1.0 - xDiff;
      const Real periodVel =     - relU;
#if 0
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
#endif
      const Real totalTerm = valIangPdy + valPangIdy + valIangIdy;
      const Real totalDiff = difIangPdy + difPangIdy + difIangIdy;
      cFish->correctTrajectory(totalTerm, totalDiff, sim.time, sim.dt);
      cFish->correctTailPeriod(periodFac, periodVel, sim.time, sim.dt);
    }
    // if absIy<EPS then we have just one fish that the simulation box follows
    // therefore we control the average angle but not the Y disp (which is 0)
    else if (bCorrectTrajectory && sim.dt>0)
    {
      const Real avgAngVel = cFish->avgAngVel, absAngVel = std::fabs(avgAngVel);
      const Real absAvelDiff = avgAngVel>0? velAVavg : -velAVavg;
      const Real coefInst = angDiff*avgAngVel>0 ? 0.01 : 1, coefAvg = 0.1;
      const Real termInst = angDiff*absAngVel;
      const Real diffInst = angDiff*absAvelDiff + angVel*absAngVel;
      const Real totalTerm = coefInst*termInst + coefAvg*avgDangle;
      const Real totalDiff = coefInst*diffInst + coefAvg*velDAavg;

#if 0
      if(not sim.muteAll) {
        std::ofstream filePID;
        std::stringstream ssF;
        ssF<<sim.path2file<<"/PID_"<<obstacleID<<".dat";
        filePID.open(ssF.str().c_str(), std::ios::app);
        filePID<<time<<" "<<coefInst*termInst<<" "<<coefInst*diffInst
                     <<" "<<coefAvg*avgDangle<<" "<<coefAvg*velDAavg<<"\n";
      }
#endif
      cFish->correctTrajectory(totalTerm, totalDiff, sim.time, sim.dt);
    }
  }
  Fish::create(vInfo);
}

void StefanFish::act(const Real t_rlAction, const std::vector<Real>& a) const
{
  CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  cFish->execute(sim.time, t_rlAction, a);
}

Real StefanFish::getLearnTPeriod() const
{
  const CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  //return cFish->periodPIDval;
  return cFish->next_period;
}

Real StefanFish::getPhase(const Real t) const
{
  const CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  const Real T0 = cFish->time0;
  const Real Ts = cFish->timeshift;
  const Real Tp = cFish->periodPIDval;
  const Real arg  = 2*M_PI*((t-T0)/Tp +Ts) + M_PI*phaseShift;
  const Real phase = std::fmod(arg, 2*M_PI);
  return (phase<0) ? 2*M_PI + phase : phase;
}

std::vector<Real> StefanFish::state( const std::vector<double>& origin ) const
{
  const CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  std::vector<Real> S(16,0);
  S[0] = ( center[0] - origin[0] )/ length;
  S[1] = ( center[1] - origin[1] )/ length;
  S[2] = getOrientation();
  S[3] = getPhase( sim.time );
  S[4] = getU() * Tperiod / length;
  S[5] = getV() * Tperiod / length;
  S[6] = getW() * Tperiod;
  S[7] = cFish->lastTact;
  S[8] = cFish->lastCurv;
  S[9] = cFish->oldrCurv;

    S.resize(16);

    // Get velInfo
    const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo();

    // Get fish skin
    const auto &DU = myFish->upperSkin, &DL = myFish->lowerSkin;

    //// Sensor Signal on Front of Fish ////
    ////////////////////////////////////////

    // get front-point
    const std::array<Real,2> pFront = {DU.xSurf[0], DU.ySurf[0]};

    // first point of the two skins is the same
    // normal should be almost the same: take the mean
    const std::array<Real,2> normalFront = { (DU.normXSurf[0] + DL.normXSurf[0]) / 2,
                                             (DU.normYSurf[0] + DL.normYSurf[0]) / 2 };

    // compute shear stress
    std::array<Real,2> tipShear = getShear( pFront, normalFront, velInfo );

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
      const std::array<Real,2> pSide = { D.midX[iHeadSide], D.midY[iHeadSide] };

      // get normal to surface
      const std::array<Real,2> normSide = { D.normXSurf[iHeadSide], D.normYSurf[iHeadSide] };

      // compute shear stress
      std::array<Real,2> sideShear = getShear( pSide, normSide, velInfo );

      // now figure out how to rotate it along the fish skin for consistency:
      const Real dX = D.xSurf[iHeadSide+1] - D.xSurf[iHeadSide];
      const Real dY = D.ySurf[iHeadSide+1] - D.ySurf[iHeadSide];
      const Real proj = dX * normSide[0] - dY * normSide[1];
      const Real tangX = proj>0?  normSide[0] : -normSide[0]; // s.t. tang points from head
      const Real tangY = proj>0? -normSide[1] :  normSide[1]; // to tail, normal outward
      (a==0? topShear[0] : lowShear[0]) = sideShear[0] * normSide[0] + sideShear[1] * normSide[1];
      (a==0? topShear[1] : lowShear[1]) = sideShear[0] * tangX + sideShear[1] * tangY;
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

  #ifndef STEFANS_NEIGHBOUR_STATE
    return S;
  #else 
    S.resize(22);

    // Get all shapes in simulaton
    const auto& shapes = sim.shapes;
    const size_t N = shapes.size();

    // Compute distance vector pointing from agent to neighbours
    std::vector<std::vector<Real>> distanceVector(N, std::vector<Real>(2));
    std::vector<std::pair<Real, size_t>> distances(N);
    for( size_t i = 0; i<N; i++ )
    {
      const double distX = shapes[i]->center[0] - center[0];
      const double distY = shapes[i]->center[1] - center[1];
      distanceVector[i][0] = distX;
      distanceVector[i][1] = distY;
      distances[i].first = std::sqrt( distX*distX + distY*distY );
      distances[i].second = i; 
    }

    // Only add distance vector for (first) three neighbours to state
    std::sort( distances.begin(), distances.end() );
    for( size_t i = 0; i<3; i++ )
    {
      // Ignore first entry, which will be the distance to itself
      S[16+2*i]   = distanceVector[distances[i+1].second][0];
      S[16+2*i+1] = distanceVector[distances[i+1].second][1];
    }

    return S;
  #endif
}

/* helpers to compute sensor information */

// function that finds block id of block containing pos (x,y)
ssize_t StefanFish::holdingBlockID(const std::array<Real,2> pos, const std::vector<cubism::BlockInfo>& velInfo) const
{
  for(size_t i=0; i<velInfo.size(); ++i)
  {
    // get gridspacing in block
    const Real h = velInfo[i].h;

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
      // point lies inside this block
      return i;
    }
  }
  // rank does not contain point
  return -1;
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

// returns shear at given surface location
std::array<Real, 2> StefanFish::getShear(const std::array<Real,2> pSurf, const std::array<Real,2> normSurf, const std::vector<cubism::BlockInfo>& velInfo) const
{
  // Buffer to broadcast velcities and gridspacing
  Real velocityH[3] = {0.0, 0.0, 0.0};

  // 1. Compute surface velocity on surface
  // get blockId of surface
  ssize_t blockIdSurf = holdingBlockID(pSurf, velInfo);

  // get surface velocity if block containing point found
  char error = false;
  if( blockIdSurf >= 0 ) {
    // get block
    const auto& skinBinfo = velInfo[blockIdSurf];

    // check whether obstacle block exists
    if(obstacleBlocks[blockIdSurf] == nullptr ){
      printf("[CUP2D, rank %u] velInfo[%lu] contains point (%f,%f), but obstacleBlocks[%lu] is a nullptr! obstacleBlocks.size()=%lu\n", sim.rank, blockIdSurf, pSurf[0], pSurf[1], blockIdSurf, obstacleBlocks.size());
      const std::vector<cubism::BlockInfo>& chiInfo = sim.chi->getBlocksInfo();
      const auto& chiBlock = chiInfo[blockIdSurf];
      ScalarBlock & __restrict__ CHI = *(ScalarBlock*) chiBlock.ptrBlock;
      for( size_t i = 0; i<ScalarBlock::sizeX; i++) 
      for( size_t j = 0; j<ScalarBlock::sizeY; j++)
      {
        const auto pos = chiBlock.pos<Real>(i, j);
        printf("i,j=%ld,%ld: pos=(%f,%f) with chi=%f\n", i, j, pos[0], pos[1], CHI(i,j).s);
      }
      fflush(0);
      error = true;
      // abort();
    }
    else{
      // get origin of block
      const std::array<Real,2> oBlockSkin = skinBinfo.pos<Real>(0, 0);

      // get gridspacing on this block
      velocityH[2] = velInfo[blockIdSurf].h;

      // get index of point in block
      const std::array<int,2> iSkin = safeIdInBlock(pSurf, oBlockSkin, 1/velocityH[2]);

      // get deformation velocity
      const Real udefX = obstacleBlocks[blockIdSurf]->udef[iSkin[1]][iSkin[0]][0];
      const Real udefY = obstacleBlocks[blockIdSurf]->udef[iSkin[1]][iSkin[0]][1];

      // compute velocity of skin point
      velocityH[0] = u - omega * (pSurf[1]-centerOfMass[1]) + udefX;
      velocityH[1] = v + omega * (pSurf[0]-centerOfMass[0]) + udefY;
    }
  }

  // DEBUG purposes
  #if 1
  MPI_Allreduce(MPI_IN_PLACE, &blockIdSurf, 1, MPI_INT64_T, MPI_MAX, sim.chi->getCartComm());
  if( sim.rank == 0 && blockIdSurf == -1 )
  {
    printf("ABORT: coordinate (%g,%g) could not be associated to ANY obstacle block\n", (double)pSurf[0], (double)pSurf[1]);
    fflush(0);
    abort();
  }

  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_CHAR, MPI_LOR, sim.chi->getCartComm());
  if( error )
  {
    sim.dumpAll("failed");
    abort();
  }
  #endif

  // Allreduce to Bcast surface velocity
  MPI_Allreduce(MPI_IN_PLACE, velocityH, 3, MPI_Real, MPI_SUM, sim.chi->getCartComm());

  // Assign skin velocities and grid-spacing
  const Real uSkin = velocityH[0];
  const Real vSkin = velocityH[1];
  const Real h     = velocityH[2];
  const Real invh = 1/h;

  // Reset buffer to 0
  velocityH[0] = 0.0; velocityH[1] = 0.0; velocityH[2] = 0.0;

  // 2. Compute flow velocity away from surface
  // compute point on lifted surface
  const std::array<Real,2> pLiftedSurf = { pSurf[0] + h * normSurf[0],
                                           pSurf[1] + h * normSurf[1] };

  // get blockId of lifted surface
  const ssize_t blockIdLifted = holdingBlockID(pLiftedSurf, velInfo);

  // get surface velocity if block containing point found
  if( blockIdLifted >= 0 ) {
    // get block
    const auto& liftedBinfo = velInfo[blockIdLifted];

    // get origin of block
    const std::array<Real,2> oBlockLifted = liftedBinfo.pos<Real>(0, 0);

    // get inverse gridspacing in block
    const Real invhLifted = 1/velInfo[blockIdLifted].h;

    // get index for sensor
    const std::array<int,2> iSens = safeIdInBlock(pLiftedSurf, oBlockLifted, invhLifted);

    // get velocity field at point
    const VectorBlock& b = * (const VectorBlock*) liftedBinfo.ptrBlock;
    velocityH[0] = b(iSens[0], iSens[1]).u[0];
    velocityH[1] = b(iSens[0], iSens[1]).u[1];
  }

  // Allreduce to Bcast flow velocity
  MPI_Allreduce(MPI_IN_PLACE, velocityH, 3, MPI_Real, MPI_SUM, sim.chi->getCartComm());

  // Assign lifted skin velocities
  const Real uLifted = velocityH[0];
  const Real vLifted = velocityH[1];

  // return shear
  return std::array<Real, 2>{{(uLifted - uSkin) * invh,
                              (vLifted - vSkin) * invh }};

};

void CurvatureFish::computeMidline(const Real t, const Real dt)
{
  periodScheduler.transition(t,transition_start,transition_start+transition_duration,current_period,next_period);
  periodScheduler.gimmeValues(t,periodPIDval,periodPIDdif);
  if (transition_start < t && t < transition_start+transition_duration)//timeshift also rampedup
  {
	  timeshift = (t - time0)/periodPIDval + timeshift;
	  time0 = t;
  }

  // define interpolation points on midline
  const std::array<Real ,6> curvaturePoints = { (Real)0, (Real).15*length,
    (Real).4*length, (Real).65*length, (Real).9*length, length
  };
  // define values of curvature at interpolation points
  const std::array<Real ,6> curvatureValues = {
      (Real)0.82014/length, (Real)1.46515/length, (Real)2.57136/length,
      (Real)3.75425/length, (Real)5.09147/length, (Real)5.70449/length
  };
  // define interpolation points for RL action
  const std::array<Real,7> bendPoints = {(Real)-.5, (Real)-.25, (Real)0,
    (Real).25, (Real).5, (Real).75, (Real)1};

  // transition curvature from 0 to target values
  #if 1 // ramp-up over Tperiod
  const std::array<Real,6> curvatureZeros = std::array<Real, 6>();
  curvatureScheduler.transition(0,0,Tperiod,curvatureZeros ,curvatureValues);
  #else // no rampup for debug
  curvatureScheduler.transition(t,0,Tperiod,curvatureValues,curvatureValues);
  #endif

  // write curvature values
  curvatureScheduler.gimmeValues(t, curvaturePoints, Nm, rS, rC, vC);
  rlBendingScheduler.gimmeValues(t, periodPIDval, length, bendPoints, Nm, rS, rB, vB);

  // next term takes into account the derivative of periodPIDval in darg:
  const Real diffT = 1 - (t-time0)*periodPIDdif/periodPIDval;
  // time derivative of arg:
  const Real darg = 2*M_PI/periodPIDval * diffT;
  const Real arg0 = 2*M_PI*((t-time0)/periodPIDval +timeshift) +M_PI*phaseShift;

  #pragma omp parallel for schedule(static)
  for(int i=0; i<Nm; ++i) {
    const Real arg = arg0 - 2*M_PI*rS[i]/length;
    rK[i] = amplitudeFactor* rC[i]*(std::sin(arg)     + rB[i] +curv_PID_fac);
    vK[i] = amplitudeFactor*(vC[i]*(std::sin(arg)     + rB[i] +curv_PID_fac)
                            +rC[i]*(std::cos(arg)*darg+ vB[i] +curv_PID_dif));
    assert(not std::isnan(rK[i]));
    assert(not std::isinf(rK[i]));
    assert(not std::isnan(vK[i]));
    assert(not std::isinf(vK[i]));
  }

  // solve frenet to compute midline parameters
  IF2D_Frenet2D::solve(Nm, rS, rK,vK, rX,rY, vX,vY, norX,norY, vNorX,vNorY);
  #if 0
   {
    FILE * f = fopen("stefan_profile","w");
    for(int i=0;i<Nm;++i)
      fprintf(f,"%d %g %g %g %g %g %g %g %g %g\n",
        i,rS[i],rX[i],rY[i],vX[i],vY[i],
        vNorX[i],vNorY[i],width[i],height[i]);
    fclose(f);
   }
  #endif
}

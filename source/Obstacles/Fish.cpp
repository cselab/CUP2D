//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "Fish.h"
#include "FishData.h"
//#include <sstream>
//#include <iomanip>

using namespace cubism;

//#define profile( arg ) do { profiler.arg; } while (0)
#define profile( func ) do { } while (0)

void Fish::create(const std::vector<BlockInfo>& vInfo)
{
  //// 0) clear obstacle blocks
  for(auto & entry : obstacleBlocks) 
    delete entry;
  obstacleBlocks.clear();

  //// 1) Update Midline and compute surface
  assert(myFish!=nullptr);
  profile(push_start("midline"));
  myFish->computeMidline(sim.time, sim.dt);
  myFish->computeSurface();
  profile(pop_stop());

  //// 2) Integrate Linear and Angular Momentum and shift Fish accordingly
  profile(push_start("2dmoments"));
  // returns area, CoM_internal, vCoM_internal:
  area_internal = myFish->integrateLinearMomentum(CoM_internal, vCoM_internal);
  // takes CoM_internal, vCoM_internal, puts CoM in and nullifies  lin mom:
  myFish->changeToCoMFrameLinear(CoM_internal, vCoM_internal);
  angvel_internal_prev = angvel_internal;
  // returns mom of intertia and angvel:
  J_internal = myFish->integrateAngularMomentum(angvel_internal);
  // rotates fish midline to current angle and removes angular moment:
  myFish->changeToCoMFrameAngular(theta_internal, angvel_internal);
  #if 0 //ndef NDEBUG
  {
    Real dummy_CoM_internal[2], dummy_vCoM_internal[2], dummy_angvel_internal;
    // check that things are zero
    const Real area_internal_check =
    myFish->integrateLinearMomentum(dummy_CoM_internal, dummy_vCoM_internal);
    myFish->integrateAngularMomentum(dummy_angvel_internal);
    const Real EPS = 10*std::numeric_limits<Real>::epsilon();
    assert(std::fabs(dummy_CoM_internal[0])<EPS);
    assert(std::fabs(dummy_CoM_internal[1])<EPS);
    assert(std::fabs(myFish->linMom[0])<EPS);
    assert(std::fabs(myFish->linMom[1])<EPS);
    assert(std::fabs(myFish->angMom)<EPS);
    assert(std::fabs(area_internal - area_internal_check) < EPS);
  }
  #endif
  profile(pop_stop());
  myFish->surfaceToCOMFrame(theta_internal, CoM_internal);

  //// 3) Create Bounding Boxes around Fish
  //- performance of create seems to decrease if VolumeSegment_OBB are bigger
  //- this code groups segments together and finds a bounding box (maximal
  //  x and y coords) to then be able to check intersection with cartesian grid
  const int Nsegments = (myFish->Nm-1)/8, Nm = myFish->Nm;
  assert((Nm-1)%Nsegments==0);
  profile(push_start("boxes"));

  std::vector<AreaSegment*> vSegments(Nsegments, nullptr);
  #pragma omp parallel for schedule(static)
  for(int i=0; i<Nsegments; ++i)
  {
    const int next_idx = (i+1)*(Nm-1)/Nsegments, idx = i * (Nm-1)/Nsegments;
    // find bounding box based on this
    Real bbox[2][2] = {{1e9, -1e9}, {1e9, -1e9}};
    for(int ss=idx; ss<=next_idx; ++ss)
    {
      const Real xBnd[2]={myFish->rX[ss] -myFish->norX[ss]*myFish->width[ss],
                          myFish->rX[ss] +myFish->norX[ss]*myFish->width[ss]};
      const Real yBnd[2]={myFish->rY[ss] -myFish->norY[ss]*myFish->width[ss],
                          myFish->rY[ss] +myFish->norY[ss]*myFish->width[ss]};
      const Real maxX=std::max(xBnd[0],xBnd[1]), minX=std::min(xBnd[0],xBnd[1]);
      const Real maxY=std::max(yBnd[0],yBnd[1]), minY=std::min(yBnd[0],yBnd[1]);
      bbox[0][0] = std::min(bbox[0][0], minX);
      bbox[0][1] = std::max(bbox[0][1], maxX);
      bbox[1][0] = std::min(bbox[1][0], minY);
      bbox[1][1] = std::max(bbox[1][1], maxY);
    }
    const Real DD = 4*sim.getH(); //two points on each side
    //const Real safe_distance = info.h_gridpoint; // one point on each side
    AreaSegment*const tAS=new AreaSegment(std::make_pair(idx,next_idx),bbox,DD);
    tAS->changeToComputationalFrame(center, orientation);
    vSegments[i] = tAS;
  }
  profile(pop_stop());

  //// 4) Interpolate shape with computational grid
  profile(push_start("intersect"));
  const auto N = vInfo.size();
  std::vector<std::vector<AreaSegment*>*> segmentsPerBlock (N, nullptr);
  obstacleBlocks = std::vector<ObstacleBlock*> (N, nullptr);

  #pragma omp parallel for schedule(static)
  for(size_t i=0; i<vInfo.size(); ++i)
  {
    const BlockInfo & info = vInfo[i];
    Real pStart[2], pEnd[2];
    info.pos(pStart, 0, 0);
    info.pos(pEnd, ScalarBlock::sizeX-1, ScalarBlock::sizeY-1);

    for(size_t s=0; s<vSegments.size(); ++s)
      if(vSegments[s]->isIntersectingWithAABB(pStart,pEnd))
      {
        if(segmentsPerBlock[info.blockID] == nullptr)
          segmentsPerBlock[info.blockID] = new std::vector<AreaSegment*>(0);
        segmentsPerBlock[info.blockID]->push_back(vSegments[s]);
      }

    // allocate new blocks if necessary
    if(segmentsPerBlock[info.blockID] not_eq nullptr)
    {
      assert(obstacleBlocks[info.blockID] == nullptr);
      ObstacleBlock * const block = new ObstacleBlock();
      assert(block not_eq nullptr);
      obstacleBlocks[info.blockID] = block;
      block->clear();
    }
  }
  assert(not segmentsPerBlock.empty());
  assert(segmentsPerBlock.size() == obstacleBlocks.size());
  profile(pop_stop());

  #pragma omp parallel
  {
    const PutFishOnBlocks putfish(*myFish, center, orientation);

    #pragma omp for schedule(dynamic)
    for(size_t i=0; i<vInfo.size(); i++)
    {
      const auto pos = segmentsPerBlock[vInfo[i].blockID];
      if(pos not_eq nullptr)
      {
        ObstacleBlock*const block = obstacleBlocks[vInfo[i].blockID];
        assert(block not_eq nullptr);
        putfish(vInfo[i], *(ScalarBlock*)vInfo[i].ptrBlock, block, *pos);
      }
    }
  }

  // clear vSegments
  for(auto & E : vSegments) { if(E not_eq nullptr) delete E; }
  for(auto & E : segmentsPerBlock)  { if(E not_eq nullptr) delete E; }

  profile(pop_stop());
  if (sim.step % 100 == 0 && sim.verbose)
  {
    profile(printSummary());
    profile(reset());
  }
}

void Fish::updatePosition(double dt)
{
  // update position and angles
  Shape::updatePosition(dt);
  theta_internal -= dt*angvel_internal; // negative: we subtracted this angvel
}

void Fish::resetAll()
{
  CoM_internal[0] = 0; CoM_internal[1] = 0;
  vCoM_internal[0] = 0; vCoM_internal[1] = 0;
  theta_internal = 0; angvel_internal = 0; angvel_internal_prev = 0;
  Shape::resetAll();
  myFish->resetAll();
}

Fish::~Fish()
{
  if(myFish not_eq nullptr) {
    delete myFish;
    myFish = nullptr;
  }
}

void Fish::removeMoments(const std::vector<cubism::BlockInfo>& vInfo)
{
  Shape::removeMoments(vInfo);
  myFish->surfaceToComputationalFrame(orientation, centerOfMass);
  myFish->computeSkinNormals(orientation, centerOfMass);
  #if 0
  {
    std::stringstream ssF;
    ssF<<"skinPoints"<<std::setfill('0')<<std::setw(9)<<sim.step<<".dat";
    std::ofstream ofs (ssF.str().c_str(), std::ofstream::out);
    for(size_t i=0; i<myFish->upperSkin.Npoints; ++i)
      ofs<<myFish->upperSkin.xSurf[i]  <<" "<<myFish->upperSkin.ySurf[i]<<" " <<myFish->upperSkin.normXSurf[i]  <<" "<<myFish->upperSkin.normYSurf[i]  <<"\n";
    for(size_t i=myFish->lowerSkin.Npoints; i>0; --i)
      ofs<<myFish->lowerSkin.xSurf[i-1]<<" "<<myFish->lowerSkin.ySurf[i-1]<<" "<<myFish->lowerSkin.normXSurf[i-1]<<" "<<myFish->lowerSkin.normYSurf[i-1]<<"\n";
    ofs.flush();
    ofs.close();
  }
  #endif
}

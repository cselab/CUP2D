#pragma once

#include "cuda_runtime.h"
#include "../include/helper_cuda.h"

class DeviceProfiler
{
public:
  DeviceProfiler() : elapsed_(0.)
  {
    checkCudaErrors(cudaEventCreate(&start_));
    checkCudaErrors(cudaEventCreate(&stop_));
  }

  ~DeviceProfiler() 
  {
    checkCudaErrors(cudaEventDestroy(start_));
    checkCudaErrors(cudaEventDestroy(stop_));
  }

  void startProfiler(cudaStream_t stream)
  {
    checkCudaErrors(cudaEventRecord(start_, stream));
  }

  void stopProfiler(cudaStream_t stream)
  {
    checkCudaErrors(cudaEventRecord(stop_, stream));
    checkCudaErrors(cudaEventSynchronize(stop_));

    float event_time = 0.;
    checkCudaErrors(cudaEventElapsedTime(&event_time, start_, stop_));
    elapsed_ += event_time;
  }
  float elapsed() {return elapsed_; }

protected:
  float elapsed_;
  cudaEvent_t start_;
  cudaEvent_t stop_;
};

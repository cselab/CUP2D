#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

#include "Poisson/helper_cuda.h"
#include "Poisson/bicgstab.h"

double rmsVec(
    const std::vector<double>& a,
    const std::vector<double>& b);

int main(){

  // ----------------------- Test BiCGSTAB ---------------------------
  int N = 10;
  int nnz; // Init counter at zero
  std::vector<double> cooValA;
  std::vector<int> cooRowA;
  std::vector<int> cooColA;
  std::vector<double> x_true(N);
  std::vector<double> x(N, 1.);
  std::vector<double> b(N, 0.);


  // Iterate from the back to produce unsorted COO 1D 2nd order FD matrix
  nnz = 0;
  for(int i(N-1); i >= 0; i--)
  {
    // Set solution
    x_true[i] = (i % 3) + 1;

    if(i == (N-1))
    { // Set lone diagonal element to ensure inversibility
      cooValA.push_back(-2.);
      // cooValA.push_back(unif(rd));
      cooRowA.push_back(i);
      cooColA.push_back(i);
      nnz++;

      // Set RHS vec given the solution
      b[i] += (x_true[i] * cooValA.back()); 
    }
    else
    {
      // Set diagonal element
      cooValA.push_back(-2.);
      // cooValA.push_back(unif(rd));
      cooRowA.push_back(i);
      cooColA.push_back(i);
      nnz++;
      // Add contribution from element to RHS vec
      b[i] += (x_true[i] * cooValA.back()); 

      // Set off diagonal element
      cooValA.push_back(1.);
      // cooValA.push_back(unif(rd));
      cooRowA.push_back(i);
      cooColA.push_back(i+1);
      nnz++;
      // Add contribution from element to RHS vec
      b[i] += (x_true[i+1] * cooValA.back()); 
    }
  }

  std::cout << "BiCGSTAB launch... \n";

  // Call BiCGSTAB
  BiCGSTAB(
      N, 
      N, 
      nnz, 
      cooValA.data(), 
      cooRowA.data(), 
      cooColA.data(), 
      x.data(),
      b.data(),
      1e-20,
      1e-20,
      0);
  
  double rms = rmsVec(x_true, x);
  std::cout << "BiCGSTAB: RMS error between true and estimate: " << rms << std::endl;;

    
  // ----------------------------------- test preconditioning ---------------------
  std::random_device rd;
  std::uniform_real_distribution<double> unif(-5., 5.);

  int bsz = 64; // block size BSX*BSY
  N = 100 * bsz;
  std::vector<double> Q(bsz * bsz);
  std::vector<double> v(N);
  std::vector<double> z(N);
  std::vector<double> z_true(N);

  for (int i(0); i < bsz * bsz; i++){
    Q[i] = unif(rd);
  }
  for (int i(0); i < N; i++){
    v[i] = unif(rd);
  }
  for (int i(0); i < N; i++){
    z_true[i] = 0.;
    for (int j(0); j < bsz; j++){
      z_true[i] += Q[(i % bsz) * bsz + j]*v[i - i % bsz + j];
    }
  }

  double* d_Q;
  double* d_v;
  double* d_z;
  checkCudaErrors(cudaMalloc(&d_Q, bsz * bsz * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_v, N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_z, N * sizeof(double)));
  checkCudaErrors(cudaMemcpy(d_Q, Q.data(), bsz * bsz * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_v, v.data(), N * sizeof(double), cudaMemcpyHostToDevice));

  preconditionVec<<<100, bsz, (bsz +bsz*bsz)*sizeof(double)>>>(N, d_Q, d_v);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(z.data(), d_v, N * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_Q));
  checkCudaErrors(cudaFree(d_v));
  checkCudaErrors(cudaFree(d_z));

  rms = rmsVec(z_true, z);
  std::cout << "Preconditionner: RMS error between true and estimate: " << rms << std::endl;;
  return 0;
}

double rmsVec(
    const std::vector<double>& a,
    const std::vector<double>& b)
{
  size_t sz = std::min(a.size(), b.size());
  double rms = 0;
  for (int i(0); i < sz; i++)
  {
    rms += std::pow(a[i] - b[i],2.);
  }
  return std::sqrt(rms);
}

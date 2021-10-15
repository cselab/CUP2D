#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

#include "Poisson/bicgstab.h"

void genSparseLinSys(
    int N, 
    int &nnz,
    std::vector<double>& cooValA,
    std::vector<int>& cooRowA,
    std::vector<int>& cooColA,
    std::vector<double>& x,
    std::vector<double>& b);

double rmsVec(
    const std::vector<double>& a,
    const std::vector<double>& b);

int main(){

  // Construct an inversible sparse linear system
  int N = 10;
  int nnz = 0;
  std::vector<double> cooValA;
  std::vector<int> cooRowA;
  std::vector<int> cooColA;
  std::vector<double> x_true(N);
  std::vector<double> x_est(N, 1.);
  std::vector<double> b(N, 0.);
  genSparseLinSys(N, nnz, cooValA, cooRowA, cooColA, x_true, b);

  std::cout << "BiCGSTAB launch... \n";

  // Call BiCGSTAB
  BiCGSTAB(
      N, 
      N, 
      nnz, 
      cooValA.data(), 
      cooRowA.data(), 
      cooColA.data(), 
      x_est.data(),
      b.data(),
      1e-20,
      1e-20);
  
  double rms = rmsVec(x_true, x_est);
  std::cout << "BiCGSTAB done, RMS error between true and estimate: " << rms << std::endl;;

  return 0;
}

void genSparseLinSys(
    int N, 
    int &nnz,
    std::vector<double>& cooValA,
    std::vector<int>& cooRowA,
    std::vector<int>& cooColA,
    std::vector<double>& x,
    std::vector<double>& b)
{
  // Initialize random number generator
  // std::random_device rd;
  // std::uniform_real_distribution<double> unif(-5., 5.);

  // Iterate from the back to produce unsorted COO LinSys
  nnz = 0;
  for(int i(N-1); i >= 0; i--)
  {
    // Set solution
    // x[i] = unif(rd);
    x[i] = (i % 3) + 1;

    if(i == (N-1))
    { // Set lone diagonal element to ensure inversibility
      cooValA.push_back(-2.);
      // cooValA.push_back(unif(rd));
      cooRowA.push_back(i);
      cooColA.push_back(i);
      nnz++;

      // Set RHS vec given the solution
      b[i] += (x[i] * cooValA.back()); 
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
      b[i] += (x[i] * cooValA.back()); 

      // Set off diagonal element
      cooValA.push_back(1.);
      // cooValA.push_back(unif(rd));
      cooRowA.push_back(i);
      cooColA.push_back(i+1);
      nnz++;
      // Add contribution from element to RHS vec
      b[i] += (x[i+1] * cooValA.back()); 
    }
  }
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

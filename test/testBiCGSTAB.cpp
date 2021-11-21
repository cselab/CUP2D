#include <iostream>
#include <vector>

#include "test.h"
#include "../source/Poisson/helper_cuda.h"
#include "../source/Poisson/bicgstab.h"

int main(){

  // ----------------------- Test BiCGSTAB ---------------------------
  int B = 64;
  int N = 1000 * B;
  int nnz; // Init counter at zero
  std::vector<double> cooValA;
  std::vector<int> cooRowA;
  std::vector<int> cooColA;
  std::vector<double> P(B * B, 0.); // preconditioner
  std::vector<double> x_true(N);
  std::vector<double> x(N, -1.);
  std::vector<double> b(N, 0.);

  // Make identity
  for (int i(0); i < B; i++) P[i*B + i] = 1.;

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
      10);
  double error = rms(x.size(), x_true.data(), x.data());
  std::cout << "BiCGSTAB: RMS error between true and estimate: " << error << std::endl;

  // Re-initialize x
  x = std::vector<double>(N, -1.);
  std::cout << "pBiCGSTAB launch... \n";
  pBiCGSTAB(
      N, 
      N, 
      nnz, 
      cooValA.data(), 
      cooRowA.data(), 
      cooColA.data(), 
      x.data(),
      b.data(),
      B,
      P.data(),
      1e-20,
      1e-20,
      10);
  error = rms(x.size(), x_true.data(), x.data());
  std::cout << "pBiCGSTAB: RMS error between true and estimate: " << error << std::endl;

    
 return 0;
}

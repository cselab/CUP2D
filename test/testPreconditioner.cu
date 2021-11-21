#include <iostream>
#include <vector>

#include "test.h"
#include "../source/Poisson/helper_cuda.h"
#include "../source/Poisson/bicgstab.h"

static constexpr int BSX = 8;
static constexpr int BSY = 8;


double getA_local(int I1,int I2);
void makePreconditioner_CPU(
    std::vector<std::vector<double>> &L,
    std::vector<double> &Ld,
    std::vector<std::vector<std::pair<int,double>>> &L_row,
    std::vector<std::vector<std::pair<int,double>>> &L_col);
void makePreconditioner_GPU(
    std::vector<std::vector<double>> &L,
    std::vector<std::vector<double>> &L_inv,
    std::vector<double> &P_,
    std::vector<double> &P_inv_);
void preconditionVec_CPU(
    std::vector<double> &z,
    std::vector<double> &Ld,
    std::vector<std::vector<std::pair<int,double>>> &L_row,
    std::vector<std::vector<std::pair<int,double>>> &L_col);

int main(){

  int B = BSX*BSY; // block size BSX*BSY
  int N = 141056 * B;

  std::vector<double> z_init(N);
  for (int i(0); i < N; i++) 
    z_init[i] = std::pow(-1.,i) * std::sqrt(i) * double(i) / double(N);
  std::vector<double> z_cpu{z_init};
  std::vector<double> z_gpu{z_init};

  std::vector<std::vector<double>> L_cpu;
  std::vector<double> Ld;
  std::vector<std::vector<std::pair<int,double>>> L_row;
  std::vector<std::vector<std::pair<int,double>>> L_col;
  makePreconditioner_CPU(L_cpu, Ld, L_row, L_col);
  preconditionVec_CPU(z_cpu, Ld, L_row, L_col);

  std::cout << "L_cpu: \n";
  for (int i(0); i < L_cpu.size(); i++){
    print_mat(1, i, L_cpu[i].data());
  }
  
  std::vector<std::vector<double>> L_gpu;
  std::vector<std::vector<double>> L_inv;
  std::vector<double> P(B*B, 0.);
  std::vector<double> P_inv;
  makePreconditioner_GPU(L_gpu, L_inv, P, P_inv);

  double* d_P_inv;
  double* d_z;
  checkCudaErrors(cudaMalloc(&d_P_inv, B * B * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_z, N * sizeof(double)));
  checkCudaErrors(cudaMemcpy(d_P_inv, P_inv.data(), B * B * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_z, z_init.data(), N * sizeof(double), cudaMemcpyHostToDevice));

  preconditionVec<<<1000, B, (B +B*B)*sizeof(double)>>>(N, d_P_inv, d_z);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(z_gpu.data(), d_z, N * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_P_inv));
  checkCudaErrors(cudaFree(d_z));

//  std::vector<double> matmul_res(B*B, 0.);
//  rm_gemm(B, B, B, P_inv.data(), P.data(), matmul_res.data());
//
//  std::cout << "Matrix inversion: \n";
//  print_mat(B, B, matmul_res.data());

  int print_offset = 0*B;
  std::cout << "z_init: \n";
  print_mat(1, B, z_init.data()+print_offset);
  std::cout << "z_cpu: \n";
  print_mat(1, B, z_cpu.data()+print_offset);
  std::cout << "z_gpu: \n";
  print_mat(1, B, z_gpu.data()+print_offset);

  std::vector<double> z_gemv(N, 0.);
  for (int i(0); i < N; i+=B)
    rm_gemv(B, B, P_inv.data(), z_init.data()+i, z_gemv.data()+i);
  std::cout << "z_gemv: \n";
  print_mat(1, B, z_gemv.data() + print_offset);

  std::vector<double> z_cholesky{ z_init };
  for (int i(0); i < N; i+=B)
    cholesky_fb_substitution(L_gpu, z_cholesky.data()+i);
  std::cout << "z_cholesky: \n";
  print_mat(1, B, z_cholesky.data() + print_offset);

  std::cout << "||z_cpu|| = " << nrm2(z_cpu.size(), z_cpu.data()) << std::endl;
  std::cout << "||z_gpu|| = " << nrm2(z_gpu.size(), z_gpu.data()) << std::endl;
  std::cout << "||z_cpu - z_gpu|| = " << rms(z_gpu.size(), z_gpu.data(), z_cpu.data()) << std::endl;
  std::cout << "||z_gemv - z_gpu|| = " << rms(z_gpu.size(), z_gpu.data(), z_gemv.data()) << std::endl;
  std::cout << "||z_cholesky - z_gpu|| = " << rms(z_gpu.size(), z_gpu.data(), z_cholesky.data()) << std::endl;

  

  return 0;
}

double getA_local(int I1,int I2) //matrix for Poisson's equation on a uniform grid
{
  int j1 = I1 / BSX;
  int i1 = I1 % BSX;
  int j2 = I2 / BSY;
  int i2 = I2 % BSY;
  if (i1==i2 && j1==j2)
  {
    return 4.0;
  }
  else if (abs(i1-i2) + abs(j1-j2) == 1)
  {
    return -1.0;
  }
  else
  {
    return 0.0;
  }
}  

void makePreconditioner_CPU(
    std::vector<std::vector<double>> &L,
    std::vector<double> &Ld,
    std::vector<std::vector<std::pair<int,double>>> &L_row,
    std::vector<std::vector<std::pair<int,double>>> &L_col)
{ //here we compute the Cholesky decomposition of the preconditioner (we do this only once) but we save one inverse for every thread
  int N = BSX*BSY;
  L.resize(N);

  for (int i = 0 ; i<N ; i++)
  {
    L[i].resize(i+1);
  }
  for (int i = 0 ; i<N ; i++)
  {
    double s1=0;
    for (int k=0; k<=i-1; k++)
      s1 += L[i][k]*L[i][k];
    L[i][i] = std::sqrt(getA_local(i,i) - s1);
    for (int j=i+1; j<N; j++)
    {
      double s2 = 0;
      for (int k=0; k<=i-1; k++)
        s2 += L[i][k]*L[j][k];
      L[j][i] = (getA_local(j,i)-s2) / L[i][i];
    }
  }

  L_row.resize(N);
  L_col.resize(N);
  for (int i = 0 ; i<N ; i++)
  {
    Ld.push_back(1.0/L[i][i]);
    for (int j = 0 ; j < i ; j++)
    {
      if ( std::abs(L[i][j]) > 1e-10 ) 
        L_row[i].push_back({j,L[i][j]});
    }
  }
  for (int j = 0 ; j<N ; j++)
  {
    for (int i = j ; i < N ; i++)
    {
      if ( std::abs(L[i][j]) > 1e-10 ) 
        L_col[j].push_back({i,L[i][j]});
    }
  }
}

void makePreconditioner_GPU(
    std::vector<std::vector<double>> &L,
    std::vector<std::vector<double>> &L_inv,
    std::vector<double> &P_,
    std::vector<double> &P_inv_)
{
  const int N = BSX*BSY;
  L.resize(N);
  L_inv.resize(N);
  for (int i = 0 ; i<N ; i++)
  {
    L[i].resize(i+1);
    L_inv[i].resize(i+1);
    // L_inv will act as right block in GJ algorithm, init it as identity
    for (int j(0); j<=i; j++){
      L_inv[i][j] = (i == j) ? 1. : 0.;
    }
  }

  // compute the Cholesky decomposition of the preconditioner with Cholesky-Crout
  for (int i = 0 ; i<N ; i++)
  {
    double s1=0;
    for (int k=0; k<=i-1; k++)
      s1 += L[i][k]*L[i][k];
    L[i][i] = std::sqrt(getA_local(i,i) - s1);
    for (int j=i+1; j<N; j++)
    {
      double s2 = 0;
      for (int k=0; k<=i-1; k++)
        s2 += L[i][k]*L[j][k];
      L[j][i] = (getA_local(j,i)-s2) / L[i][i];
    }
  }

//  for (int i = 0 ; i<N ; i++)
//  for (int j(0); j<i; j++)
//  {
//    if (std::abs(L[i][j]) < 1e-10) std::cout << "(" << i << "," << j << ") TOO SMALL \n";
//    L[i][j] = std::abs(L[i][j]) < 1e-10 ? 0 : L[i][j];
//  }

  /* Compute the inverse of the Cholesky decomposition L using Gauss-Jordan elimination.
     L will act as the left block (it does not need to be modified in the process), 
     L_inv will act as the right block and at the end of the algo will contain the inverse*/
  for (int br(0); br<N; br++)
    { // 'br' - base row in which all columns up to L_lb[br][br] are already zero
    const double bsf = 1. / L[br][br]; // scaling factor for base row
    for (int c(0); c<=br; c++)
    {
      L_inv[br][c] *= bsf;
    }

    for (int wr(br+1); wr<N; wr++)
    { // 'wr' - working row where elements below L_lb[br][br] will be set to zero
      const double wsf = L[wr][br];
      for (int c(0); c<=br; c++)
      { // For the right block matrix the trasformation has to be applied for the whole row
        L_inv[wr][c] -= (wsf * L_inv[br][c]);
      }
    }
  }

  // P_inv_ holds inverse preconditionner in row major order!  This is leads to better memory access
  // in the kernel that applies this preconditioner, but note that cuBLAS assumes column major
  // matrices by default
  P_inv_.resize(N * N); // use linear indexing for this matrix
  for (int i(0); i<N; i++){
    for (int j(0); j<N; j++){
      double aux = 0.;
      for (int k(0); k<N; k++){
        aux += (i <= k && j <=k) ? L_inv[k][i] * L_inv[k][j] : 0.; // P_inv_ = (L^T)^{-1} L^{-1}
      }
      P_inv_[i*N+j] = aux;
    }
  }

  // Construct fully P
  for (int i(0); i < N; i++)
  for (int j(0); j < N; j++)
    P_[i*N + j] = getA_local(i,j);
}

void preconditionVec_CPU(
    std::vector<double> &z,
    std::vector<double> &Ld,
    std::vector<std::vector<std::pair<int,double>>> &L_row,
    std::vector<std::vector<std::pair<int,double>>> &L_col)
{
  const int N   = BSX*BSY;

  for (size_t i=0; i < z.size(); i+=N) //go over the blocks
  {
    //1. z = L^{-1}r
    for (int I = 0; I < N ; I++)
    {
      double rhs = 0.0;
      for (size_t jj = 0 ; jj < L_row[I].size(); jj++)
      {
        const int J = L_row[I][jj].first;
        const int iy = J / BSX;
        const int ix = J % BSX;
        const int k = i + iy*BSX + ix;
        rhs += L_row[I][jj].second*z[k];
      }
      const int iy = I / BSX;
      const int ix = I % BSX;
      const int k = i + iy*BSX + ix;
      z[k] = (z[k] - rhs)*Ld[I];
    }
    //2. z = L^T{-1}r
    for (int I = N-1; I >= 0 ; I--)
    {
      double rhs = 0.0;
      for (size_t jj = 0 ; jj < L_col[I].size(); jj++)
      {
        const int J = L_col[I][jj].first;
        const int iy = J / BSX;
        const int ix = J % BSX;
        const int k = i + iy*BSX + ix;
        rhs += L_col[I][jj].second*z[k];
      }
      const int iy = I / BSX;
      const int ix = I % BSX;
      const int k = i + iy*BSX + ix;
      z[k] = (z[k] - rhs) *Ld[I];
    }

    for (int iy=0;iy<BSY;iy++){
      for (int ix=0;ix<BSX;ix++){
        const int k = i + iy*BSX + ix;
        z[k] = -z[k];
      }
    }
  }
}



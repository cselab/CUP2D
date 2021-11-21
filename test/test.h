#pragma once

#include <cmath>
#include <iostream>

double nrm2(
    const int N,
    const double a[])
{
  double aux = 0.;
  for (int i(0); i < N; i++)
    aux += a[i]*a[i];

  return std::sqrt(aux);
}


double rms(
    const int N,
    const double a[],
    const double b[])
{
  double rms = 0;
  for (int i(0); i < N; i++)
    rms += std::pow(a[i] - b[i],2.);

  return std::sqrt(rms);
}

void print_mat( // print row-major matrix
    const int m,
    const int n,
    const double A[])
{
  for (int i(0); i < m; i++)
  {
    for (int j(0); j < n; j++){
      double val = A[i*n + j];
      val = (std::abs(val) < 1e-15) ? 0 : val;
      if (j < n - 1)
        std::cout << val << ", ";
      else 
        std::cout << val << std::endl;
    }
  }
}

void rm_gemv( // row-major matrix-vector multiplication
    const int m,
    const int n,
    const double A[],
    const double x[],
    double y[])
{
  for (int i(0); i < m; i++)
  {
    double aux = 0.;
    for (int j(0); j < n; j++)
    {
      aux += A[i*n + j] * x[j];
    }
    y[i] = aux;
  }
}

void rm_gemm( // row-major matrix-matrix multiplication
    const int m,
    const int n,
    const int k,
    const double A[],
    const double B[],
    double C[])
{
  for (int i(0); i < m; i++)
  for (int j(0); j < n; j++)
  {
    double aux = 0.;
    for (int q(0); q < k; q++)
    {
      aux += A[i*k + q] * B[q*n + j];
    }
    C[i*n + j] = aux;
  }
}

void cholesky_fb_substitution( // forward and backward substitution
    std::vector<std::vector<double>> &L,
    double x[])
{
  // store in row major L + L^T for convenience
  std::vector<double> LLT(L.size() * L.size());
  for (int i(0); i < L.size(); i++)
  for (int j(0); j < L.size(); j++)
    LLT[i*L.size() + j] = (j <= i) ? L[i][j] : L[j][i];

  // forward subsititution
  for (int i(0); i < L.size(); i++)
  {
    double rhs = 0.;   
    for (int j(0); j < i; j++)
      rhs += LLT[i*L.size()+j] * x[j];

    x[i] = (x[i] - rhs) / LLT[i*L.size()+i];
  }

  // backward substitution
  for (int i(L.size()-1); i>=0; i--)
  {
    double rhs = 0.;
    for (int j(L.size()-1); j > i; j--) 
      rhs += LLT[i*L.size()+j] * x[j];

    x[i] = (x[i] - rhs) / LLT[i*L.size()+i];
  }
}

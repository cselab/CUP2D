#pragma once

extern "C" void BiCGSTAB(
    const int m, // rows
    const int n, // cols
    const int nnz, // no. of non-zero elements
    const double* const h_cooValA,
    const int* const h_cooRowA,
    const int* const h_cooColA,
    double* const h_x,
    const double* const h_b,
    const double max_error,
    const double max_rel_error,
    const int max_restarts); // if max_restarts == 0 defaults to normal BiCGSTAB without tricks

extern "C" void pBiCGSTAB(
    const int m, // rows
    const int n, // cols
    const int nnz, // no. of non-zero elements
    const double* const h_cooValA,
    const int* const h_cooRowA,
    const int* const h_cooColA,
    double* const h_x, // contains initial guess
    const double* const h_b,
    const int B, // block size BSX * BSY
    const double* const h_P_inv,
    const double max_error,
    const double max_rel_error,
    const int max_restarts); // if max_restarts == 0 defaults to normal BiCGSTAB without tricks

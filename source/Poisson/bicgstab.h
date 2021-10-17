#pragma once

extern "C" void BiCGSTAB(
    const int m, // rows
    const int n, // cols
    const int nnz, // no. of non-zero elements
    double* const h_cooValA,
    int* const h_cooRowA,
    int* const h_cooColA,
    double* const h_x,
    double* const h_b,
    const double max_error,
    const double max_rel_error,
    const int max_restarts);

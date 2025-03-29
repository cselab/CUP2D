#pragma once
#include <cassert>
#include <iostream>
#include <math.h>
#include <vector>
namespace cubism {
class SpaceFillingCurve {
protected:
  int BX;
  int BY;
  int BZ;
  int levelMax;
  bool isRegular;
  int base_level;
  std::vector<std::vector<long long>> Zsave;
  std::vector<std::vector<int>> i_inverse;
  std::vector<std::vector<int>> j_inverse;
  std::vector<std::vector<int>> k_inverse;
  long long AxestoTranspose(const int *X_in, int b) const {
    if (b == 0) {
      assert(X_in[0] == 0);
      assert(X_in[1] == 0);
      assert(X_in[2] == 0);
      return 0;
    }
    const int n = 3;
    int X[3] = {X_in[0], X_in[1], X_in[2]};
    assert(b - 1 >= 0);
    int M = 1 << (b - 1), P, Q, t;
    int i;
    for (Q = M; Q > 1; Q >>= 1) {
      P = Q - 1;
      for (i = 0; i < n; i++)
        if (X[i] & Q)
          X[0] ^= P;
        else {
          t = (X[0] ^ X[i]) & P;
          X[0] ^= t;
          X[i] ^= t;
        }
    }
    for (i = 1; i < n; i++)
      X[i] ^= X[i - 1];
    t = 0;
    for (Q = M; Q > 1; Q >>= 1)
      if (X[n - 1] & Q)
        t ^= Q - 1;
    for (i = 0; i < n; i++)
      X[i] ^= t;
    long long retval = 0;
    long long a = 0;
    const long long one = 1;
    const long long two = 2;
    for (long long level = 0; level < b; level++) {
      const long long a0 = ((one) << (a)) * ((long long)X[2] >> level & one);
      const long long a1 =
          ((one) << (a + one)) * ((long long)X[1] >> level & one);
      const long long a2 =
          ((one) << (a + two)) * ((long long)X[0] >> level & one);
      retval += a0 + a1 + a2;
      a += 3;
    }
    return retval;
  }
  void TransposetoAxes(long long index, long long *X, int b) const {
    const int n = 3;
    X[0] = 0;
    X[1] = 0;
    X[2] = 0;
    if (b == 0 && index == 0)
      return;
    long long aa = 0;
    const long long one = 1;
    const long long two = 2;
    for (long long i = 0; index > 0; i++) {
      long long x2 = index % two;
      index = index / two;
      long long x1 = index % two;
      index = index / two;
      long long x0 = index % two;
      index = index / two;
      X[0] += x0 * (one << aa);
      X[1] += x1 * (one << aa);
      X[2] += x2 * (one << aa);
      aa += 1;
    }
    int N = 2 << (b - 1), P, Q, t;
    int i;
    t = X[n - 1] >> 1;
    for (i = n - 1; i >= 1; i--)
      X[i] ^= X[i - 1];
    X[0] ^= t;
    for (Q = 2; Q != N; Q <<= 1) {
      P = Q - 1;
      for (i = n - 1; i >= 0; i--)
        if (X[i] & Q)
          X[0] ^= P;
        else {
          t = (X[0] ^ X[i]) & P;
          X[0] ^= t;
          X[i] ^= t;
        }
    }
  }

public:
  SpaceFillingCurve(){};
  SpaceFillingCurve(int a_BX, int a_BY, int a_BZ, int lmax)
      : BX(a_BX), BY(a_BY), BZ(a_BZ), levelMax(lmax) {
    int n_max = std::max(std::max(BX, BY), BZ);
    base_level = (log(n_max) / log(2));
    if (base_level < (double)(log(n_max) / log(2)))
      base_level++;
    i_inverse.resize(lmax);
    j_inverse.resize(lmax);
    k_inverse.resize(lmax);
    Zsave.resize(lmax);
    {
      const int l = 0;
      int aux = pow(pow(2, l), 3);
      i_inverse[l].resize(BX * BY * BZ * aux, -1);
      j_inverse[l].resize(BX * BY * BZ * aux, -1);
      k_inverse[l].resize(BX * BY * BZ * aux, -1);
      Zsave[l].resize(BX * BY * BZ * aux, -1);
    }
    isRegular = true;
#pragma omp parallel for collapse(3)
    for (int k = 0; k < BZ; k++)
      for (int j = 0; j < BY; j++)
        for (int i = 0; i < BX; i++) {
          const int c[3] = {i, j, k};
          long long index = AxestoTranspose(c, base_level);
          long long substract = 0;
          for (long long h = 0; h < index; h++) {
            long long X[3] = {0, 0, 0};
            TransposetoAxes(h, X, base_level);
            if (X[0] >= BX || X[1] >= BY || X[2] >= BZ)
              substract++;
          }
          index -= substract;
          if (substract > 0)
            isRegular = false;
          i_inverse[0][index] = i;
          j_inverse[0][index] = j;
          k_inverse[0][index] = k;
          Zsave[0][k * BX * BY + j * BX + i] = index;
        }
  }
  long long forward(const int l, const int i, const int j, const int k) {
    const int aux = 1 << l;
    if (l >= levelMax)
      return 0;
    long long retval;
    if (!isRegular) {
      const int I = i / aux;
      const int J = j / aux;
      const int K = k / aux;
      assert(!(I >= BX || J >= BY || K >= BZ));
      const int c2_a[3] = {i - I * aux, j - J * aux, k - K * aux};
      retval = AxestoTranspose(c2_a, l);
      retval += IJK_to_index(I, J, K) * aux * aux * aux;
    } else {
      const int c2_a[3] = {i, j, k};
      retval = AxestoTranspose(c2_a, l + base_level);
    }
    return retval;
  }
  void inverse(long long Z, int l, int &i, int &j, int &k) {
    if (isRegular) {
      long long X[3] = {0, 0, 0};
      TransposetoAxes(Z, X, l + base_level);
      i = X[0];
      j = X[1];
      k = X[2];
    } else {
      long long aux = 1 << l;
      long long Zloc = Z % (aux * aux * aux);
      long long X[3] = {0, 0, 0};
      TransposetoAxes(Zloc, X, l);
      long long index = Z / (aux * aux * aux);
      int I, J, K;
      index_to_IJK(index, I, J, K);
      i = X[0] + I * aux;
      j = X[1] + J * aux;
      k = X[2] + K * aux;
    }
    return;
  }
  long long IJK_to_index(int I, int J, int K) {
    long long index = Zsave[0][(J + K * BY) * BX + I];
    return index;
  }
  void index_to_IJK(long long index, int &I, int &J, int &K) {
    I = i_inverse[0][index];
    J = j_inverse[0][index];
    K = k_inverse[0][index];
    return;
  }
  long long Encode(int level, long long Z, int index[3]) {
    int lmax = levelMax;
    long long retval = 0;
    int ix = index[0];
    int iy = index[1];
    int iz = index[2];
    for (int l = level; l >= 0; l--) {
      long long Zp = forward(l, ix, iy, iz);
      retval += Zp;
      ix /= 2;
      iy /= 2;
      iz /= 2;
    }
    ix = 2 * index[0];
    iy = 2 * index[1];
    iz = 2 * index[2];
    for (int l = level + 1; l < lmax; l++) {
      long long Zc = forward(l, ix, iy, iz);
      Zc -= Zc % 8;
      retval += Zc;
      int ix1, iy1, iz1;
      ix1 = ix;
      iy1 = iy;
      iz1 = iz;
      inverse(Zc, l, ix1, iy1, iz1);
      ix = 2 * ix1;
      iy = 2 * iy1;
      iz = 2 * iz1;
    }
    retval += level;
    return retval;
  }
};
} // namespace cubism

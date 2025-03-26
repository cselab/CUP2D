#pragma once
#include "SpaceFillingCurve2D.h"
#include <array>
#include <cassert>
namespace cubism {
enum State : signed char { Leave = 0, Refine = 1, Compress = -1 };
struct TreePosition {
  int position{-3};
  bool CheckCoarser() const { return position == -2; }
  bool CheckFiner() const { return position == -1; }
  bool Exists() const { return position >= 0; }
  int rank() const { return position; }
  void setrank(const int r) { position = r; }
  void setCheckCoarser() { position = -2; }
  void setCheckFiner() { position = -1; }
};
struct BlockInfo {
  long long blockID;
  long long blockID_2;
  long long Z;
  long long Znei[3][3][3];
  long long halo_block_id;
  long long Zparent;
  long long Zchild[2][2][2];
  double h;
  double origin[3];
  int index[3];
  int level;
  void *ptrBlock{nullptr};
  void *auxiliary;
  bool changed2;
  State state;
  static int levelMax(int l = 0) {
    static int lmax = l;
    return lmax;
  }
  static int blocks_per_dim(int i, int nx = 0, int ny = 0) {
    static int a[2] = {nx, ny};
    return a[i];
  }
  static SpaceFillingCurve2D *SFC() {
    static SpaceFillingCurve2D Zcurve(blocks_per_dim(0), blocks_per_dim(1),
                                      levelMax());
    return &Zcurve;
  }
  static long long forward(int level, int ix, int iy) {
    return (*SFC()).forward(level, ix, iy);
  }
  static long long Encode(int level, long long Z, int index[2]) {
    return (*SFC()).Encode(level, Z, index);
  }
  static void inverse(long long Z, int l, int &i, int &j) {
    (*SFC()).inverse(Z, l, i, j);
  }
  template <typename T> inline void pos(T p[2], int ix, int iy) const {
    p[0] = origin[0] + h * (ix + 0.5);
    p[1] = origin[1] + h * (iy + 0.5);
  }
  template <typename T> inline std::array<T, 2> pos(int ix, int iy) const {
    std::array<T, 2> result;
    pos(result.data(), ix, iy);
    return result;
  }
  bool operator<(const BlockInfo &other) const {
    return (blockID_2 < other.blockID_2);
  }
  BlockInfo() {};
  void setup(const int a_level, const double a_h, const double a_origin[3],
             const long long a_Z) {
    level = a_level;
    Z = a_Z;
    state = Leave;
    level = a_level;
    h = a_h;
    origin[0] = a_origin[0];
    origin[1] = a_origin[1];
    origin[2] = a_origin[2];
    changed2 = true;
    auxiliary = nullptr;
    const int TwoPower = 1 << level;
    inverse(Z, level, index[0], index[1]);
    index[2] = 0;
    const int Bmax[3] = {blocks_per_dim(0) * TwoPower,
                         blocks_per_dim(1) * TwoPower, 1};
    for (int i = -1; i < 2; i++)
      for (int j = -1; j < 2; j++)
        for (int k = -1; k < 2; k++)
          Znei[i + 1][j + 1][k + 1] =
              forward(level, (index[0] + i + Bmax[0]) % Bmax[0],
                      (index[1] + j + Bmax[1]) % Bmax[1]);
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++)
          Zchild[i][j][k] =
              forward(level + 1, 2 * index[0] + i, 2 * index[1] + j);
    Zparent = (level == 0)
                  ? 0
                  : forward(level - 1, (index[0] / 2 + Bmax[0]) % Bmax[0],
                            (index[1] / 2 + Bmax[1]) % Bmax[1]);
    blockID_2 = Encode(level, Z, index);
    blockID = blockID_2;
  }
  long long Znei_(const int i, const int j, const int k) const {
    assert(abs(i) <= 1);
    assert(abs(j) <= 1);
    assert(abs(k) <= 1);
    return Znei[1 + i][1 + j][1 + k];
  }
};
} // namespace cubism

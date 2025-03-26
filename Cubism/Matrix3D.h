
#pragma once
#include <cassert>

#ifndef CUBISM_ALIGNMENT
#define CUBISM_ALIGNMENT 32
#endif

namespace cubism {

template <class DataType, template <typename T> class allocator>
class Matrix3D {
private:
  DataType *m_pData{nullptr};
  unsigned int m_vSize[3]{0, 0, 0};
  unsigned int m_nElements{0};
  unsigned int m_nElementsPerSlice{0};

public:
  void _Release() {
    if (m_pData != nullptr) {
      free(m_pData);
      m_pData = nullptr;
    }
  }

  void _Setup(unsigned int nSizeX, unsigned int nSizeY, unsigned int nSizeZ) {
    _Release();

    m_vSize[0] = nSizeX;
    m_vSize[1] = nSizeY;
    m_vSize[2] = nSizeZ;

    m_nElementsPerSlice = nSizeX * nSizeY;

    m_nElements = nSizeX * nSizeY * nSizeZ;

    posix_memalign((void **)&m_pData, std::max(8, CUBISM_ALIGNMENT),
                   sizeof(DataType) * m_nElements);
    assert(m_pData != nullptr);
  }

  ~Matrix3D() { _Release(); }

  Matrix3D(unsigned int nSizeX, unsigned int nSizeY, unsigned int nSizeZ)
      : m_pData(nullptr), m_nElements(0), m_nElementsPerSlice(0) {
    _Setup(nSizeX, nSizeY, nSizeZ);
  }

  Matrix3D() : m_pData(nullptr), m_nElements(-1), m_nElementsPerSlice(-1) {}

  Matrix3D(const Matrix3D &m) = delete;

  Matrix3D(Matrix3D &&m)
      : m_pData{m.m_pData}, m_vSize{m.m_vSize[0], m.m_vSize[1], m.m_vSize[2]},
        m_nElements{m.m_nElements}, m_nElementsPerSlice{m.m_nElementsPerSlice} {
    m.m_pData = nullptr;
  }

  inline Matrix3D &operator=(const Matrix3D &m) {
#ifndef NDEBUG
    assert(m_vSize[0] == m.m_vSize[0]);
    assert(m_vSize[1] == m.m_vSize[1]);
    assert(m_vSize[2] == m.m_vSize[2]);
#endif
    for (unsigned int i = 0; i < m_nElements; i++)
      m_pData[i] = m.m_pData[i];
    return *this;
  }

  inline Matrix3D &operator=(DataType d) {
    for (unsigned int i = 0; i < m_nElements; i++)
      m_pData[i] = d;

    return *this;
  }

  inline Matrix3D &operator=(const double a) {
    for (unsigned int i = 0; i < m_nElements; i++)
      m_pData[i].set(a);
    return *this;
  }

  inline DataType &Access(unsigned int ix, unsigned int iy,
                          unsigned int iz) const {
#ifndef NDEBUG
    assert(ix < m_vSize[0]);
    assert(iy < m_vSize[1]);
    assert(iz < m_vSize[2]);
#endif
    return m_pData[iz * m_nElementsPerSlice + iy * m_vSize[0] + ix];
  }

  inline const DataType &Read(unsigned int ix, unsigned int iy,
                              unsigned int iz) const {
#ifndef NDEBUG
    assert(ix < m_vSize[0]);
    assert(iy < m_vSize[1]);
    assert(iz < m_vSize[2]);
#endif
    return m_pData[iz * m_nElementsPerSlice + iy * m_vSize[0] + ix];
  }

  inline DataType &LinAccess(unsigned int i) const {
#ifndef NDEBUG
    assert(i < m_nElements);
#endif
    return m_pData[i];
  }

  inline unsigned int getNumberOfElements() const { return m_nElements; }

  inline unsigned int getNumberOfElementsPerSlice() const {
    return m_nElementsPerSlice;
  }

  inline unsigned int *getSize() const { return (unsigned int *)m_vSize; }

  inline unsigned int getSize(int dim) const { return m_vSize[dim]; }
};

} // namespace cubism

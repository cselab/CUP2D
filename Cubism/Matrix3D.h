/*
 *  Matrix3D.h
 *  Cubism
 *
 *  Created by Diego Rossinelli on 10/19/06.
 *  Copyright 2006 ETH Zurich. All rights reserved.
 *
 */
#pragma once
#include <cassert>

#ifndef CUBISM_ALIGNMENT
#define CUBISM_ALIGNMENT 32
#endif

namespace cubism {

/**
 * A wrapper class for a 3D array of data.
 * @tparam DataType: the kind of data the 3D array is for
 * @tparam allocator: object responsible for allocating the data
 */
template <class DataType, template <typename T> class allocator>
class Matrix3D
{
 private:
   DataType *m_pData{nullptr}; ///< pointer to data
   unsigned int m_vSize[3]{0, 0, 0}; ///< three dimensions (X,Y,Z) (sizes) of array of data 
   unsigned int m_nElements{0}; ///< total number of elements saved (XxYxZ) 
   unsigned int m_nElementsPerSlice{0}; ///< shorthand for XxY

 public:

   /// Deallocate existing data. 
   void _Release()
   {
      if (m_pData != nullptr)
      {
         free(m_pData);
         m_pData = nullptr;
      }
   }

   /// Deallocate existing data and reallocate memory for a nSizeX x nSizeY x nSizeZ array.
   void _Setup(unsigned int nSizeX, unsigned int nSizeY, unsigned int nSizeZ)
   {
      _Release();

      m_vSize[0] = nSizeX;
      m_vSize[1] = nSizeY;
      m_vSize[2] = nSizeZ;

      m_nElementsPerSlice = nSizeX * nSizeY;

      m_nElements = nSizeX * nSizeY * nSizeZ;

      posix_memalign((void **)&m_pData, std::max(8, CUBISM_ALIGNMENT), sizeof(DataType) * m_nElements);
      assert(m_pData != nullptr);
   }

   /// Destructor.
   ~Matrix3D() { _Release(); }

   /// Constructor, calls _Setup()
   Matrix3D(unsigned int nSizeX, unsigned int nSizeY, unsigned int nSizeZ) : m_pData(nullptr), m_nElements(0), m_nElementsPerSlice(0) { _Setup(nSizeX, nSizeY, nSizeZ); }

   /// Constructor, does not allocate memory.
   Matrix3D() : m_pData(nullptr), m_nElements(-1), m_nElementsPerSlice(-1) {}

   Matrix3D(const Matrix3D &m) = delete;

   ///Copy constructor.
   Matrix3D(Matrix3D &&m) : m_pData{m.m_pData}, m_vSize{m.m_vSize[0], m.m_vSize[1], m.m_vSize[2]}, m_nElements{m.m_nElements}, m_nElementsPerSlice{m.m_nElementsPerSlice}
   {
      m.m_pData = nullptr;
   }

   /// Copy another matrix3D to this one
   inline Matrix3D &operator=(const Matrix3D &m)
   {
      #ifndef NDEBUG
         assert(m_vSize[0] == m.m_vSize[0]);
         assert(m_vSize[1] == m.m_vSize[1]);
         assert(m_vSize[2] == m.m_vSize[2]);
      #endif
      for (unsigned int i = 0; i < m_nElements; i++) m_pData[i] = m.m_pData[i];
      return *this;
   }

   /// Set all elements to a given element of the same datatype
   inline Matrix3D &operator=(DataType d)
   {
      for (unsigned int i = 0; i < m_nElements; i++) m_pData[i] = d;

      return *this;
   }

   /// Set all elements to a number, applicable only is data is doubles/floats
   inline Matrix3D &operator=(const double a)
   {
      for (unsigned int i = 0; i < m_nElements; i++) m_pData[i].set(a);
      return *this;
   }

   /// Access an element.
   inline DataType &Access(unsigned int ix, unsigned int iy, unsigned int iz) const
   {
      #ifndef NDEBUG
         assert(ix < m_vSize[0]);
         assert(iy < m_vSize[1]);
         assert(iz < m_vSize[2]);
      #endif
      return m_pData[iz * m_nElementsPerSlice + iy * m_vSize[0] + ix];
   }

   /// Read an element withoud changing it.
   inline const DataType &Read(unsigned int ix, unsigned int iy, unsigned int iz) const
   {
      #ifndef NDEBUG
         assert(ix < m_vSize[0]);
         assert(iy < m_vSize[1]);
         assert(iz < m_vSize[2]);
      #endif
      return m_pData[iz * m_nElementsPerSlice + iy * m_vSize[0] + ix];
   }

   /// Access elements of the array in sequential order, useful for pointwise operations
   inline DataType &LinAccess(unsigned int i) const
   {
      #ifndef NDEBUG
         assert(i < m_nElements);
      #endif
      return m_pData[i];
   }

   /// Get total number of elements of the array
   inline unsigned int getNumberOfElements() const { return m_nElements; }

   /// Get elements on each XY slice/plane of the array
   inline unsigned int getNumberOfElementsPerSlice() const { return m_nElementsPerSlice; }

   /// Get array of sizes for data
   inline unsigned int *getSize() const { return (unsigned int *)m_vSize; }

   /// Get array of size in the 'dim' direction 
   inline unsigned int getSize(int dim) const { return m_vSize[dim]; }
};

}//namespace cubism

#pragma once
#include "Grid.h"
#include "GridMPI.h"
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <hdf5.h>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <vector>
namespace fs = std::filesystem;
template <typename T> hid_t get_hdf5_type();
template <> inline hid_t get_hdf5_type<long long>() { return H5T_NATIVE_LLONG; }
template <> inline hid_t get_hdf5_type<short int>() { return H5T_NATIVE_SHORT; }
template <> inline hid_t get_hdf5_type<int>() { return H5T_NATIVE_INT; }
template <> inline hid_t get_hdf5_type<float>() { return H5T_NATIVE_FLOAT; }
template <> inline hid_t get_hdf5_type<double>() { return H5T_NATIVE_DOUBLE; }
template <> inline hid_t get_hdf5_type<long double>() {
  return H5T_NATIVE_LDOUBLE;
}
#include "BlockInfo.h"
namespace cubism {
struct StreamerScalar {
  static constexpr int NCHANNELS = 1;
  template <typename TBlock, typename T>
  static inline void operate(TBlock &b, const int ix, const int iy,
                             const int iz, T output[NCHANNELS]) {
    output[0] = b(ix, iy, iz).s;
  }
  static std::string prefix() { return std::string(""); }
  static const char *getAttributeName() { return "Scalar"; }
};
struct StreamerVector {
  static constexpr int NCHANNELS = 3;
  template <typename TBlock, typename T>
  static void operate(TBlock &b, const int ix, const int iy, const int iz,
                      T output[NCHANNELS]) {
    for (int i = 0; i < TBlock::ElementType::DIM; i++)
      output[i] = b(ix, iy, iz).u[i];
  }
  static std::string prefix() { return std::string(""); }
  static const char *getAttributeName() { return "Vector"; }
};
template <typename data_type>
void save_buffer_to_file(const std::vector<data_type> &buffer,
                         const int NCHANNELS, MPI_Comm &comm,
                         const std::string &name,
                         const std::string &dataset_name, const hid_t &file_id,
                         const hid_t &fapl_id) {
  assert(buffer.size() % NCHANNELS == 0);
  unsigned long long MyCells = buffer.size() / NCHANNELS;
  unsigned long long TotalCells;
  MPI_Allreduce(&MyCells, &TotalCells, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                comm);
  hsize_t base_tmp[1] = {0};
  MPI_Exscan(&MyCells, &base_tmp[0], 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
  base_tmp[0] *= NCHANNELS;
  hid_t dataset_id, fspace_id, mspace_id;
  hsize_t dims[1] = {(hsize_t)TotalCells * NCHANNELS};
  fspace_id = H5Screate_simple(1, dims, NULL);
  dataset_id =
      H5Dcreate(file_id, dataset_name.c_str(), get_hdf5_type<data_type>(),
                fspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hsize_t count[1] = {MyCells * NCHANNELS};
  fspace_id = H5Dget_space(dataset_id);
  mspace_id = H5Screate_simple(1, count, NULL);
  H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, base_tmp, NULL, count, NULL);
  H5Dwrite(dataset_id, get_hdf5_type<data_type>(), mspace_id, fspace_id,
           fapl_id, buffer.data());
  H5Sclose(mspace_id);
  H5Sclose(fspace_id);
  H5Dclose(dataset_id);
#if 0
    hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t cdims[1];
    cdims[0] = 8*8*8;
    if (compression==false)
    {
        const int PtsPerElement = 8;
        cdims[0] *= PtsPerElement * DIMENSION;
    }
    H5Pset_chunk(plist_id, 1, cdims);
    H5Pset_deflate(plist_id, 6);
    dataset_id = H5Dcreate(file_id, dataset_name.c_str(), get_hdf5_type<data_type>(), fspace_id, H5P_DEFAULT, plist_id, H5P_DEFAULT);
    hsize_t count[1] = {MyCells*NCHANNELS};
    fspace_id = H5Dget_space(dataset_id);
    mspace_id = H5Screate_simple(1, count, NULL);
    H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, base_tmp, NULL, count, NULL);
    H5Dwrite(dataset_id, get_hdf5_type<data_type>(), mspace_id, fspace_id, fapl_id, buffer.data());
    H5Sclose(mspace_id);
    H5Sclose(fspace_id);
    H5Dclose(dataset_id);
    H5Pclose(plist_id);
#endif
}
static double latestTime{-1.0};
template <typename TStreamer, typename hdf5Real, typename TGrid>
void DumpHDF5_MPI(TGrid &grid, typename TGrid::Real absTime,
                  const std::string &fname, const std::string &dpath = ".",
                  const bool dumpGrid = true) {
  const bool SaveGrid = latestTime < absTime && dumpGrid;
  int gridCount = 0;
  for (auto &p : fs::recursive_directory_iterator(dpath)) {
    if (p.path().extension() == ".h5") {
      std::string g = p.path().stem().string();
      g.resize(4);
      if (g == "grid") {
        gridCount++;
      }
    }
  }
  if (SaveGrid == false)
    gridCount--;
  latestTime = absTime;
  typedef typename TGrid::BlockType B;
  const int nX = B::sizeX;
  const int nY = B::sizeY;
  const int nZ = B::sizeZ;
  const int NCHANNELS = TStreamer::NCHANNELS;
  MPI_Comm comm = grid.getWorldComm();
  const int rank = grid.myrank;
  std::ostringstream filename;
  std::ostringstream fullpath;
  filename << fname;
  fullpath << dpath << "/" << filename.str();
  const int PtsPerElement = 4;
  std::vector<BlockInfo> &MyInfos = grid.getBlocksInfo();
  unsigned long long MyCells = MyInfos.size() * nX * nY * nZ;
  unsigned long long TotalCells;
  MPI_Allreduce(&MyCells, &TotalCells, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                comm);
  H5open();
  hid_t file_id, fapl_id;
  fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(fapl_id, comm, MPI_INFO_NULL);
  file_id = H5Fcreate((fullpath.str() + ".h5").c_str(), H5F_ACC_TRUNC,
                      H5P_DEFAULT, fapl_id);
  H5Pclose(fapl_id);
  H5Fclose(file_id);
  hid_t file_id_grid, fapl_id_grid;
  std::stringstream gridFile_s;
  gridFile_s << "grid" << std::setfill('0') << std::setw(9) << gridCount
             << ".h5";
  std::string gridFile = gridFile_s.str();
  std::stringstream gridFilePath_s;
  gridFilePath_s << dpath << "/grid" << std::setfill('0') << std::setw(9)
                 << gridCount << ".h5";
  std::string gridFilePath = gridFilePath_s.str();
  if (SaveGrid) {
    fapl_id_grid = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id_grid, comm, MPI_INFO_NULL);
    file_id_grid = H5Fcreate(gridFilePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                             fapl_id_grid);
    H5Pclose(fapl_id_grid);
    H5Fclose(file_id_grid);
  }
  if (rank == 0 && dumpGrid) {
    std::ostringstream myfilename;
    myfilename << filename.str();
    std::stringstream s;
    s << "<?xml version=\"1.0\" ?>\n";
    s << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    s << "<Xdmf Version=\"2.0\">\n";
    s << "<Domain>\n";
    s << " <Grid Name=\"OctTree\" GridType=\"Uniform\">\n";
    s << "  <Time Value=\"" << std::scientific << absTime << "\"/>\n\n";
    s << "   <Topology NumberOfElements=\"" << TotalCells
      << "\" TopologyType=\"Quadrilateral\"/>\n";
    s << "     <Geometry GeometryType=\"XY\">\n";
    s << "        <DataItem ItemType=\"Uniform\"  Dimensions=\" "
      << TotalCells * PtsPerElement << " " << DIMENSION
      << "\" NumberType=\"Float\" Precision=\" " << (int)sizeof(float)
      << "\" Format=\"HDF\">\n";
    s << "            " << gridFile.c_str() << ":/" << "vertices" << "\n";
    s << "        </DataItem>\n";
    s << "     </Geometry>\n";
    s << "     <Attribute Name=\"data\" AttributeType=\""
      << TStreamer::getAttributeName() << "\" Center=\"Cell\">\n";
    s << "        <DataItem ItemType=\"Uniform\"  Dimensions=\" " << TotalCells
      << " " << NCHANNELS << "\" NumberType=\"Float\" Precision=\" "
      << (int)sizeof(hdf5Real) << "\" Format=\"HDF\">\n";
    s << "            " << (myfilename.str() + ".h5").c_str() << ":/" << "data"
      << "\n";
    s << "        </DataItem>\n";
    s << "     </Attribute>\n";
    s << " </Grid>\n";
    s << "</Domain>\n";
    s << "</Xdmf>\n";
    std::string st = s.str();
    FILE *xmf = 0;
    xmf = fopen((fullpath.str() + "-new.xmf").c_str(), "w");
    fprintf(xmf, "%s", st.c_str());
    fclose(xmf);
  }
  std::string name = fullpath.str() + ".h5";
  fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(fapl_id, comm, MPI_INFO_NULL);
  file_id = H5Fopen(name.c_str(), H5F_ACC_RDWR, fapl_id);
  H5Pclose(fapl_id);
  fapl_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(fapl_id, H5FD_MPIO_COLLECTIVE);
  {
    std::vector<short int> bufferlevel(MyInfos.size());
    std::vector<long long> bufferZ(MyInfos.size());
    for (size_t i = 0; i < MyInfos.size(); i++) {
      bufferlevel[i] = MyInfos[i].level;
      bufferZ[i] = MyInfos[i].Z;
    }
    save_buffer_to_file<short int>(bufferlevel, 1, comm, fullpath.str() + ".h5",
                                   "blockslevel", file_id, fapl_id);
    save_buffer_to_file<long long>(bufferZ, 1, comm, fullpath.str() + ".h5",
                                   "blocksZ", file_id, fapl_id);
  }
  if (SaveGrid) {
    fapl_id_grid = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id_grid, comm, MPI_INFO_NULL);
    file_id_grid = H5Fopen(gridFilePath.c_str(), H5F_ACC_RDWR, fapl_id_grid);
    H5Pclose(fapl_id_grid);
    fapl_id_grid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(fapl_id_grid, H5FD_MPIO_COLLECTIVE);
    std::vector<float> buffer(MyCells * PtsPerElement * DIMENSION);
    for (size_t i = 0; i < MyInfos.size(); i++) {
      const BlockInfo &info = MyInfos[i];
      const float h2 = 0.5 * info.h;
      for (int z = 0; z < nZ; z++)
        for (int y = 0; y < nY; y++)
          for (int x = 0; x < nX; x++) {
            const int bbase = (i * nZ * nY * nX + z * nY * nX + y * nX + x) *
                              PtsPerElement * DIMENSION;
            double p[2];
            info.pos(p, x, y);
            buffer[bbase] = p[0] - h2;
            buffer[bbase + 1] = p[1] - h2;
            buffer[bbase + DIMENSION] = p[0] - h2;
            buffer[bbase + DIMENSION + 1] = p[1] + h2;
            buffer[bbase + 2 * DIMENSION] = p[0] + h2;
            buffer[bbase + 2 * DIMENSION + 1] = p[1] + h2;
            buffer[bbase + 3 * DIMENSION] = p[0] + h2;
            buffer[bbase + 3 * DIMENSION + 1] = p[1] - h2;
          }
    }
    save_buffer_to_file<float>(buffer, 1, comm, gridFilePath, "vertices",
                               file_id_grid, fapl_id_grid);
    H5Pclose(fapl_id_grid);
    H5Fclose(file_id_grid);
  }
  {
    std::vector<hdf5Real> buffer(MyCells * NCHANNELS);
    for (size_t i = 0; i < MyInfos.size(); i++) {
      const BlockInfo &info = MyInfos[i];
      B &b = *(B *)info.ptrBlock;
      for (int z = 0; z < nZ; z++)
        for (int y = 0; y < nY; y++)
          for (int x = 0; x < nX; x++) {
            hdf5Real output[NCHANNELS]{0};
            TStreamer::operate(b, x, y, z, output);
            for (int nc = 0; nc < NCHANNELS; nc++) {
              buffer[(i * nZ * nY * nX + z * nY * nX + y * nX + x) * NCHANNELS +
                     nc] = output[nc];
            }
          }
    }
    save_buffer_to_file<hdf5Real>(buffer, NCHANNELS, comm,
                                  fullpath.str() + ".h5", "data", file_id,
                                  fapl_id);
  }
  H5Pclose(fapl_id);
  H5Fclose(file_id);
  H5close();
}
} // namespace cubism

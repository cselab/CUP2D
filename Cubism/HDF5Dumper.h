#pragma once
#include "BlockInfo.h"
#include "Grid.h"
#include "GridMPI.h"
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <vector>
namespace cubism {
struct StreamerScalar {
  static constexpr int NCHANNELS = 1;
  template <typename TBlock, typename T>
  static inline void operate(TBlock &b, const int ix, const int iy,
                             const int iz, T output[NCHANNELS]) {
    output[0] = b(ix, iy, iz).s;
  }
};
struct StreamerVector {
  static constexpr int NCHANNELS = 3;
  template <typename TBlock, typename T>
  static void operate(TBlock &b, const int ix, const int iy, const int iz,
                      T output[NCHANNELS]) {
    for (int i = 0; i < TBlock::ElementType::DIM; i++)
      output[i] = b(ix, iy, iz).u[i];
  }
};
template <typename TStreamer, typename hdf5Real, typename TGrid>
void Dump(TGrid &grid, typename TGrid::Real absTime, const char *fname,
          const std::string &dpath = ".", const bool dumpGrid = true) {
  static double latestTime{-1.0};
  static long gridCount = 0;
  MPI_File mpi_file;
  char xyz_path[FILENAME_MAX], attr_path[FILENAME_MAX], xdmf_path[FILENAME_MAX];
  const bool SaveGrid = latestTime < absTime && dumpGrid;
  if (SaveGrid) {
    gridCount++;
  }
  latestTime = absTime;
  snprintf(xyz_path, sizeof xyz_path, "xyz.%09ld.raw", gridCount);
  snprintf(xdmf_path, sizeof xdmf_path, "%s.%09ld.xdmf2", fname, gridCount);
  typedef typename TGrid::BlockType B;
  const int nX = B::sizeX;
  const int nY = B::sizeY;
  const int nZ = B::sizeZ;
  const int NCHANNELS = TStreamer::NCHANNELS;
  int rank, size, i;
  MPI_Comm comm = grid.getWorldComm();
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  std::vector<BlockInfo> &MyInfos = grid.getBlocksInfo();
  unsigned long long ncell = MyInfos.size() * nX * nY * nZ;
  unsigned long long offset, ncell_total;
  MPI_Exscan(&ncell, &offset, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
  if (rank == 0)
    offset = 0;
  ncell_total = ncell + offset;
  if (rank == size - 1 && dumpGrid) {
    FILE *xmf = fopen(xdmf_path, "w");
    if (!xmf) {
      fprintf(stderr, "%s:%d: Failed to open .xdmf2 file", __FILE__, __LINE__);
      MPI_Abort(comm, 1);
    }

    struct {
      const char *name;
      int nc;
    } attrs[] = {
        {"chi", 1},
        {"pres", 1},
        {"tmp", 1},
        {"vel", 2},
    };

    fprintf(xmf,
            "<Xdmf\n"
            "    Version=\"2.0\">\n"
            "  <Domain>\n"
            "    <Grid>\n"
            "      <Time Value=\"%.16e\"/>\n"
            "      <Topology\n"
            "          Dimensions=\"%lld\"\n"
            "          TopologyType=\"Quadrilateral\"/>\n"
            "      <Geometry\n"
            "          GeometryType=\"XY\">\n"
            "        <DataItem\n"
            "            Dimensions=\"%lld 2\"\n"
            "            Format=\"Binary\">\n"
            "          %s\n"
            "        </DataItem>\n"
            "      </Geometry>\n",
            absTime, ncell_total, 4 * ncell_total, xyz_path);
    for (i = 0; i < sizeof attrs / sizeof *attrs; i++) {
      snprintf(attr_path, sizeof attr_path, "%s.%09ld.raw", attrs[i].name,
               gridCount);
      fprintf(xmf,
              "      <Attribute\n"
              "          Name=\"%s\"\n"
              "          AttributeType=\"%s\"\n"
              "          Center=\"Cell\">\n"
              "        <DataItem\n"
              "            Dimensions=\"%lld %d\"\n"
              "            Precision=\"%d\"\n"
              "            Format=\"Binary\">\n"
              "          %s\n"
              "        </DataItem>\n"
              "      </Attribute>\n",
              attrs[i].name, attrs[i].nc == 1 ? "Scalar" : "Vector",
              ncell_total, attrs[i].nc, (int)sizeof(hdf5Real), attr_path);
    }
    fprintf(xmf, "    </Grid>\n"
                 "  </Domain>\n"
                 "</Xdmf>\n");
    fclose(xmf);
  }
  if (SaveGrid) {
    std::vector<float> buffer(ncell * 4 * 2);
    for (size_t i = 0; i < MyInfos.size(); i++) {
      const BlockInfo &info = MyInfos[i];
      const float h2 = 0.5 * info.h;
      for (int z = 0; z < nZ; z++)
        for (int y = 0; y < nY; y++)
          for (int x = 0; x < nX; x++) {
            const int bbase =
                (i * nZ * nY * nX + z * nY * nX + y * nX + x) * 4 * 2;
            double p[2];
            info.pos(p, x, y);
            buffer[bbase] = p[0] - h2;
            buffer[bbase + 1] = p[1] - h2;
            buffer[bbase + 2] = p[0] - h2;
            buffer[bbase + 3] = p[1] + h2;
            buffer[bbase + 4] = p[0] + h2;
            buffer[bbase + 5] = p[1] + h2;
            buffer[bbase + 6] = p[0] + h2;
            buffer[bbase + 7] = p[1] - h2;
          }
    }
    MPI_File_open(comm, xyz_path, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &mpi_file);
    MPI_File_write_at_all(mpi_file, 8 * offset * sizeof buffer[0],
                          buffer.data(), 8 * ncell * sizeof buffer[0],
                          MPI_UINT8_T, MPI_STATUS_IGNORE);
    MPI_File_close(&mpi_file);
  }
  {
    std::vector<hdf5Real> buffer(ncell * NCHANNELS);
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
    snprintf(attr_path, sizeof attr_path, "%s.%09ld.raw", fname, gridCount);
    MPI_File_open(comm, attr_path, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &mpi_file);
    MPI_File_write_at_all(mpi_file, NCHANNELS * offset * sizeof buffer[0],
                          buffer.data(), NCHANNELS * ncell * sizeof buffer[0],
                          MPI_UINT8_T, MPI_STATUS_IGNORE);
    MPI_File_close(&mpi_file);
  }
}
} // namespace cubism

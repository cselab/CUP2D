#include <iostream>
#include <vector>
#include <hdf5.h>
#include <mpi.h>
#include <string>
#include <sstream>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <algorithm>    // std::sort
#include <variant>
namespace fs = std::filesystem;

void convert_to_float(std::string filename, std::string gridname)
{
  const int dimension = 2;
  const int ptsPerElement = 4;
  const int nx = 8;
  const int ny = 8;
  //const int C = 2;
  size_t blocks = 0;

  H5open();

  hid_t file_id = H5Fopen((filename+".h5").c_str(), H5F_ACC_RDWR, H5P_DEFAULT);


  std::vector<short int>levels;
  {
    hid_t dataset_id, fspace_id;
    hsize_t dim;
    dataset_id = H5Dopen2(file_id, "blockslevel", H5P_DEFAULT);
    fspace_id = H5Dget_space(dataset_id);
    H5Sget_simple_extent_dims(fspace_id, &dim, NULL);
    hid_t dtype =  H5Dget_type(dataset_id);

    levels.resize(dim);

    const bool isInt = H5Tequal(dtype, H5T_NATIVE_INT);
    if (isInt)
    {
      std::vector<int> levels_int(dim);
      H5Dread(dataset_id, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, levels_int.data());
      for (size_t i = 0 ; i < dim ; i++)
        levels[i] = (short int) levels_int[i];
    }
    else
    {
      H5Dread(dataset_id, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, levels.data());
    }
    H5Dclose(dataset_id);
    H5Sclose(fspace_id);
  }

  //read data
  std::vector<double> amr;
  {
    hid_t dataset_id, fspace_id;
    hsize_t dim;
    dataset_id = H5Dopen2(file_id, "data", H5P_DEFAULT);
    fspace_id = H5Dget_space(dataset_id);
    H5Sget_simple_extent_dims(fspace_id, &dim, NULL);
    amr.resize(dim);
    H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, amr.data());
    H5Dclose(dataset_id);
    H5Sclose(fspace_id);
    blocks = dim / nx / ny;
  }

  hid_t file_id_grid = H5Fopen((gridname+".h5").c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  std::vector<float>vertices;
  {
    hid_t dataset_id, fspace_id;
    hsize_t dim;
    dataset_id = H5Dopen2(file_id_grid, "vertices", H5P_DEFAULT);
    fspace_id = H5Dget_space(dataset_id);
    H5Sget_simple_extent_dims(fspace_id, &dim, NULL);
    hid_t dtype =  H5Dget_type(dataset_id);

    vertices.resize(dim);

    const bool isDouble = H5Tequal(dtype, H5T_NATIVE_DOUBLE);
    if (isDouble)
    {
      std::vector<double> vertices_double(dim);
      H5Dread(dataset_id, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, vertices_double.data());
      for (size_t i = 0 ; i < dim ; i++)
        vertices[i] = (float) vertices_double[i];
    }
    else
    {
      H5Dread(dataset_id, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, vertices.data());
    }
    H5Dclose(dataset_id);
    H5Sclose(fspace_id);
  }
  H5Fclose(file_id);
  H5Fclose(file_id_grid);

  const int NCHANNELS =  (vertices.size() == amr.size()*ptsPerElement*dimension) ? 1 : 3;
  blocks /= NCHANNELS;

  std::vector<float> data_c    ;
  std::vector<float> vertices_c;
  data_c.reserve(amr.size()/4);
  vertices_c.reserve(vertices.size()/4);
  for (size_t i = 0 ; i < blocks ; i++)
  {
    int C=1;
    //if (levels[i] == 10) C = 4;
    for (int y = 0 ; y < ny ; y+=C)
    for (int x = 0 ; x < nx ; x+=C)
    {
      float element [NCHANNELS] = {0.0};
      float magnitude = 0;
      for (int j = 0 ; j < NCHANNELS; j++)
      {
        for (int yl = y; yl < y+C; yl++)
        for (int xl = x; xl < x+C; xl++)
        {
          element[j] += (float)amr[(i*nx*ny+yl*nx+xl)*NCHANNELS+j];
        }
        element[j] /= (C*C);
        magnitude += element[j]*element[j];
      }

      if (magnitude < 1e-2) continue;

      for (int j = 0 ; j < NCHANNELS; j++)
        data_c.push_back(element[j]);

      const int bbase00 = (i*ny*nx+ y     *nx+x    )*ptsPerElement*dimension;
      const int bbase10 = (i*ny*nx+ y     *nx+x+C-1)*ptsPerElement*dimension;
      const int bbase01 = (i*ny*nx+(y+C-1)*nx+x    )*ptsPerElement*dimension;
      const int bbase11 = (i*ny*nx+(y+C-1)*nx+x+C-1)*ptsPerElement*dimension;
      const int offset00 = 0;
      const int offset10 = 3*dimension;
      const int offset11 = 2*dimension;
      const int offset01 =  dimension;
      const float xm00 = vertices[bbase00+offset00  ];
      const float ym00 = vertices[bbase00+offset00+1];
      const float xm10 = vertices[bbase10+offset10  ];
      const float ym10 = vertices[bbase10+offset10+1];
      const float xm01 = vertices[bbase01+offset01  ];
      const float ym01 = vertices[bbase01+offset01+1];
      const float xm11 = vertices[bbase11+offset11  ];
      const float ym11 = vertices[bbase11+offset11+1];
      vertices_c.push_back(xm00);
      vertices_c.push_back(ym00);
      vertices_c.push_back(xm10);
      vertices_c.push_back(ym10);
      vertices_c.push_back(xm11);
      vertices_c.push_back(ym11);
      vertices_c.push_back(xm01);
      vertices_c.push_back(ym01);  
    }
  }

  std::vector<float> vertices_grid(vertices.size()/nx/ny,0.0);
  for (size_t i = 0 ; i < blocks ; i++)
  {
    for (int y = 0 ; y < ny ; y+=ny)
    for (int x = 0 ; x < nx ; x+=nx)
    {
      const int bbase00 = (i*ny*nx+ y      *nx+x     )*ptsPerElement*dimension;
      const int bbase10 = (i*ny*nx+ y      *nx+x+nx-1)*ptsPerElement*dimension;
      const int bbase01 = (i*ny*nx+(y+ny-1)*nx+x     )*ptsPerElement*dimension;
      const int bbase11 = (i*ny*nx+(y+ny-1)*nx+x+nx-1)*ptsPerElement*dimension;

      const int offset00 = 0;
      const int offset10 = 3*dimension;
      const int offset11 = 2*dimension;
      const int offset01 =  dimension;

      const float xm00 = vertices[bbase00+offset00  ];
      const float ym00 = vertices[bbase00+offset00+1];
      const float xm10 = vertices[bbase10+offset10  ];
      const float ym10 = vertices[bbase10+offset10+1];
      const float xm01 = vertices[bbase01+offset01  ];
      const float ym01 = vertices[bbase01+offset01+1];
      const float xm11 = vertices[bbase11+offset11  ];
      const float ym11 = vertices[bbase11+offset11+1];
      const int bbasef = i*ptsPerElement*dimension;
      vertices_grid[bbasef              ]=xm00;
      vertices_grid[bbasef            +1]=ym00;
      vertices_grid[bbasef+  dimension  ]=xm10;
      vertices_grid[bbasef+  dimension+1]=ym10;
      vertices_grid[bbasef+2*dimension  ]=xm11;
      vertices_grid[bbasef+2*dimension+1]=ym11;
      vertices_grid[bbasef+3*dimension  ]=xm01;
      vertices_grid[bbasef+3*dimension+1]=ym01;  
    }
  }

  hid_t file_id1 = H5Fcreate((filename+"-compressed.h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  {
    hsize_t dims[1] = { (hsize_t)data_c.size() };
    hid_t fspace_id  = H5Screate_simple(1, dims, NULL);
    hid_t dataset_id = H5Dcreate (file_id1,"data-c",H5T_NATIVE_FLOAT,fspace_id,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    H5Sclose(fspace_id);
    fspace_id = H5Dget_space(dataset_id);
    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT,H5P_DEFAULT,fspace_id,H5P_DEFAULT,data_c.data());
    H5Sclose(fspace_id);
    H5Dclose(dataset_id);
  }
  {
    hsize_t dims[1] = { (hsize_t)vertices_c.size() };
    hid_t fspace_id  = H5Screate_simple(1, dims, NULL);
    hid_t dataset_id = H5Dcreate (file_id1,"vertices-c",H5T_NATIVE_FLOAT,fspace_id,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    H5Sclose(fspace_id);
    fspace_id = H5Dget_space(dataset_id);
    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT,H5P_DEFAULT,fspace_id,H5P_DEFAULT,vertices_c.data());
    H5Sclose(fspace_id);
    H5Dclose(dataset_id);
  }
  {
    hsize_t dims[1] = { (hsize_t)vertices_grid.size() };
    hid_t fspace_id  = H5Screate_simple(1, dims, NULL);
    hid_t dataset_id = H5Dcreate (file_id1,"vertices-grid",H5T_NATIVE_FLOAT,fspace_id,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    H5Sclose(fspace_id);
    fspace_id = H5Dget_space(dataset_id);
    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT,H5P_DEFAULT,fspace_id,H5P_DEFAULT,vertices_grid.data());
    H5Sclose(fspace_id);
    H5Dclose(dataset_id);
  }

  H5Fclose(file_id1);

  H5close();

  {
    const long long TotalCells = data_c.size()/NCHANNELS;
    std::ostringstream myfilename;
    myfilename << filename;
    std::stringstream s;
    s << "<?xml version=\"1.0\" ?>\n";
    s << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    s << "<Xdmf Version=\"2.0\">\n";
    s << "<Domain>\n";
    s << " <Grid Name=\"OctTree\" GridType=\"Uniform\">\n";
    s << "   <Topology NumberOfElements=\"" << TotalCells << "\" TopologyType=\"Quadrilateral\"/>\n";
    s << "     <Geometry GeometryType=\"XY\">\n";
    s << "        <DataItem ItemType=\"Uniform\"  Dimensions=\" " << TotalCells*ptsPerElement << " " << dimension << "\" NumberType=\"Float\" Precision=\" " << (int)sizeof(float) << "\" Format=\"HDF\">\n";
    s << "            " << (myfilename.str() + "-compressed.h5").c_str() << ":/" << "vertices-c" << "\n";
    s << "        </DataItem>\n";
    s << "     </Geometry>\n";
    if (NCHANNELS == 1)
    s << "     <Attribute Name=\"data\" AttributeType=\"" << "Scalar" << "\" Center=\"Cell\">\n";
    else
    s << "     <Attribute Name=\"data\" AttributeType=\"" << "Vector" << "\" Center=\"Cell\">\n";
    s << "        <DataItem ItemType=\"Uniform\"  Dimensions=\" " << TotalCells << " " << NCHANNELS << "\" NumberType=\"Float\" Precision=\" " << (int)sizeof(float) << "\" Format=\"HDF\">\n";
    s << "            " << (myfilename.str() + "-compressed.h5").c_str() << ":/" << "data-c" << "\n";
    s << "        </DataItem>\n";
    s << "     </Attribute>\n";
    s << " </Grid>\n";
    s << "</Domain>\n";
    s << "</Xdmf>\n";
    std::string st = s.str();
    FILE *xmf = 0;
    xmf = fopen((myfilename.str() + "-compressed.xmf").c_str(), "w");
    fprintf(xmf, st.c_str());
    fclose(xmf);
  }

  {
    const long long TotalCells = blocks;
    std::ostringstream myfilename;
    myfilename << filename;
    std::stringstream s;
    s << "<?xml version=\"1.0\" ?>\n";
    s << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    s << "<Xdmf Version=\"2.0\">\n";
    s << "<Domain>\n";
    s << " <Grid Name=\"OctTree\" GridType=\"Uniform\">\n";
    s << "   <Topology NumberOfElements=\"" << TotalCells << "\" TopologyType=\"Quadrilateral\"/>\n";
    s << "     <Geometry GeometryType=\"XY\">\n";
    s << "        <DataItem ItemType=\"Uniform\"  Dimensions=\" " << TotalCells*ptsPerElement << " " << dimension << "\" NumberType=\"Float\" Precision=\" " << (int)sizeof(float) << "\" Format=\"HDF\">\n";
    s << "            " << (myfilename.str() + "-compressed.h5").c_str() << ":/" << "vertices-grid" << "\n";
    s << "        </DataItem>\n";
    s << "     </Geometry>\n";
    s << " </Grid>\n";
    s << "</Domain>\n";
    s << "</Xdmf>\n";
    std::string st = s.str();
    FILE *xmf = 0;
    xmf = fopen((myfilename.str() + "-grid.xmf").c_str(), "w");
    fprintf(xmf, st.c_str());
    fclose(xmf);
  }
} 

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank,size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  std::vector<std::string> filenames;
  std::vector<std::string> gridnames;
  std::string path("./");
  std::string ext(".h5");
  for (auto &p : fs::recursive_directory_iterator(path))
  {
    if (p.path().extension() == ext)
    {
      std::string s = p.path().stem().string();
      std::string g = p.path().stem().string();
      g.resize(4);
      if ( s.back() != 's' && s.back() != 'm' && g != "grid")
      {
        filenames.push_back(p.path().stem().string());
        //uncomment for old format:
        //gridnames.push_back(p.path().stem().string());
      }
      
      if ( g  == "grid" )
      {
        gridnames.push_back(p.path().stem().string());
      }
    }
  }
  std::sort(filenames.begin(),filenames.end());
  std::sort(gridnames.begin(),gridnames.end());
  MPI_Barrier(MPI_COMM_WORLD);
  if (filenames.size() % gridnames.size() != 0)
  {
    std::cerr << "Number of files and grids are not compatible. " << std::endl;
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  const size_t fields = filenames.size() / gridnames.size();
  for (size_t f = 0; f < fields ; f++)
  {
     MPI_Barrier(MPI_COMM_WORLD);
     for (size_t g = 0 ; g < gridnames.size(); g+= size)
     {
       const size_t i = f * gridnames.size() + g;
       if (i+rank >= filenames.size()) continue;
       if (g+rank >= gridnames.size()) continue;
       std::cout << "converting " << filenames[i+rank]<< ":"<< i+rank << " /" << filenames.size() << std::endl;
       convert_to_float(filenames[i+rank],gridnames[g+rank]);
     }
  }
  MPI_Finalize();
  return 0;
}

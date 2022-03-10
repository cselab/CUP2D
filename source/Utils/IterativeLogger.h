#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#include "mpi.h"
#include "cuda_runtime.h"

#include "../include/json.hpp" // https://github.com/nlohmann/json
#include "../include/helper_cuda.h"


class BiCGSTABLoggerCPU {
  public: 
    BiCGSTABLoggerCPU(MPI_Comm m_comm, int max_steps) : step_(0), max_steps_(max_steps) 
  {
    MPI_Comm_rank(m_comm, &rank_);
  }

    // Method to be called at the beginning of each time-step to save metadata 
    void new_step(int m)
    {
      j_["metadata"][std::to_string(step_)]["m"] = m;
    }

    // Method to log BiCGSTAB coefficients
    void log_coeffs(
        const int iter,
        const double alpha,
        const double rho,
        const double beta,
        const double omega)
    {
      j_["bicgstab"][std::to_string(step_)][std::to_string(iter)]["alpha"] = alpha;
      j_["bicgstab"][std::to_string(step_)][std::to_string(iter)]["rho"] = rho;
      j_["bicgstab"][std::to_string(step_)][std::to_string(iter)]["beta"] = beta;
      j_["bicgstab"][std::to_string(step_)][std::to_string(iter)]["omega"] = omega;
    }

    // Method to log a BiCGSTAB vector
    template<class val_t>
    void log_vec(
        int iter,
        std::string tag,
        std::vector<val_t> &vec)
    {
      j_["bicgstab"][std::to_string(step_)][std::to_string(iter)][tag] = vec;
    }

    void print_coeffs(const int iter)
    {
        
      if (rank_ == 0)
      {
        double alpha = j_["bicgstab"][std::to_string(step_)][std::to_string(iter)]["alpha"]; 
        double rho = j_["bicgstab"][std::to_string(step_)][std::to_string(iter)]["rho"];
        double beta = j_["bicgstab"][std::to_string(step_)][std::to_string(iter)]["beta"];
        double omega = j_["bicgstab"][std::to_string(step_)][std::to_string(iter)]["omega"];

        std::cerr << "  [BiCGSTAB Logger rank " << rank_ << "]: Iteration: " << iter 
                  << ", alpha: " << alpha << ", rho: " << rho 
                  << ", beta: " << beta << ", omega: " << omega << std::endl;
      }
    }

    void print_vec(
        const std::string tag, 
        const int len, 
        const int offset, 
        const std::vector<double> &vec)
    {
      std::cerr << tag << ": [";
      for (int i(offset); i < len + offset; i++)
        std::cerr << vec[i] << ", ";
      std::cerr << std::endl;
    }

    void dump(std::string base)
    { 
      step_ += 1;
      if (step_ >= max_steps_)
      {
        std::cerr << "  [BiCGSTAB Logger rank " << rank_ << "]: Writing log to file...\n";
        std::string filename = base + "rank_" + std::to_string(rank_) + ".json";

        std::ofstream file;
        file.open(filename, std::ios::trunc);
        file << j_;
        file.close();
        
        throw std::runtime_error("[BiCGSTAB Logger]: a single dump is all this logger lives for.");
      }
    }

  protected:
    int rank_;
    int step_;
    int max_steps_;
    nlohmann::json j_;
};

class BiCGSTABLoggerGPU : public BiCGSTABLoggerCPU {
  public:
    BiCGSTABLoggerGPU(MPI_Comm m_comm, int max_steps) : BiCGSTABLoggerCPU(m_comm, max_steps) {}

    // Method to be called at the beginning of each time-step to save metadata 
    void new_step(int m, int halo, int loc_nnz, int bd_nnz)
    {
      j_["metadata"][std::to_string(step_)]["m"] = m;
      j_["metadata"][std::to_string(step_)]["halo"] = halo;
      j_["metadata"][std::to_string(step_)]["loc_nnz"] = loc_nnz;
      j_["metadata"][std::to_string(step_)]["bd_nnz"] = bd_nnz;
    }

    // Redefine here because templating prevents method from being virtual
    template<class val_t>
    void log_vec(
        int iter,
        std::string tag,
        std::vector<val_t> &vec)
    {
      j_["bicgstab"][std::to_string(step_)][std::to_string(iter)][tag] = vec;
    }

    // Method to log a BiCGSTAB vector
    void log_vec(
        cudaStream_t solver_stream,
        const int iter,
        const std::string tag,
        const int m, 
        const double* const d_vec)
    {
      std::vector<double> vec(m);
      checkCudaErrors(cudaMemcpyAsync(vec.data(), d_vec, m * sizeof(double), cudaMemcpyDeviceToHost, solver_stream));
      checkCudaErrors(cudaStreamSynchronize(solver_stream));
      j_["bicgstab"][std::to_string(step_)][std::to_string(iter)][tag] = vec;
    }

    void print_vec(
        const std::string tag, 
        const int len,
        const int offset,
        cudaStream_t solver_stream,
        const double* const d_vec)
    {
      std::vector<double> vec(len);
      checkCudaErrors(cudaMemcpyAsync(vec.data(), &d_vec[offset], len * sizeof(double), cudaMemcpyDeviceToHost, solver_stream));
      checkCudaErrors(cudaStreamSynchronize(solver_stream));
      BiCGSTABLoggerCPU::print_vec(tag, len, 0, vec);
    }
};

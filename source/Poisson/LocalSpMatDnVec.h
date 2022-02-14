#pragma once

#include <map>
#include <vector>
#include <set>
#include <mpi.h>

class SpRowInfo
{
  public:
    const int rank_;
    const long long idx_; // global row index
    std::map<long long, double> loc_colval_; // col_idx->val map
    std::map<long long, double> bd_colval_;
    // neirank_cols_[i] holds {rank of non-local col, idx of non-local col}
    std::vector<std::pair<int,long long>> neirank_cols_;

    SpRowInfo(const int &rank, const long long &row_idx, const int &neirank_max) : rank_(rank), idx_(row_idx) 
    { 
      neirank_cols_.reserve(neirank_max); 
    }
    ~SpRowInfo() = default;

    void mapColVal(const long long &col_idx, const double &val) 
    { 
      loc_colval_[col_idx] += val; 
    }
    void mapColVal(const int &rank, const long long &col_idx, const double &val) 
    {
      if (rank == rank_)
        mapColVal(col_idx, val);
      else
      {
        bd_colval_[col_idx] += val;
        neirank_cols_.push_back({rank, col_idx});
      }
    }
};

class LocalSpMatDnVec 
{
  public:
    LocalSpMatDnVec(const int &rank, const MPI_Comm &m_comm, const int &comm_size); 
    ~LocalSpMatDnVec() = default;

    int rank_;
    MPI_Comm m_comm_;
    int comm_size_; 

    int m_;
    int halo_;
    int loc_nnz_;
    int bd_nnz_;

    // Local rows of linear system + dense vecs
    std::vector<double> loc_cooValA_;
    std::vector<long long> loc_cooRowA_long_;
    std::vector<long long> loc_cooColA_long_;
    std::vector<double> x_;
    std::vector<double> b_;

    // Non-local rows with columns belonging to halo using rank-local indexing
    std::vector<double> bd_cooValA_;
    std::vector<long long> bd_cooRowA_long_;
    std::vector<long long> bd_cooColA_long_;

    // Identical to above, but with integers
    std::vector<int> loc_cooRowA_int_;
    std::vector<int> loc_cooColA_int_;
    std::vector<int> bd_cooRowA_int_;
    std::vector<int> bd_cooColA_int_;

    // bd_recv_set_[r] contains columns that need to be received from rank 'r'
    std::vector<std::set<long long>> bd_recv_set_;

    // Vectors that contain rules for sending and receiving
    std::vector<int> recv_ranks_;
    std::vector<int> recv_offset_;
    std::vector<int> recv_sz_;

    std::vector<int> send_ranks_;
    std::vector<int> send_offset_;
    std::vector<int> send_sz_;
    std::vector<int> send_buff_pack_idx_;

    void reserve(const int &N);
    void cooPushBackVal(const double &val, const long long &row, const long long &col);
    void cooPushBackRow(const SpRowInfo &row);
    void make(const std::vector<long long>& Nrows_xcumsum);
};

#pragma once

#include <map>
#include <vector>
#include <set>
#include <mpi.h>

#define SZ_MSG_TAG  100
#define VEC_MSG_TAG 101

class SpRowInfo
{
  public:
    const long long idx_; // global row index
    std::map<long long, double> colval_; // col_idx->val map
    // neirank_cols_[i] holds {rank of non-local col, idx of non-local col}
    std::vector<std::pair<int,long long>> neirank_cols_;

    SpRowInfo(const long long &row_idx, const int &neirank_max) : idx_(row_idx) 
    { 
      neirank_cols_.reserve(neirank_max); 
    }
    ~SpRowInfo() = default;

    void mapColVal(const long long &col_idx, const double &val) 
    { 
      colval_[col_idx] += val; 
    }
    void logNeiRankCol(const int &rank, const long long &col_idx) 
    {
      neirank_cols_.push_back({rank, col_idx});
    }
};

class LocalSpMatDnVec 
{
  public:
    LocalSpMatDnVec(const int rank, MPI_Comm m_comm, const int comm_size); 
    ~LocalSpMatDnVec() = default;

    int rank_;
    int comm_size_; 
    MPI_Comm m_comm_;

    int m_;
    int loc_nnz_;
    int bd_nnz_;
    int lower_halo_;
    int upper_halo_;

    // Local rows of linear system + dense vecs
    std::vector<double> loc_cooValA_;
    std::vector<long long> loc_cooRowA_;
    std::vector<long long> loc_cooColA_;
    std::vector<double> x_;
    std::vector<double> b_;

    // Non-local rows with columns belonging to halo using rank-local indexing
    std::vector<double> bd_cooValA_;
    std::vector<long long> bd_cooRowA_;
    std::vector<long long> bd_cooColA_;

    // bd_recv_set_[r] contains columns that need to be received from rank 'r'
    std::vector<std::set<long long>> bd_recv_set_;
    // bd_recv_vec_[r] same as bd_recv_vec_[r] but in contigious memory
    std::vector<std::vector<long long>> bd_recv_vec_;
    std::vector<std::vector<long long>> bd_send_vec_;

    // Identical to above, but with integers
    std::vector<int> loc_cooRowA_int_;
    std::vector<int> loc_cooColA_int_;
    std::vector<int> bd_cooRowA_int_;
    std::vector<int> bd_cooColA_int_;
    std::vector<std::vector<int>> bd_send_vec_int_;

    void reserve(const int &N);
    void cooPushBackVal(const double &val, const long long &row, const long long &col);
    void cooPushBackRow(const SpRowInfo &row);
    void make(const std::vector<long long>& Nrows_xcumsum);
};

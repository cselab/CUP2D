#include <unordered_map>
#include <algorithm>
//#include <execution>
#include <iostream>

#include "LocalSpMatDnVec.h"

#define SZ_MSG_TAG  100
#define VEC_MSG_TAG 101

LocalSpMatDnVec::LocalSpMatDnVec(const int &rank, const MPI_Comm &m_comm, const int &comm_size) 
  : rank_(rank), m_comm_(m_comm), comm_size_(comm_size), bd_recv_set_(comm_size)
{
  recv_ranks_.reserve(comm_size); 
  recv_offset_.reserve(comm_size); 
  recv_sz_.reserve(comm_size); 
  send_ranks_.reserve(comm_size); 
  send_offset_.reserve(comm_size); 
  send_sz_.reserve(comm_size); 
}

void LocalSpMatDnVec::reserve(const int &N)
{
  m_ = N;

  // Clear previous contents and reserve excess memory
  for (size_t i(0); i < bd_recv_set_.size(); i++)
    bd_recv_set_[i].clear();

  loc_cooValA_.clear(); loc_cooValA_.reserve(6*N);
  loc_cooRowA_.clear(); loc_cooRowA_.reserve(6*N);
  loc_cooColA_.clear(); loc_cooColA_.reserve(6*N);
  bd_cooValA_.clear(); bd_cooValA_.reserve(N);
  bd_cooRowA_.clear(); bd_cooRowA_.reserve(N);
  bd_cooColA_.clear(); bd_cooColA_.reserve(N);

  x_.resize(N);
  b_.resize(N);

  loc_cooRowA_int_.clear(); loc_cooRowA_int_.reserve(6*N);
  loc_cooColA_int_.clear(); loc_cooColA_int_.reserve(6*N);
  bd_cooRowA_int_.clear(); bd_cooRowA_int_.reserve(N);
  bd_cooColA_int_.clear(); bd_cooColA_int_.reserve(N);
}

void LocalSpMatDnVec::cooPushBackVal(const double &val, const long long &row, const long long &col)
{
  loc_cooValA_.push_back(val);  
  loc_cooRowA_.push_back(row);
  loc_cooColA_.push_back(col);
}

void LocalSpMatDnVec::cooPushBackRow(const SpRowInfo &row)
{
  if (row.neirank_cols_.empty())
  { // We are dealing with local rows
    for (const auto &[col_idx, val] : row.colval_)
    {
      loc_cooValA_.push_back(val);  
      loc_cooRowA_.push_back(row.idx_);
      loc_cooColA_.push_back(col_idx);
    }
  } 
  else 
  { // Non empty send/recv vector means non-local rows
    for (const auto &[col_idx, val] : row.colval_)
    {
      bd_cooValA_.push_back(val);  
      bd_cooRowA_.push_back(row.idx_);
      bd_cooColA_.push_back(col_idx);
    }
    // Update recv set
    for (const auto &[rank, col_idx] : row.neirank_cols_)
      bd_recv_set_[rank].insert(col_idx);
  }
}

void LocalSpMatDnVec::make(const std::vector<long long> &Nrows_xcumsum)
{
  loc_nnz_ = loc_cooValA_.size();
  bd_nnz_  = bd_cooValA_.size();
  lower_halo_ = 0;
  upper_halo_ = 0;
  
  std::vector<int> bd_recv_sz(comm_size_); // ensure buffer available throughout function scope for MPI
  std::vector<std::vector<long long>> bd_recv_vec(comm_size_);
  for (int r(0); r < comm_size_; r++)
  {
    // Convert recv_set -> recv_vec to prep for communication
    bd_recv_vec[r] = std::vector<long long>(bd_recv_set_[r].begin(), bd_recv_set_[r].end());    

    bd_recv_sz[r] = (int)bd_recv_vec[r].size(); // cast as int for MPI
    if (r < rank_)
      lower_halo_ += bd_recv_sz[r];
    else if (r > rank_)
      upper_halo_ += bd_recv_sz[r];

    // Each rank knows what it needs to receive, but needs to tell other ranks what to send to it
    if (r != rank_)
    {
      MPI_Request req1;
      MPI_Isend(&bd_recv_sz[r], 1, MPI_INT, r, SZ_MSG_TAG, m_comm_, &req1); // send size of upcoming message
      if (!bd_recv_vec[r].empty())
      {
        MPI_Request req2;
        MPI_Isend(bd_recv_vec[r].data(), bd_recv_sz[r], MPI_LONG_LONG, r, VEC_MSG_TAG, m_comm_, &req2); // send message
      }
    }
  }

  // Make receiving rules into halos of vectors
  recv_ranks_.clear();
  recv_offset_.clear();
  recv_sz_.clear();
  int offset = 0;
  for (int r(0); r < rank_; r++)
  {
    if (!bd_recv_vec[r].empty())
    {
      recv_ranks_.push_back(r); 
      recv_offset_.push_back(offset);
      recv_sz_.push_back(bd_recv_sz[r]);
      offset += bd_recv_sz[r];
    }
  }
  offset = 0;
  for (int r(rank_+1); r <comm_size_; r++)
  {
    if (!bd_recv_vec[r].empty())
    {
      recv_ranks_.push_back(r); 
      recv_offset_.push_back(offset + lower_halo_ + m_);
      recv_sz_.push_back(bd_recv_sz[r]);
      offset += bd_recv_sz[r];
    }
  }


  // Receive from other ranks the indices they will require for SpMV
  std::vector<std::vector<long long>> bd_send_vec(comm_size_);
  std::vector<MPI_Request> recv_req(comm_size_);
  for (int r(0); r < comm_size_; r++)
  {
    if (r != rank_)
    {
      int bd_send_sz;
      MPI_Recv(&bd_send_sz, 1, MPI_INT, r, SZ_MSG_TAG, m_comm_, MPI_STATUS_IGNORE);
      bd_send_vec[r] = std::vector<long long>(bd_send_sz);
      if (!bd_send_vec[r].empty())
        MPI_Irecv(bd_send_vec[r].data(), bd_send_sz, MPI_LONG_LONG, r, VEC_MSG_TAG, m_comm_, &recv_req[r]);
    }
  }

  // Make sending rules from a 'send' buffer
  send_ranks_.clear();
  send_offset_.clear();
  send_sz_.clear();
  offset = 0;
  for (int r(0); r < comm_size_; r++)
  {
    if (r !=rank_ && !bd_send_vec[r].empty())
    {
      send_ranks_.push_back(r);
      send_offset_.push_back(offset);
      send_sz_.push_back(bd_send_vec[r].size());
      offset += bd_send_vec[r].size();
    }
  }
  send_buff_pack_idx_.resize(offset);

  // Now re-index the linear system from global to local indexing
  // First shift the entire linear system such that the first global index of purely local row starts after lower halo
  const long long shift = -Nrows_xcumsum[rank_] + (long long)lower_halo_;
  auto shift_func = [&shift](const long long &s) -> int { return int(s + shift); };
  std::transform(loc_cooRowA_.begin(), loc_cooRowA_.end(), loc_cooRowA_int_.begin(), shift_func);
  std::transform(loc_cooColA_.begin(), loc_cooColA_.end(), loc_cooColA_int_.begin(), shift_func);
  std::transform(bd_cooRowA_.begin(), bd_cooRowA_.end(), bd_cooRowA_int_.begin(), shift_func);
  std::transform(bd_cooColA_.begin(), bd_cooColA_.end(), bd_cooColA_int_.begin(), shift_func);

  // Map indices of columns from other ranks to the lower and upper halos (while accounting for the shift)
  std::unordered_map<long long, int> bd_glob_loc; 
  bd_glob_loc.reserve(lower_halo_ + upper_halo_);
  int loc_idx = 0;
  for (int r(0); r < rank_; r++)
  for (size_t i(0); i < bd_recv_vec[r].size(); i++)
  {
    bd_glob_loc[bd_recv_vec[r][i]+shift] = loc_idx;
    loc_idx++;
  }
  loc_idx = 0;
  for (int r(rank_+1); r < comm_size_; r++)
  for (size_t i(0); i < bd_recv_vec[r].size(); i++)
  {
    bd_glob_loc[bd_recv_vec[r][i]+shift] = m_ + lower_halo_ + loc_idx;
    loc_idx++;
  }

  // Reindex shifted halos to local indexing
  auto halo_reindex_func = [this, &bd_glob_loc](int &s) {
    if (s < this->lower_halo_ || s > this->lower_halo_+this->m_)
      s = bd_glob_loc[s];
  };
  std::for_each(bd_cooColA_int_.begin(), bd_cooColA_int_.end(), halo_reindex_func);

  // Block until bd_send_vec received indices other ranks require
  for (int r(0); r < comm_size_; r++)
  {
    if (r != rank_ && !bd_send_vec[r].empty())
      MPI_Wait(&recv_req[r], MPI_STATUS_IGNORE);
  }

  // Local indexing for vector information to send to other ranks
  for (size_t i(0); i < send_ranks_.size(); i++)
    std::transform(bd_send_vec[send_ranks_[i]].begin(), bd_send_vec[send_ranks_[i]].end(), &send_buff_pack_idx_[send_offset_[i]], shift_func);

}


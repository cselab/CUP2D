#include <unordered_map>
#include <algorithm>
//#include <execution>
#include <iostream>

#include "LocalSpMatDnVec.h"

#define SZ_MSG_TAG  100
#define VEC_MSG_TAG 101

LocalSpMatDnVec::LocalSpMatDnVec(MPI_Comm m_comm) 
  : m_comm_(m_comm)
{
  // MPI
  MPI_Comm_rank(m_comm_, &rank_);
  MPI_Comm_size(m_comm_, &comm_size_);

  bd_recv_set_.resize(comm_size_);
  bd_recv_vec_.resize(comm_size_);
  recv_ranks_.reserve(comm_size_); 
  recv_offset_.reserve(comm_size_); 
  recv_sz_.reserve(comm_size_);
  send_ranks_.reserve(comm_size_); 
  send_offset_.reserve(comm_size_); 
  send_sz_.reserve(comm_size_); 
}

void LocalSpMatDnVec::reserve(const int &N)
{
  m_ = N;

  // Clear previous contents and reserve excess memory
  for (size_t i(0); i < bd_recv_set_.size(); i++)
    bd_recv_set_[i].clear();

  loc_cooValA_.clear(); loc_cooValA_.reserve(6*N);
  loc_cooRowA_long_.clear(); loc_cooRowA_long_.reserve(6*N);
  loc_cooColA_long_.clear(); loc_cooColA_long_.reserve(6*N);
  bd_cooValA_.clear(); bd_cooValA_.reserve(N);
  bd_cooRowA_long_.clear(); bd_cooRowA_long_.reserve(N);
  bd_cooColA_long_.clear(); bd_cooColA_long_.reserve(N);

  x_.resize(N);
  b_.resize(N);
}

void LocalSpMatDnVec::cooPushBackVal(const double &val, const long long &row, const long long &col)
{
  loc_cooValA_.push_back(val);  
  loc_cooRowA_long_.push_back(row);
  loc_cooColA_long_.push_back(col);
}

void LocalSpMatDnVec::cooPushBackRow(const SpRowInfo &row)
{
  for (const auto &[col_idx, val] : row.loc_colval_)
  {
    loc_cooValA_.push_back(val);  
    loc_cooRowA_long_.push_back(row.idx_);
    loc_cooColA_long_.push_back(col_idx);
  }
  if (!row.neirank_cols_.empty())
  { 
    for (const auto &[col_idx, val] : row.bd_colval_)
    {
      bd_cooValA_.push_back(val);  
      bd_cooRowA_long_.push_back(row.idx_);
      bd_cooColA_long_.push_back(col_idx);
    }
    // Update recv set
    for (const auto &[rank, col_idx] : row.neirank_cols_)
    {
      bd_recv_set_[rank].insert(col_idx);
    }
  }
}

void LocalSpMatDnVec::make(const std::vector<long long> &Nrows_xcumsum)
{
  loc_nnz_ = loc_cooValA_.size();
  bd_nnz_  = bd_cooValA_.size();
  
  halo_ = 0;
  std::vector<int> bd_recv_sz(comm_size_); // ensure buffer available throughout function scope for MPI
  for (int r(0); r < comm_size_; r++)
  {
    if (r != rank_)
    {
      // Convert recv_set -> recv_vec to prep for communication
      bd_recv_vec_[r] = std::vector<long long>(bd_recv_set_[r].begin(), bd_recv_set_[r].end());    

      bd_recv_sz[r] = (int)bd_recv_vec_[r].size(); // cast as int for MPI
      halo_ += bd_recv_sz[r];

      // Each rank knows what it needs to receive, but needs to tell other ranks what to send to it
      MPI_Request req1;
      MPI_Isend(&bd_recv_sz[r], 1, MPI_INT, r, SZ_MSG_TAG, m_comm_, &req1); // send size of upcoming message
      if (!bd_recv_vec_[r].empty())
      {
        MPI_Request req2;
        MPI_Isend(bd_recv_vec_[r].data(), bd_recv_sz[r], MPI_LONG_LONG, r, VEC_MSG_TAG, m_comm_, &req2); // send message
      }
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

  // Set receiving rules into halo
  recv_ranks_.clear();
  recv_offset_.clear();
  recv_sz_.clear();
  int offset = 0;
  for (int r(0); r < comm_size_; r++)
  {
    if (r != rank_ && !bd_recv_vec_[r].empty())
    {
      recv_ranks_.push_back(r); 
      recv_offset_.push_back(offset);
      recv_sz_.push_back(bd_recv_vec_[r].size());
      offset += (int)bd_recv_vec_[r].size();
    }
  }

  // Set sending rules from a 'send' buffer
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
      offset += (int)bd_send_vec[r].size();
    }
  }
  send_buff_pack_idx_.resize(offset);

  // Now re-index the linear system from global to local indexing
  const long long shift = -Nrows_xcumsum[rank_];
  auto shift_func = [&shift](const long long &s) -> int { return int(s + shift); };
  loc_cooRowA_int_.resize(loc_nnz_);
  loc_cooColA_int_.resize(loc_nnz_);
  bd_cooRowA_int_.resize(bd_nnz_);
  // Shift rows and columns to local indexing
  std::transform(loc_cooRowA_long_.begin(), loc_cooRowA_long_.end(), loc_cooRowA_int_.begin(), shift_func);
  std::transform(loc_cooColA_long_.begin(), loc_cooColA_long_.end(), loc_cooColA_int_.begin(), shift_func);
  std::transform(bd_cooRowA_long_.begin(), bd_cooRowA_long_.end(), bd_cooRowA_int_.begin(), shift_func);

  // Map indices of columns from other ranks to the halo
  std::unordered_map<long long, int> glob_halo; 
  glob_halo.reserve(halo_);
  int halo_idx = m_;
  for (size_t i(0); i < recv_ranks_.size(); i++)
  for (size_t j(0); j < bd_recv_vec_[recv_ranks_[i]].size(); j++)
  {
    glob_halo[bd_recv_vec_[recv_ranks_[i]][j]] = halo_idx;
    halo_idx++;
  }

  // Reindex columns belonging to other ranks to the halo
  auto halo_reindex_func = [&glob_halo](const long long &s) -> int { return glob_halo[s]; };
  bd_cooColA_int_.resize(bd_nnz_);
  std::transform(bd_cooColA_long_.begin(), bd_cooColA_long_.end(), bd_cooColA_int_.begin(), halo_reindex_func);

  // Block until bd_send_vec received indices other ranks require
  for (int r(0); r < comm_size_; r++)
  {
    if (r != rank_ && !bd_send_vec[r].empty())
      MPI_Wait(&recv_req[r], MPI_STATUS_IGNORE);
  }

  // Local indexing for vector information to send to other ranks
  for (size_t i(0); i < send_ranks_.size(); i++)
    std::transform(bd_send_vec[send_ranks_[i]].begin(), bd_send_vec[send_ranks_[i]].end(), &send_buff_pack_idx_[send_offset_[i]], shift_func);

  std::cerr << "  [LocalLS]: Rank: " << rank_ << ", m: " << m_ << ", halo: " << halo_ << std::endl;
}


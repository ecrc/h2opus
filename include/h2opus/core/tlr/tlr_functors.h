#ifndef __H2OPUS_TLR_FUNCTORS_H__
#define __H2OPUS_TLR_FUNCTORS_H__

template <class T> struct TLR_ColRankUpdate
{
  private:
    T *U_col_base, *V_col_base;
    T **U_ptrs, **V_ptrs;
    int block_size, row_offset, col_index, n_block;
    int *rank_prefix_sum, *block_ranks;
    const int *new_col_ranks;
    bool sym;

  public:
    TLR_ColRankUpdate(T *U_col_base, T *V_col_base, T **U_ptrs, T **V_ptrs, int *block_ranks, const int *new_col_ranks,
                      int *rank_prefix_sum, int block_size, int col_index, int n_block, int row_offset, bool sym)
    {
        this->U_col_base = U_col_base;
        this->V_col_base = V_col_base;
        this->U_ptrs = U_ptrs;
        this->V_ptrs = V_ptrs;
        this->block_ranks = block_ranks;
        this->new_col_ranks = new_col_ranks;
        this->rank_prefix_sum = rank_prefix_sum;
        this->block_size = block_size;
        this->col_index = col_index;
        this->n_block = n_block;
        this->row_offset = row_offset;
        this->sym = sym;
    }

    inline __host__ __device__ void operator()(const unsigned int &update_index) const
    {
        int row_index = update_index + row_offset;
        int linear_index = col_index * n_block + row_index;

        if (row_index != col_index)
        {
            int rank_offset = (update_index == 0 ? 0 : rank_prefix_sum[update_index - 1]);
            int offset = block_size * rank_offset;
            U_ptrs[linear_index] = U_col_base + offset;
            V_ptrs[linear_index] = V_col_base + offset;
            block_ranks[linear_index] = new_col_ranks[update_index];

            if (sym)
            {
                linear_index = row_index * n_block + col_index;
                U_ptrs[linear_index] = V_col_base + offset;
                V_ptrs[linear_index] = U_col_base + offset;
                block_ranks[linear_index] = new_col_ranks[update_index];
            }
        }
    }
};

template <class T> struct TLR_ClearUpperTriangle
{
  private:
    T **U_ptrs, **V_ptrs;
    int *block_ranks;
    int n;

  public:
    TLR_ClearUpperTriangle(T **U_ptrs, T **V_ptrs, int *block_ranks, int n)
    {
        this->U_ptrs = U_ptrs;
        this->V_ptrs = V_ptrs;
        this->block_ranks = block_ranks;
        this->n = n;
    }

    inline __host__ __device__ void operator()(const unsigned int &k) const
    {
        int i = k % n, j = k / n;
        if (i < j)
        {
            U_ptrs[k] = V_ptrs[k] = NULL;
            block_ranks[k] = 0;
        }
    }
};

template <class T> struct TLR_SetTransposePtrs
{
  private:
    T **U_ptrs, **V_ptrs;
    int col_index, block_start, n_block;
    int *ranks;

  public:
    TLR_SetTransposePtrs(T **U_ptrs, T **V_ptrs, int *ranks, int col_index, int block_start, int n_block)
    {
        this->U_ptrs = U_ptrs;
        this->V_ptrs = V_ptrs;
        this->ranks = ranks;
        this->col_index = col_index;
        this->block_start = block_start;
        this->n_block = n_block;
    }

    inline __host__ __device__ void operator()(const unsigned int &i) const
    {
        int tile_index = col_index * n_block + i + block_start;
        int sym_tile_index = col_index + (i + block_start) * n_block;

        U_ptrs[sym_tile_index] = V_ptrs[tile_index];
        V_ptrs[sym_tile_index] = U_ptrs[tile_index];
        ranks[sym_tile_index] = ranks[tile_index];
    }
};

#endif

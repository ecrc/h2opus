#include <h2opus/core/hara_weak_util.h>
#include <h2opus/core/hcompress.h>
#include <h2opus/core/hlru.h>
#include <h2opus/core/horthog.h>

#include <h2opus/util/debug_routines.h>
#include <h2opus/util/gpu_err_check.h>
#include <h2opus/util/timer.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Template routines
/////////////////////////////////////////////////////////////////////////////////////////////////////////
template <int hw>
void hara_template(HMatrixSampler *sampler, THMatrix<hw> &hmatrix, int max_rank, int r, H2Opus_Real eps, int bs,
                   h2opusHandle_t h2opus_handle, bool verbose)
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;
    // typedef typename VectorContainer<hw, int >::type IntVector;
    // typedef typename VectorContainer<hw, H2Opus_Real*>::type RealPointerArray;

    assert(bs == 16 || bs == 32);

    int depth = hmatrix.u_basis_tree.depth;
    int n = hmatrix.n;

    // We reuse the samples workspace to construct the dense diagonal leaves
    // so make sure we have enough memory to accomodate that update
    int sample_ws_cols = std::max(max_rank, hmatrix.u_basis_tree.leaf_size);
    RealVector sampled_U(n * sample_ws_cols, 0), sampled_V(n * sample_ws_cols, 0);

    TLowRankUpdate<hw> low_rank_update;
    TWeightAccelerationPacket<hw> packet(0, 0, 0);

    std::vector<int> samples_per_level(depth, 0);

    Timer<hw> timer;
    timer.init();

    std::vector<double> sampling_time(depth, 0), lru_time(depth, 0);
    std::vector<double> orthog_time(depth, 0), trunc_time(depth, 0);
    std::vector<double> level_time(depth, 0);

    double total_sampling_time = 0, total_lru_time = 0, total_orthog_time = 0, total_trunc_time = 0;
    double dense_sampling_time, dense_update_time, total_time = 0;
    int total_samples = 0;
    const H2Opus_Real ara_scale = 10 * sqrt(2 / M_PI);

    for (int level = 1; level < depth; level++)
    {
        if (verbose)
            printf("=============================================\nSampling level "
                   "%d\n=============================================\n",
                   level);

        H2Opus_Real ara_tol = eps / ara_scale; // / (depth - level);
        H2Opus_Real trunc_tol = ara_tol / 2;

        timer.start();

        hara_weak_admissibility_low_rank_update(sampler, hmatrix, low_rank_update, vec_ptr(sampled_U),
                                                vec_ptr(sampled_V), level, max_rank, r, ara_tol, bs, h2opus_handle);

        sampling_time[level] = timer.stop();
        samples_per_level[level] = 2 * low_rank_update.total_rank;
        total_samples += samples_per_level[level];

        if (verbose)
            printf("Level %d samples = %d\n", level, samples_per_level[level]);

        low_rank_update.setRankPerUpdate(bs);
        int done_applying = (low_rank_update.total_rank > 0 ? H2OPUS_LRU_NOT_DONE : H2OPUS_LRU_DONE);

        while (done_applying != H2OPUS_LRU_DONE)
        {
            timer.start();
            done_applying = hlru_sym(hmatrix, low_rank_update, h2opus_handle);
            lru_time[level] += timer.stop();

            timer.start();
            horthog(hmatrix, h2opus_handle);
            orthog_time[level] += timer.stop();

            timer.start();
            hcompress(hmatrix, packet, trunc_tol, h2opus_handle);
            trunc_time[level] += timer.stop();
        }

        level_time[level] = sampling_time[level] + lru_time[level] + orthog_time[level] + trunc_time[level];
        total_sampling_time += sampling_time[level];
        total_lru_time += lru_time[level];
        total_orthog_time += orthog_time[level];
        total_trunc_time += trunc_time[level];
        total_time += level_time[level];
    }

    // Generate diagonal dense updates - use the U and V samples as temporary data
    TDenseBlockUpdate<hw> dense_update;

    timer.start();
    hara_weak_admissibility_dense_update(sampler, hmatrix, vec_ptr(sampled_U), vec_ptr(sampled_V), dense_update,
                                         h2opus_handle);
    dense_sampling_time = timer.stop();

    timer.start();
    hlru_dense_block_update(hmatrix, dense_update, h2opus_handle);
    dense_update_time = timer.stop();

    total_lru_time += dense_update_time;
    total_sampling_time += dense_sampling_time;
    total_time += dense_update_time + dense_sampling_time;
    total_samples += hmatrix.u_basis_tree.leaf_size;

    if (verbose)
    {
        printf("%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n", "Level", "Sampling", "LRU", "Orthog", "Trunc",
               "Total", "Rank", "Samples");
        for (int level = 0; level < depth; level++)
        {
            printf("%-10d %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10d %-10d\n", level, sampling_time[level],
                   lru_time[level], orthog_time[level], trunc_time[level], level_time[level],
                   hmatrix.u_basis_tree.getLevelRank(level), samples_per_level[level]);
        }

        printf("%-10s %-10.4f %-10.4f %-10s %-10s %-10s %-10s %-10d\n", "Dense", dense_sampling_time, dense_update_time,
               "-", "-", "-", "-", hmatrix.u_basis_tree.leaf_size);

        printf("%-10s %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10s %-10d\n", "Overall", total_sampling_time,
               total_lru_time, total_orthog_time, total_trunc_time, total_time, "-", total_samples);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Interface routines
/////////////////////////////////////////////////////////////////////////////////////////////////////////
void hara(HMatrixSampler *sampler, HMatrix &hmatrix, int max_rank, int r, H2Opus_Real eps, int bs,
          h2opusHandle_t h2opus_handle, bool verbose)
{
    hara_template<H2OPUS_HWTYPE_CPU>(sampler, hmatrix, max_rank, r, eps, bs, h2opus_handle, verbose);
}

#ifdef H2OPUS_USE_GPU
void hara(HMatrixSampler *sampler, HMatrix_GPU &hmatrix, int max_rank, int r, H2Opus_Real eps, int bs,
          h2opusHandle_t h2opus_handle, bool verbose)
{
    hara_template<H2OPUS_HWTYPE_GPU>(sampler, hmatrix, max_rank, r, eps, bs, h2opus_handle, verbose);
}
#endif

#ifndef __DENSE_ERROR_APPROXIMATION_H__
#define __DENSE_ERROR_APPROXIMATION_H__

#include <h2opus/core/hmatrix.h>
#include <h2opus/util/blas_wrappers.h>
#include <h2opus/util/debug_routines.h>

inline std::vector<int> FisherYatesShuffle(int samples, int elements)
{
    std::vector<int> a(samples), source(elements);

    for (int i = 0; i < elements; i++)
        source[i] = i;

    for (int i = 0; i < samples; i++)
    {
        int j = (int)((float)rand() / RAND_MAX * (elements - i - 1));
        a[i] = source[j];
        std::swap(source[j], source[elements - i - 1]);
    }
    return a;
}

template <class MatGen>
inline H2Opus_Real getHgemvApproximationError(int matrix_dim, MatGen &mat_gen, H2Opus_Real percentage_rows,
                                              H2Opus_Real *y_h, H2Opus_Real *x)
{
    int samples = percentage_rows * matrix_dim;

    double total_err = 0;
    std::vector<int> row_samples = FisherYatesShuffle(samples, matrix_dim);

#pragma omp parallel for reduction(+ : total_err)
    for (int sample = 0; sample < samples; sample++)
    {
        int i = row_samples[sample];
        double y_sample = 0;
        for (int j = 0; j < matrix_dim; j++)
            y_sample += mat_gen.generateEntry(i, j) * x[j];
        double err = (double)(y_sample - y_h[i]);
        total_err += err * err;
    }
    total_err /= samples;
    total_err *= matrix_dim;

    double y_norm = 0;
    for (int i = 0; i < matrix_dim; i++)
        y_norm += y_h[i] * y_h[i];
    // printf("Norm y: %e\n", sqrt(y_norm));
    return (H2Opus_Real)sqrt(total_err / y_norm);
    // return (H2Opus_Real)sqrt(total_err);
}

template <class MatGen>
inline H2Opus_Real getHgemvApproximationInfinityError(int matrix_dim, MatGen &mat_gen, H2Opus_Real *y_h, H2Opus_Real *x)
{
    double max_error = 0;

#pragma omp parallel for reduction(max : max_error)
    for (int i = 0; i < matrix_dim; i++)
    {
        double y_sample = 0;
        for (int j = 0; j < matrix_dim; j++)
            y_sample += mat_gen.generateEntry(i, j) * x[j];

        if (y_sample != 0)
        {
            double err = fabs((double)(y_sample - y_h[i]) / y_sample);
            if (err > max_error)
                max_error = err;
        }
    }
    return max_error;
}

template <class MatGen>
inline H2Opus_Real getHgemvApproximation2NormError(int matrix_dim, MatGen &mat_gen, H2Opus_Real *y_h, H2Opus_Real *x)
{
    double total_error = 0;
    double y_norm = 0;

#pragma omp parallel for reduction(+ : total_error)
    for (int i = 0; i < matrix_dim; i++)
    {
        double y_sample = 0;
        for (int j = 0; j < matrix_dim; j++)
            y_sample += mat_gen.generateEntry(i, j) * x[j];

        double err = (double)(y_sample - y_h[i]);
        total_error += err * err;
        y_norm += y_sample * y_sample;
    }

    return sqrt(total_error / y_norm);
}

template <class MatGen>
inline H2Opus_Real getApproximationErrorEstimate(HMatrix &hmatrix, MatGen &mat_gen, int largest_dim, int level_samples)
{
    BasisTree &u_basis_tree = hmatrix.u_basis_tree;
    BasisTree &v_basis_tree = (hmatrix.sym ? u_basis_tree : hmatrix.v_basis_tree);

    // Now expand the basis trees
    int max_rank = hmatrix.u_basis_tree.level_data.getLargestRank();

    H2Opus_Real *u_matrices = (H2Opus_Real *)malloc(sizeof(H2Opus_Real) * hmatrix.n * max_rank * u_basis_tree.depth);
    H2Opus_Real *v_matrices = (H2Opus_Real *)malloc(sizeof(H2Opus_Real) * hmatrix.n * max_rank * v_basis_tree.depth);

    assert(u_matrices && v_matrices);

    expandBasis(u_basis_tree, 0, u_matrices, 0, hmatrix.n, max_rank);
    expandBasis(v_basis_tree, 0, v_matrices, 0, hmatrix.n, max_rank);

    std::vector<H2Opus_Real> level_errors(hmatrix.hnodes.depth, 0);

#pragma omp parallel for
    for (int level = hmatrix.hnodes.depth - 1; level >= 0; level--)
    {
        H2Opus_Real *sv_matrix = (H2Opus_Real *)malloc(sizeof(H2Opus_Real) * largest_dim * max_rank);
        H2Opus_Real *usv_matrix = (H2Opus_Real *)malloc(sizeof(H2Opus_Real) * largest_dim * largest_dim);

        int level_start, level_end, level_size;
        hmatrix.hnodes.getCouplingLevelRange(level, level_start, level_end);
        level_size = hmatrix.hnodes.getCouplingLevelSize(level);
        std::vector<int> node_samples = FisherYatesShuffle(std::min(level_samples, level_size), level_size);

        H2Opus_Real level_error = 0;
        int averaged_nodes = 0;

        for (int node_sample = 0; node_sample < node_samples.size(); node_sample++)
        {
            int node_index = level_start + node_samples[node_sample];
            assert(node_index < level_end);

            int tree_index = hmatrix.hnodes.rank_leaf_tree_index[node_index];
            int u_index = hmatrix.hnodes.node_u_index[tree_index];
            int v_index = hmatrix.hnodes.node_v_index[tree_index];

            int u_1 = u_basis_tree.node_start[u_index], v_1 = v_basis_tree.node_start[v_index];
            int u_2 = u_1 + u_basis_tree.node_len[u_index] - 1, v_2 = v_1 + v_basis_tree.node_len[v_index] - 1;

            H2Opus_Real *s_matrix = hmatrix.hnodes.getCouplingMatrix(level, node_index - level_start);
            H2Opus_Real *u_matrix = u_matrices + hmatrix.n * max_rank * level + u_1;
            H2Opus_Real *v_matrix = v_matrices + hmatrix.n * max_rank * level + v_1;

            int v_len = v_basis_tree.node_len[v_index];
            int v_dim = v_basis_tree.getLevelRank(level);
            int u_len = u_basis_tree.node_len[u_index];

            if (v_len > largest_dim || u_len > largest_dim)
                continue;
            averaged_nodes++;

            // Calculate S * V'
            cblas_gemm(CblasNoTrans, CblasTrans, v_dim, v_len, v_dim, 1, s_matrix, v_dim, v_matrix, hmatrix.n, 0,
                       sv_matrix, v_dim);
            // Calculate U * (S * V')
            cblas_gemm(CblasNoTrans, CblasNoTrans, u_len, v_len, v_dim, 1, u_matrix, hmatrix.n, sv_matrix, v_dim, 0,
                       usv_matrix, u_len);

            for (int j = 0; j < u_len; j++)
            {
                for (int k = 0; k < v_len; k++)
                {
                    int full_matrix_i = u_1 + j;
                    int full_matrix_j = v_1 + k;
                    H2Opus_Real hmatrix_entry = usv_matrix[j + k * u_len];
                    // for(int l = 0; l < v_dim; l++)
                    //    hmatrix_entry += u_matrix[j + l * hmatrix.n] * sv_matrix[l + k * v_dim];
                    H2Opus_Real full_entry = mat_gen.generateEntry(full_matrix_i, full_matrix_j);
                    H2Opus_Real entry_error = (hmatrix_entry - full_entry);
                    level_error += entry_error * entry_error;
                }
            }
        }

        if (averaged_nodes == 0)
        {
            assert(level != hmatrix.hnodes.depth - 1);
            level_errors[level] = level_errors[level + 1];
        }
        else
        {
            level_errors[level] = level_error / averaged_nodes;
        }

        printf("Level %d error: %e\n", level, level_errors[level]);

        free(sv_matrix);
        free(usv_matrix);
    }

    H2Opus_Real error = 0.0;
    for (int level = 0; level < hmatrix.hnodes.depth; level++)
    {
        int level_size = hmatrix.hnodes.getCouplingLevelSize(level);
        error += level_errors[level] * level_size;
    }

    free(u_matrices);
    free(v_matrices);

    H2Opus_Real hfrob_norm = frobeniusHNorm(hmatrix);
    return sqrt(error) / hfrob_norm;
}

template <class MatGen> inline H2Opus_Real getApproximationError(HMatrix &hmatrix, MatGen &mat_gen)
{
    BasisTree &u_basis_tree = hmatrix.u_basis_tree;
    BasisTree &v_basis_tree = (hmatrix.sym ? u_basis_tree : hmatrix.v_basis_tree);

    // Now expand the basis trees
    int max_rank = hmatrix.u_basis_tree.level_data.getLargestRank();

    H2Opus_Real *u_matrices = (H2Opus_Real *)malloc(sizeof(H2Opus_Real) * hmatrix.n * max_rank * u_basis_tree.depth);
    H2Opus_Real *v_matrices = (H2Opus_Real *)malloc(sizeof(H2Opus_Real) * hmatrix.n * max_rank * v_basis_tree.depth);

    assert(u_matrices && v_matrices);

    expandBasis(u_basis_tree, 0, u_matrices, 0, hmatrix.n, max_rank);
    expandBasis(v_basis_tree, 0, v_matrices, 0, hmatrix.n, max_rank);

    H2Opus_Real error = 0.0;

    for (int level = 0; level < hmatrix.hnodes.depth; level++)
    {
        int level_start, level_end;
        hmatrix.hnodes.getCouplingLevelRange(level, level_start, level_end);

        for (int node_index = level_start; node_index < level_end; node_index++)
        {
            int tree_index = hmatrix.hnodes.rank_leaf_tree_index[node_index];
            int u_index = hmatrix.hnodes.node_u_index[tree_index];
            int v_index = hmatrix.hnodes.node_v_index[tree_index];

            int u_1 = u_basis_tree.node_start[u_index], v_1 = v_basis_tree.node_start[v_index];
            int u_2 = u_1 + u_basis_tree.node_len[u_index] - 1, v_2 = v_1 + v_basis_tree.node_len[v_index] - 1;

            H2Opus_Real *s_matrix = hmatrix.hnodes.getCouplingMatrix(level, node_index - level_start);
            H2Opus_Real *u_matrix = u_matrices + hmatrix.n * max_rank * level + u_1;
            H2Opus_Real *v_matrix = v_matrices + hmatrix.n * max_rank * level + v_1;

            int v_len = v_basis_tree.node_len[v_index];
            int v_dim = v_basis_tree.getLevelRank(level);

            H2Opus_Real *sv_matrix = (H2Opus_Real *)malloc(sizeof(H2Opus_Real) * v_len * v_dim);

            // Calculate S * V'
            for (int j = 0; j < v_dim; j++)
            {
                for (int k = 0; k < v_len; k++)
                {
                    int index = j + k * v_dim;
                    sv_matrix[index] = 0;
                    for (int l = 0; l < v_dim; l++)
                    {
                        int s_index = j + l * v_dim;
                        sv_matrix[index] += s_matrix[s_index] * v_matrix[k + l * hmatrix.n];
                    }
                }
            }

            // Calculate U * (S * V')
            int u_len = u_basis_tree.node_len[u_index];
            for (int j = 0; j < u_len; j++)
            {
                for (int k = 0; k < v_len; k++)
                {
                    int full_matrix_i = u_1 + j;
                    int full_matrix_j = v_1 + k;
                    H2Opus_Real hmatrix_entry = 0;
                    for (int l = 0; l < v_dim; l++)
                        hmatrix_entry += u_matrix[j + l * hmatrix.n] * sv_matrix[l + k * v_dim];
                    H2Opus_Real full_entry = mat_gen.generateEntry(full_matrix_i, full_matrix_j);
                    H2Opus_Real entry_error = (hmatrix_entry - full_entry);
                    // if(full_entry != 0) entry_error /= full_entry;
                    error += entry_error * entry_error;
                }
            }

            free(sv_matrix);
        }
    }

    free(u_matrices);
    free(v_matrices);

    H2Opus_Real hfrob_norm = frobeniusHNorm(hmatrix);
    return sqrt(error) / hfrob_norm;
}

#endif

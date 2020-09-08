#include <assert.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include <h2opus/core/h2opus_eps.h>
#include <h2opus/util/blas_wrappers.h>
#include <h2opus/util/debug_routines.h>

#include <thrust/random.h>
////////////////////////////////////////////////////////////////////////////////////////////////////
// Matrix structure visualization
////////////////////////////////////////////////////////////////////////////////////////////////////
// #define USE_LIB_BOARD

#ifdef USE_LIB_BOARD
#include <Board.h>
using namespace LibBoard;

// Recursive function that draws an hmatrix cell with the specified (x,y) upper left corner
// and (width, height) dimensions. If it's a full matrix, then draw it according to the colors in
// the HLib parameters, otherwise (rkmatrix) draw it with its rank in text inside the cell
void drawHmatrix(Board &board, HMatrix &hmatrix, float width, float height, int draw_level)
{
    const double LineWidth = 0.0015;
    const double font_base = 0.2;

    BasisTree &u_basis_tree = hmatrix.u_basis_tree;
    BasisTree &v_basis_tree = (hmatrix.sym ? u_basis_tree : hmatrix.v_basis_tree);
    int leaf_size = hmatrix.u_basis_tree.leaf_size;

    for (int level = 0; level < hmatrix.hnodes.depth; level++)
    {
        int level_start, level_end;
        hmatrix.hnodes.getLevelRange(level, level_start, level_end);
        int u_start_index = u_basis_tree.getLevelStart(level);
        int v_start_index = v_basis_tree.getLevelStart(level);

        // printf("Node range: %d %d\n", level_start, level_end);
        if (draw_level != -1 && level != draw_level)
            continue;
        for (int node_index = level_start; node_index < level_end; node_index++)
        {
            int node_type = hmatrix.hnodes.node_type[node_index];

            int u_index = hmatrix.hnodes.node_u_index[node_index];
            int v_index = hmatrix.hnodes.node_v_index[node_index];

            int cols = v_basis_tree.node_len[v_index];
            int rows = u_basis_tree.node_len[u_index];

            int col_start = v_basis_tree.node_start[v_index];
            int row_start = u_basis_tree.node_start[u_index];

            float node_y = height - height * (float)row_start / hmatrix.n;
            float node_x = width * (float)col_start / hmatrix.n;
            float node_width = width * (float)cols / hmatrix.n;
            float node_height = height * (float)rows / hmatrix.n;

            // int display_index = node_index;
            int display_index = hmatrix.hnodes.node_to_leaf[node_index];
            // if(u_index - u_start_index > v_index - v_start_index)
            //    display_index = hmatrix.hnodes.getHNodeIndex(level, v_index - v_start_index, u_index - u_start_index);

            if (node_type == HMATRIX_DENSE_MATRIX)
            {
                float c_inner[4] = {1, 0, 0, 1};
                board.setPenColorRGBf(c_inner[0], c_inner[1], c_inner[2], c_inner[3]);
                board.fillRectangle(node_x, node_y, node_width, node_height, 2);
            }
            else if (node_type == HMATRIX_RANK_MATRIX)
            {
                float c_inner[4] = {0.1, 1.0, 0.1, 1};
                // float c_inner[4] = {0.1, 0.1, 1.0, 1};
                board.setPenColorRGBf(c_inner[0], c_inner[1], c_inner[2], c_inner[3]);
                board.fillRectangle(node_x, node_y, node_width, node_height, 2);
                board.fillRectangle(node_x, node_y, 0.7 * (float)leaf_size / hmatrix.n,
                                    0.7 * (float)leaf_size / hmatrix.n, 2);
            }
            else if (draw_level != -1)
            {
                float c_inner[4] = {0.08, 0.72, 0.86, 1};
                board.setPenColorRGBf(c_inner[0], c_inner[1], c_inner[2], c_inner[3]);
                board.fillRectangle(node_x, node_y, node_width, node_height, 2);
            }

            if (node_type == HMATRIX_DENSE_MATRIX || node_type == HMATRIX_RANK_MATRIX || draw_level != -1)
            {
                float c_border[4] = {0, 0, 0, 1};
                board.setPenColorRGBf(c_border[0], c_border[1], c_border[2], c_border[3]);
                board.setLineWidth(LineWidth);
                board.drawRectangle(node_x, node_y, node_width, node_height, 1);

                /*std::stringstream stream;
                stream << display_index;
                std::string t = stream.str();
                float font_size = font_base * node_width;
                board.setFontSize(font_size).drawText(node_x + node_width / 2, node_y - node_height / 2, stream.str(),
                0); Rect text_bb = board.last<Text>().boundingBox(Shape::IgnoreLineWidth); double h = text_bb.height, w
                = text_bb.width; board.last<Text>().translate(-w/2, -h/2);*/
            }
        }
    }
}
#endif

void psPrintRect(std::ostream &out, float x, float y, float w, float h, float r, float g, float b, float linewidth,
                 bool fill, float fill_r, float fill_g, float fill_b, const char *text = NULL, int text_scale = 0,
                 float text_x = 0, float text_y = 0)
{
    out << "np" << std::endl
        << x << " " << y << " m" << std::endl
        << x + w << " " << y << " l" << std::endl
        << x + w << " " << y - h << " l" << std::endl
        << x << " " << y - h << " l" << std::endl
        << "cp" << std::endl;
    if (fill)
    {
        out << "gs" << std::endl
            << fill_r << " " << fill_g << " " << fill_b << " srgb" << std::endl
            << "fill" << std::endl
            << "gr" << std::endl;
    }
    out << r << " " << g << " " << b << " srgb" << std::endl << linewidth << " slw" << std::endl << "s" << std::endl;

    if (text)
    {
        out << "/Times-Roman ff" << std::endl;
        out << text_scale << " scf" << std::endl;
        out << "sf" << std::endl << "np" << std::endl;
        out << text_x << " " << text_y << " m" << std::endl;
        out << "(" << text << ") show" << std::endl;
    }
}

void drawHmatrix(std::ostream &out, HMatrix &hmatrix, float scale, int draw_level)
{
    int n = hmatrix.n;
    float bbox_padding = 0.05 * scale;
    float c_border[4] = {0, 0, 0, 1};
    float line_width = 0.015;

    out << "%!PS-Adobe-2.0 EPSF-2.0" << std::endl;
    out << "%%BoundingBox: " << -bbox_padding << " " << -bbox_padding << " " << scale + bbox_padding << " "
        << scale + bbox_padding << std::endl;
    out << "%%Origin: 0 0" << std::endl;
    out << "/np {newpath} bind def" << std::endl;
    out << "/m {moveto} bind def" << std::endl;
    out << "/l {lineto} bind def" << std::endl;
    out << "/srgb {setrgbcolor} bind def" << std::endl;
    out << "/cp {closepath} bind def" << std::endl;
    out << "/slw {setlinewidth} bind def" << std::endl;
    out << "/s {stroke} bind def" << std::endl;
    out << "/gs {gsave} bind def" << std::endl;
    out << "/gr {grestore} bind def" << std::endl;
    out << "/ff {findfont} bind def" << std::endl;
    out << "/sf {setfont} bind def" << std::endl;
    out << "/scf {scalefont} bind def" << std::endl;

    BasisTree &u_basis_tree = hmatrix.u_basis_tree;
    BasisTree &v_basis_tree = (hmatrix.sym ? u_basis_tree : hmatrix.v_basis_tree);

    for (int level = 0; level < hmatrix.hnodes.depth; level++)
    {
        int level_start, level_end;
        hmatrix.hnodes.getLevelRange(level, level_start, level_end);
        // int u_start_index = u_basis_tree.getLevelStart(level);
        // int v_start_index = v_basis_tree.getLevelStart(level);

        if (draw_level != -1 && level != draw_level)
            continue;
        for (int node_index = level_start; node_index < level_end; node_index++)
        {
            int node_type = hmatrix.hnodes.node_type[node_index];

            int u_index = hmatrix.hnodes.node_u_index[node_index];
            int v_index = hmatrix.hnodes.node_v_index[node_index];

            int cols = v_basis_tree.node_len[v_index];
            int rows = u_basis_tree.node_len[u_index];

            int col_start = v_basis_tree.node_start[v_index];
            int row_start = u_basis_tree.node_start[u_index];

            float node_width = cols * (scale / n);
            float node_height = rows * (scale / n);
            float node_x = col_start * (scale / n);
            float node_y = (n - row_start) * (scale / n);

            // int display_index = node_index;
            // int display_index = hmatrix.hnodes.node_to_leaf[node_index];
            // if(u_index - u_start_index > v_index - v_start_index)
            //    display_index = hmatrix.hnodes.getHNodeIndex(level, v_index - v_start_index, u_index - u_start_index);

            if (node_type == HMATRIX_DENSE_MATRIX)
            {
                std::stringstream stream;
                stream << col_start << "," << cols;

                float c_fill[4] = {1, 0, 0, 1};
                psPrintRect(out, node_x, node_y, node_width, node_height, c_border[0], c_border[1], c_border[2],
                            line_width, true, c_fill[0], c_fill[1], c_fill[2]
                            //, stream.str().c_str(), 12, node_x + node_width / 2, node_y - node_height / 2
                );
            }
            else if (node_type == HMATRIX_RANK_MATRIX)
            {
                float c_fill[4] = {0.1, 1.0, 0.1, 1};
                psPrintRect(out, node_x, node_y, node_width, node_height, c_border[0], c_border[1], c_border[2],
                            line_width, true, c_fill[0], c_fill[1], c_fill[2]);
            }
            else if (draw_level != -1)
            {
                float c_fill[4] = {0.08, 0.72, 0.86, 1};

                psPrintRect(out, node_x, node_y, node_width, node_height, c_border[0], c_border[1], c_border[2],
                            line_width, true, c_fill[0], c_fill[1], c_fill[2]);
            }
        }
    }
    out << "showpage" << std::endl;
    out << "%EOF" << std::endl;
}

void outputEps(HMatrix &h, const char *filename, int level)
{
#ifdef USE_LIB_BOARD
    Board board;
    drawHmatrix(board, h, 1, 1, level);
    board.scale(100);
    // board.setClippingRectangle(0, 0, 100, 100);
    board.saveEPS(filename);
#else
    std::ofstream out(filename);
    drawHmatrix(out, h, 1000, level);
    out.close();
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Simple RNG
////////////////////////////////////////////////////////////////////////////////////////////////////
void randomData(H2Opus_Real *m, size_t entries)
{
    randomData(m, entries, rand());
}

void randomData(H2Opus_Real *m, size_t entries, int seed)
{
    thrust::default_random_engine rng;
    rng.seed(seed);
    thrust::uniform_real_distribution<H2Opus_Real> dist(0, 1);
    for (size_t i = 0; i < entries; i++)
        m[i] = dist(rng);

    // thrust::generate(m, m + entries, _rand);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Block copies and prints
////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef H2OPUS_USE_GPU
template <class T> thrust::host_vector<T> copyGPUArrayT(T *data, int entries)
{
    thrust::device_ptr<T> dev_data(data);
    return thrust::host_vector<T>(dev_data, dev_data + entries);
}

thrust::host_vector<double *> copyGPUArray(double **data, int entries)
{
    return copyGPUArrayT<double *>(data, entries);
}

thrust::host_vector<float *> copyGPUArray(float **data, int entries)
{
    return copyGPUArrayT<float *>(data, entries);
}

thrust::host_vector<double> copyGPUArray(double *data, int entries)
{
    return copyGPUArrayT<double>(data, entries);
}

thrust::host_vector<float> copyGPUArray(float *data, int entries)
{
    return copyGPUArrayT<float>(data, entries);
}

thrust::host_vector<int> copyGPUArray(int *data, int entries)
{
    return copyGPUArrayT<int>(data, entries);
}

template <class T> thrust::host_vector<T> copyGPUBlockT(T *data, int ld, int rows, int cols)
{
    thrust::device_ptr<T> dev_data(data);
    thrust::host_vector<T> host_data(rows * cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            host_data[i + j * rows] = *(dev_data + i + j * ld);
    return host_data;
}

template <class T> thrust::host_vector<T> copyGPUBlockT(T **data_ptrs, int index, int ld, int rows, int cols)
{
    thrust::device_ptr<T *> dev_ptr(data_ptrs + index);
    T *dev_ptr_val = *dev_ptr;
    thrust::device_ptr<T> dev_data(dev_ptr_val);
    thrust::host_vector<T> host_data(rows * cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            host_data[i + j * rows] = *(dev_data + i + j * ld);
    return host_data;
}

thrust::host_vector<float> copyGPUBlock(float **data_ptrs, int index, int ld, int rows, int cols)
{
    return copyGPUBlockT<float>(data_ptrs, index, ld, rows, cols);
}

thrust::host_vector<double> copyGPUBlock(double **data_ptrs, int index, int ld, int rows, int cols)
{
    return copyGPUBlockT<double>(data_ptrs, index, ld, rows, cols);
}

thrust::host_vector<float> copyGPUBlock(float *data, int ld, int rows, int cols)
{
    return copyGPUBlockT<float>(data, ld, rows, cols);
}

thrust::host_vector<double> copyGPUBlock(double *data, int ld, int rows, int cols)
{
    return copyGPUBlockT<double>(data, ld, rows, cols);
}
#endif

template <class T> void printDenseMatrixHost(T *matrix, int ldm, int m, int n, int digits, const char *name)
{
    char format[64];
    assert(ldm >= m);

    sprintf(format, "%%.%de ", digits);
#ifdef H2OPUS_DOUBLE_PRECISION
    printf("%s = ([\n", name);
#else
    printf("%s = single([\n", name);
#endif
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            printf(format, matrix[i + j * ldm]);
        printf(";\n");
    }
    printf("]);\n");
}

template <class T> void printDenseMatrixT(T *matrix, int ldm, int m, int n, int digits, const char *name, int hw)
{
    if (hw == H2OPUS_HWTYPE_CPU)
        printDenseMatrixHost<T>(matrix, ldm, m, n, digits, name);
#ifdef H2OPUS_USE_GPU
    else
    {
        thrust::host_vector<T> host_matrix = copyGPUBlock(matrix, ldm, m, n);
        printDenseMatrixHost<T>(vec_ptr(host_matrix), ldm, m, n, digits, name);
    }
#endif
}

void printDenseMatrix(float *matrix, int ldm, int m, int n, int digits, const char *name, int hw)
{
    printDenseMatrixT<float>(matrix, ldm, m, n, digits, name, hw);
}

void printDenseMatrix(double *matrix, int ldm, int m, int n, int digits, const char *name, int hw)
{
    printDenseMatrixT<double>(matrix, ldm, m, n, digits, name, hw);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Orthogonality tests
////////////////////////////////////////////////////////////////////////////////////////////////////
void eye(std::vector<H2Opus_Real> &identity, int m)
{
    identity.resize(m * m);
    std::fill(identity.begin(), identity.end(), 0);
    for (int i = 0; i < m; i++)
        identity[i + i * m] = 1;
}

H2Opus_Real frobeniusError(H2Opus_Real *m1, H2Opus_Real *m2, int m, int n)
{
    H2Opus_Real err = 0;
    for (int i = 0; i < m * n; i++)
        err += (m1[i] - m2[i]) * (m1[i] - m2[i]);
    return sqrt(err);
}

H2Opus_Real orthogError(H2Opus_Real *m, int dim)
{
    // Get orthog error (so norm('fro', M - I) assuming M = A'*A for some A)
    // Ignore zero columns (i.e. M(i, i) = 0)
    H2Opus_Real err = 0;
    for (int i = 0; i < dim; i++)
    {
        if (m[i + i * dim] != 0)
        {
            for (int j = 0; j < dim; j++)
            {
                if (i != j)
                    err += m[i + j * dim] * m[i + j * dim];
            }
        }
    }
    return sqrt(err);
}

H2Opus_Real frobeniusNorm(H2Opus_Real *m, int rows, int cols)
{
    double norm = 0;
    for (int i = 0; i < rows * cols; i++)
        norm += m[i] * m[i];
    return (H2Opus_Real)sqrt(norm);
}

H2Opus_Real orthogonality(H2Opus_Real *matrix, int ld, int m, int n)
{
    std::vector<H2Opus_Real> prod(n * n, 0);
    cblas_gemm(CblasTrans, CblasNoTrans, n, n, m, 1, matrix, ld, matrix, ld, 0, &prod[0], n);
    return orthogError(&prod[0], n);
}

H2Opus_Real orthogonality(H2Opus_Real *matrix, int m, int n)
{
    return orthogonality(matrix, m, m, n);
}

H2Opus_Real getBasisOrthogonality(BasisTree &basis_tree, bool print_per_level)
{
    double orthog = 0;

    // int leaf_start = basis_tree.getLevelStart(basis_tree.depth - 1);
    int leaf_rank = basis_tree.getLevelRank(basis_tree.depth - 1);
    int leaf_size = basis_tree.leaf_size;

    for (int leaf = 0; leaf < basis_tree.basis_leaves; leaf++)
    {
        H2Opus_Real *u_leaf = basis_tree.getBasisLeaf(leaf);

        if (frobeniusNorm(u_leaf, leaf_size, leaf_rank) < H2OpusEpsilon<H2Opus_Real>::eps)
            continue;
        orthog += orthogonality(u_leaf, leaf_size, leaf_rank);
    }
    if (print_per_level && basis_tree.basis_leaves != 0)
        printf("Level %d: %e\n", basis_tree.depth - 1, orthog / basis_tree.basis_leaves);
    int total_nodes = basis_tree.basis_leaves;

    for (int level = basis_tree.depth - 2; level >= 0; level--)
    {
        int level_start, level_end;
        basis_tree.getLevelRange(level, level_start, level_end);
        int child_start = basis_tree.getLevelStart(level + 1);

        // Get the rows and cols of the nodes in this level
        int rows, cols;
        basis_tree.getTransferDims(level + 1, rows, cols);

        if (rows == 0 || cols == 0)
            continue;

        int stacked_rows = basis_tree.max_children * rows;
        std::vector<H2Opus_Real> stacked_children(stacked_rows * cols, 0);

        double level_orthog = 0;
        for (int node = level_start; node < level_end; node++)
        {
            int child_index = basis_tree.head[node];
            int r_start = 0;

            while (child_index != H2OPUS_EMPTY_NODE)
            {
                H2Opus_Real *e = basis_tree.getTransNode(level + 1, child_index - child_start);
                H2Opus_Real *s = &stacked_children[r_start];

                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        s[i + j * stacked_rows] = e[i + j * rows];

                r_start += rows;
                child_index = basis_tree.next[child_index];
            }

            if (frobeniusNorm(&stacked_children[0], stacked_rows, cols) < H2OpusEpsilon<H2Opus_Real>::eps)
                continue;
            level_orthog += orthogonality(&stacked_children[0], stacked_rows, cols);
        }
        int level_nodex = basis_tree.getLevelSize(level);
        total_nodes += level_nodex;
        if (print_per_level && level_nodex != 0)
            printf("Level %d: %e\n", level, level_orthog / level_nodex);
        orthog += level_orthog;
    }

    if (total_nodes != 0)
        orthog /= total_nodes;
    return (H2Opus_Real)orthog;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Compression errors
////////////////////////////////////////////////////////////////////////////////////////////////////
H2Opus_Real frobeniusHNorm(HMatrix &h_orthog, int dense, int low_rank)
{
    double d_frob_norm = 0, h_frob_norm = 0;

    int num_dense_leaves = h_orthog.hnodes.num_dense_leaves;
    int leaf_size = h_orthog.hnodes.leaf_size;

    for (int leaf = 0; leaf < num_dense_leaves; leaf++)
    {
        // int tree_index = h_orthog.hnodes.dense_leaf_tree_index[leaf];
        H2Opus_Real *m = h_orthog.hnodes.getDenseMatrix(leaf);
        double dense_fnorm = 0;

        for (int i = 0; i < leaf_size; i++)
            for (int j = 0; j < leaf_size; j++)
                dense_fnorm += m[i + j * leaf_size] * m[i + j * leaf_size];
        d_frob_norm += dense_fnorm;
    }

    for (int level = 0; level < h_orthog.hnodes.depth; level++)
    {
        int rank_orthog = h_orthog.hnodes.getLevelRank(level);

        int level_start, level_end;
        h_orthog.hnodes.getCouplingLevelRange(level, level_start, level_end);

        for (int leaf = level_start; leaf < level_end; leaf++)
        {
            H2Opus_Real *orthog_entries = h_orthog.hnodes.getCouplingMatrix(level, leaf - level_start);
            double orthog_fnorm = 0;
            for (int i = 0; i < rank_orthog; i++)
                for (int j = 0; j < rank_orthog; j++)
                    orthog_fnorm += orthog_entries[i + j * rank_orthog] * orthog_entries[i + j * rank_orthog];

            h_frob_norm += orthog_fnorm;
        }
    }
    // printf("%e %e\n", d_frob_norm, h_frob_norm);
    return (H2Opus_Real)sqrt((low_rank * h_frob_norm + dense * d_frob_norm));
}

H2Opus_Real frobeniusHNorm(HMatrix &h_orthog)
{
    return frobeniusHNorm(h_orthog, 1, 1);
}

H2Opus_Real truncationError(HMatrix &h_orthog, HMatrix &h_trunc)
{
    double trunc_err = 0;

    for (int level = 0; level < h_orthog.hnodes.depth; level++)
    {
        int rank_orthog = h_orthog.hnodes.getLevelRank(level);
        int rank_trunc = h_trunc.hnodes.getLevelRank(level);

        int level_start, level_end;
        h_orthog.hnodes.getCouplingLevelRange(level, level_start, level_end);

        double level_trunc_err = 0;

        for (int leaf = level_start; leaf < level_end; leaf++)
        {
            H2Opus_Real *orthog_entries = h_orthog.hnodes.getCouplingMatrix(level, leaf - level_start);
            H2Opus_Real *trunc_entries = h_trunc.hnodes.getCouplingMatrix(level, leaf - level_start);
            // int node_index = h_orthog.hnodes.rank_leaf_tree_index[leaf];

            double trunc_fnorm = 0, orthog_fnorm = 0;

            /*if(level == h_orthog.hnodes.depth - 1 && leaf == level_start)
               {
                printDenseMatrix(orthog_entries, rank_orthog, rank_orthog, 10, "S1");
                printDenseMatrix(trunc_entries, rank_trunc, rank_trunc, 10, "S2");
               }*/

            for (int i = 0; i < rank_orthog; i++)
                for (int j = 0; j < rank_orthog; j++)
                    orthog_fnorm += orthog_entries[i + j * rank_orthog] * orthog_entries[i + j * rank_orthog];

            for (int i = 0; i < rank_trunc; i++)
                for (int j = 0; j < rank_trunc; j++)
                    trunc_fnorm += trunc_entries[i + j * rank_trunc] * trunc_entries[i + j * rank_trunc];

            // orthog_fnorm = sqrt(orthog_fnorm);
            // trunc_fnorm  = sqrt(trunc_fnorm);

            // printf("Node %d Truncated: %e  -- Orthog: %e -- Diff = %e\n", node_index, trunc_fnorm, orthog_fnorm,
            // fabs(orthog_fnorm - trunc_fnorm) / orthog_fnorm);

            // level_trunc_err += (orthog_fnorm - trunc_fnorm) * (orthog_fnorm - trunc_fnorm);
            level_trunc_err += fabs(orthog_fnorm - trunc_fnorm);
        }
        int level_size = level_end - level_start;
        double level_err = sqrt(level_trunc_err);
        double avg_error = (level_size == 0 ? 0 : level_err / level_size);

        printf("Level %d truncation error = %e (avg %e)\n", level, level_err, avg_error);
        trunc_err += level_trunc_err;
    }

    return (H2Opus_Real)sqrt(trunc_err);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Printing out hmatrix data
////////////////////////////////////////////////////////////////////////////////////////////////////
void dumpMatrixTreeContainer(BasisTreeLevelData &level_data, std::vector<H2Opus_Real *> &Z_hat, int digits, int hw)
{
    char format[64];
    sprintf(format, "%%.%de ", digits);
    printf(" [\n");
    for (int i = 0; i < (int)Z_hat.size(); i++)
    {
        H2Opus_Real *Z_level = Z_hat[i];
        int rank = level_data.getLevelRank(i);
        int nodes = level_data.getLevelSize(i);
        if (rank == 0 || nodes == 0)
            continue;

        char tag[256];
        sprintf(tag, "Z%d", i);
        for (int i = 0; i < nodes; i++)
        {
            H2Opus_Real *matrix_data = Z_level + rank * rank * i;
            std::ostringstream out;
            out << tag << i;
            printDenseMatrix(matrix_data, rank, rank, rank, digits, out.str().c_str(), hw);
        }
    }
    printf(" ];\n");
}

void dumpHgemvTreeContainer(BasisTreeLevelData &level_data, std::vector<H2Opus_Real *> &Z_hat, int num_vectors,
                            int digits, int hw)
{
    char format[64];
    sprintf(format, "%%.%de ", digits);
    printf(" [\n");
    for (int i = 0; i < (int)Z_hat.size(); i++)
    {
        H2Opus_Real *Z_level = Z_hat[i];
        int rank = level_data.getLevelRank(i);
        int nodes = level_data.getLevelSize(i);
        if (rank == 0 || nodes == 0)
            continue;

        char tag[256];
        sprintf(tag, "Z%d", i);
        for (int i = 0; i < nodes; i++)
        {
            H2Opus_Real *matrix_data = Z_level + rank * num_vectors * i;
            std::ostringstream out;
            out << tag << i;
            printDenseMatrix(matrix_data, rank, rank, num_vectors, digits, out.str().c_str(), hw);
        }
    }
    printf(" ];\n");
}

void dumpCouplingMatrices(HMatrix &hmatrix, int digits)
{
    char format[64];
    sprintf(format, "%%.%de ", digits);
    BasisTree &u_basis_tree = hmatrix.u_basis_tree;
    BasisTree &v_basis_tree = (hmatrix.sym ? u_basis_tree : hmatrix.v_basis_tree);

    printf("Rank Matrices [\n");
    for (int level = 0; level < hmatrix.hnodes.depth; level++)
    {
        int rank = hmatrix.hnodes.getLevelRank(level);
        int level_start, level_end;
        hmatrix.hnodes.getCouplingLevelRange(level, level_start, level_end);

        for (int leaf = level_start; leaf < level_end; leaf++)
        {
            H2Opus_Real *entries = hmatrix.hnodes.getCouplingMatrix(level, leaf - level_start);
            int tree_index = hmatrix.hnodes.rank_leaf_tree_index[leaf];
            int u_index = hmatrix.hnodes.node_u_index[tree_index];
            int v_index = hmatrix.hnodes.node_v_index[tree_index];
            int u_1 = u_basis_tree.node_start[u_index], v_1 = v_basis_tree.node_start[v_index];
            int u_2 = u_1 + u_basis_tree.node_len[u_index] - 1, v_2 = v_1 + v_basis_tree.node_len[v_index] - 1;

            printf("\tNode %d Tree Index: %d : [%d, %d] x [%d, %d]\n\t\t", leaf, tree_index, u_1, u_2, v_1, v_2);
            printf("\n");
            for (int i = 0; i < rank; i++)
            {
                for (int j = 0; j < rank; j++)
                {
                    int local_index = i + j * rank;
                    printf(format, entries[local_index]);
                }
                printf("\n");
                if (i != rank - 1)
                    printf("\t\t");
            }
        }
    }
    printf("];\n");
}

void printMatrixTreeStructure(HMatrix &hmatrix)
{
    BasisTree &u_basis_tree = hmatrix.u_basis_tree;
    BasisTree &v_basis_tree = (hmatrix.sym ? u_basis_tree : hmatrix.v_basis_tree);

    printf("n = %d;\n", hmatrix.n);
    printf("P = [\n");
    for (size_t i = 0; i < u_basis_tree.index_map.size(); i++)
        printf("%d;\n", u_basis_tree.index_map[i]);
    printf("];\n");

    printf("UV = [\n");
    for (int level = 0; level < hmatrix.hnodes.depth; level++)
    {
        int level_start, level_end;
        hmatrix.hnodes.getCouplingLevelRange(level, level_start, level_end);

        for (int leaf = level_start; leaf < level_end; leaf++)
        {
            int tree_index = hmatrix.hnodes.rank_leaf_tree_index[leaf];
            int u_index = hmatrix.hnodes.node_u_index[tree_index];
            int v_index = hmatrix.hnodes.node_v_index[tree_index];
            int u_1 = u_basis_tree.node_start[u_index], v_1 = v_basis_tree.node_start[v_index];
            int u_2 = u_1 + u_basis_tree.node_len[u_index] - 1, v_2 = v_1 + v_basis_tree.node_len[v_index] - 1;

            printf("\t%d, %d, %d, %d; \n", u_1, u_2, v_1, v_2);
        }
    }
    printf("];\n");

    printf("Ranks=[\n");
    for (int level = 0; level < u_basis_tree.depth; level++)
        printf("%d\n", u_basis_tree.level_data.getLevelRank(level));
    printf("];\n");

    printf("Clevels = [\n");
    for (int level = 0; level < hmatrix.hnodes.depth; level++)
    {
        int level_start, level_end;
        hmatrix.hnodes.getCouplingLevelRange(level, level_start, level_end);
        printf("%d %d,\n", level_start, level_end);
    }
    printf("];\n");

    printf("DenseUV = [\n");
    int num_dense_leaves = hmatrix.hnodes.num_dense_leaves;
    for (int leaf = 0; leaf < num_dense_leaves; leaf++)
    {
        int tree_index = hmatrix.hnodes.dense_leaf_tree_index[leaf];
        int u_index = hmatrix.hnodes.node_u_index[tree_index];
        int v_index = hmatrix.hnodes.node_v_index[tree_index];

        int u_1 = u_basis_tree.node_start[u_index], v_1 = v_basis_tree.node_start[v_index];
        int u_2 = u_1 + u_basis_tree.node_len[u_index] - 1, v_2 = v_1 + v_basis_tree.node_len[v_index] - 1;

        printf("\t%d, %d, %d, %d; \n", u_1, u_2, v_1, v_2);
    }
    printf("];\n");
}

void dumpTikzMatrixIndices(HMatrix &hmatrix)
{
    BasisTree &v_basis_tree = (hmatrix.sym ? hmatrix.u_basis_tree : hmatrix.v_basis_tree);

    std::stringstream uarray, varray, dense_uarray, dense_varray;

    uarray << "\\def\\Uarray{{ ";
    varray << "\\def\\Varray{{ ";

    for (int level = hmatrix.hnodes.depth - 1; level >= 0; level--)
    {
        int level_start, level_end;
        hmatrix.hnodes.getCouplingLevelRange(level, level_start, level_end);
        int u_level_start = hmatrix.u_basis_tree.getLevelStart(level);
        int v_level_start = v_basis_tree.getLevelStart(level);

        uarray << "{";
        varray << "{";

        for (int leaf = level_start; leaf < level_end; leaf++)
        {
            int tree_index = hmatrix.hnodes.rank_leaf_tree_index[leaf];
            int u_index = hmatrix.hnodes.node_u_index[tree_index] - u_level_start;
            int v_index = hmatrix.hnodes.node_v_index[tree_index] - v_level_start;

            uarray << u_index;
            varray << v_index;
            if (leaf != level_end - 1)
            {
                uarray << ", ";
                varray << ", ";
            }
        }
        uarray << "}";
        varray << "}";
        if (level != 0)
        {
            uarray << ",";
            varray << ",";
        }
    }
    uarray << " }}\n";
    varray << " }}\n";

    dense_uarray << "\\def\\DenseU{{ ";
    dense_varray << "\\def\\DenseV{{ ";
    int num_dense_leaves = hmatrix.hnodes.num_dense_leaves;
    int u_level_start = hmatrix.u_basis_tree.getLevelStart(hmatrix.hnodes.depth - 1);
    int v_level_start = v_basis_tree.getLevelStart(hmatrix.hnodes.depth - 1);

    for (int leaf = 0; leaf < num_dense_leaves; leaf++)
    {
        int tree_index = hmatrix.hnodes.dense_leaf_tree_index[leaf];
        int u_index = hmatrix.hnodes.node_u_index[tree_index] - u_level_start;
        int v_index = hmatrix.hnodes.node_v_index[tree_index] - v_level_start;

        dense_uarray << u_index;
        dense_varray << v_index;
        if (leaf != num_dense_leaves - 1)
        {
            dense_uarray << ", ";
            dense_varray << ", ";
        }
    }
    dense_uarray << " }}\n";
    dense_varray << " }}\n";

    std::cout << uarray.str() << varray.str() << dense_uarray.str() << dense_varray.str();

    printf("\\def\\CNodes{{ ");
    for (int level = hmatrix.hnodes.depth - 1; level >= 0; level--)
    {
        printf("%d", hmatrix.hnodes.getCouplingLevelSize(level));
        if (level != 0)
            printf(", ");
    }
    printf("}}\n");

    printf("\\def\\DNodes{%d}\n", num_dense_leaves);
}

void dumpTransferLevel(BasisTree &basis_tree, int digits, int level)
{
    char format[64];
    sprintf(format, "%%.%de ", digits);

    int level_start, level_end;
    basis_tree.getLevelRange(level, level_start, level_end);

    // Get the rows and cols of the nodes in this level
    int rows, cols;
    basis_tree.getTransferDims(level, rows, cols);

    if (rows == 0 || cols == 0)
        return;

    for (int node = level_start; node < level_end; node++)
    {
        H2Opus_Real *utrans = basis_tree.getTransNode(level, node - level_start);
        printf("\tNode %d\n\t\t", node);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                printf(format, utrans[i + j * rows]);
            }
            printf("\n");
            if (i != rows - 1)
                printf("\t\t");
        }
    }
}

void dumpBasisTree(BasisTree &basis_tree, int digits, const char *label)
{
    char format[64];
    sprintf(format, "%%.%de ", digits);

    // Now the basis vectors
    printf("%s Trans [\n", label);
    for (int level = 0; level < basis_tree.depth; level++)
        dumpTransferLevel(basis_tree, digits, level);
    printf("]\n");

    printf("%s Basis [\n", label);
    int leaf_start = basis_tree.getLevelStart(basis_tree.depth - 1);
    int leaf_rank = basis_tree.getLevelRank(basis_tree.depth - 1);
    int ld = basis_tree.leaf_size;

    for (int leaf = 0; leaf < basis_tree.basis_leaves; leaf++)
    {
        int node_id = leaf + leaf_start;
        printf("\tNode %d: [%d; %d] \n\t\t", node_id, basis_tree.node_start[node_id],
               basis_tree.node_start[node_id] + basis_tree.node_len[node_id]);
        H2Opus_Real *u_leaf = basis_tree.getBasisLeaf(leaf);

        for (int i = 0; i < basis_tree.node_len[node_id]; i++)
        {
            for (int j = 0; j < leaf_rank; j++)
            {
                printf(format, u_leaf[i + j * ld]);
            }
            printf("\n");
            if (i != basis_tree.node_len[node_id] - 1)
                printf("\t\t");
        }
    }
    printf("]\n");
}

void dumpHMatrix(HMatrix &hmatrix, int digits)
{
    char format[64];
    sprintf(format, "%%.%de ", digits);

    BasisTree &u_basis_tree = hmatrix.u_basis_tree;
    BasisTree &v_basis_tree = (hmatrix.sym ? u_basis_tree : hmatrix.v_basis_tree);

    // First dump out the dense matrices
    printf("Dense Matrices [\n");
    for (int leaf = 0; leaf < hmatrix.hnodes.num_dense_leaves; leaf++)
    {
        int tree_index = hmatrix.hnodes.dense_leaf_tree_index[leaf];
        int u_index = hmatrix.hnodes.node_u_index[tree_index];
        int v_index = hmatrix.hnodes.node_v_index[tree_index];

        int u_1 = u_basis_tree.node_start[u_index], v_1 = v_basis_tree.node_start[v_index];
        int u_2 = u_1 + u_basis_tree.node_len[u_index] - 1, v_2 = v_1 + v_basis_tree.node_len[v_index] - 1;
        H2Opus_Real *m = hmatrix.hnodes.getDenseMatrix(leaf);

        int ld = hmatrix.u_basis_tree.leaf_size;

        printf("\tNode %d (%d): [%d, %d] x [%d, %d]\n\t\t", leaf, tree_index, u_1, u_2, v_1, v_2);
        for (int i = u_1; i <= u_2; i++)
        {
            for (int j = v_1; j <= v_2; j++)
            {
                int local_index = (i - u_1) + (j - v_1) * ld;
                printf(format, m[local_index]);
            }
            printf("\n");
            if (i != u_2)
                printf("\t\t");
        }
    }
    printf("];\n");

    dumpCouplingMatrices(hmatrix, digits);
    dumpBasisTree(u_basis_tree, digits, "UTree");
    if (!hmatrix.sym)
        dumpBasisTree(v_basis_tree, digits, "VTree");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Expand hmatrix to full dense matrix
////////////////////////////////////////////////////////////////////////////////////////////////////
void expandBasis(BasisTree &basis_tree, int node_index, H2Opus_Real *matrices, int depth, int n, int max_rank)
{
    int level_rank = basis_tree.getLevelRank(depth);
    int child = basis_tree.head[node_index];
    int leaf_size = basis_tree.leaf_size;

    if (child == H2OPUS_EMPTY_NODE)
    {
        H2Opus_Real *matrix_base = matrices + n * max_rank * depth;
        int leaf_start = basis_tree.getLevelStart(basis_tree.depth - 1);
        int leaf_index = node_index - leaf_start;
        H2Opus_Real *node = basis_tree.getBasisLeaf(leaf_index);

        for (int i = 0; i < basis_tree.node_len[node_index]; i++)
        {
            for (int j = 0; j < level_rank; j++)
            {
                int index = basis_tree.node_start[node_index] + i + j * n;
                matrix_base[index] = node[i + j * leaf_size];
            }
        }
    }
    else
    {
        while (child != H2OPUS_EMPTY_NODE)
        {
            expandBasis(basis_tree, child, matrices, depth + 1, n, max_rank);

            H2Opus_Real *matrix_base = matrices + n * max_rank * depth;
            H2Opus_Real *child_matrix_base = matrices + n * max_rank * (depth + 1);

            // Pointer to the sub-matrix of the parent that the child affects
            H2Opus_Real *parent_matrix = matrix_base + basis_tree.node_start[child];
            H2Opus_Real *child_matrix = child_matrix_base + basis_tree.node_start[child];

            int child_rank = basis_tree.getLevelRank(depth + 1);
            int child_start = basis_tree.getLevelStart(depth + 1);

            if (level_rank != 0 && child_rank != 0)
            {
                // Do the transformation
                int child_len = basis_tree.node_len[child];
                int child_index = child - child_start;

                H2Opus_Real *trans = basis_tree.getTransNode(depth + 1, child_index);

                cblas_gemm(CblasNoTrans, CblasNoTrans, child_len, level_rank, child_rank, 1, child_matrix, n, trans,
                           child_rank, 0, parent_matrix, n);
            }

            child = basis_tree.next[child];
        }
    }
}

void expandHmatrix(HMatrix &hmatrix, H2Opus_Real *matrix, int dense, int coupling_level)
{
    // Expand the dense leaves first
    int n = hmatrix.n;
    int num_dense_leaves = hmatrix.hnodes.num_dense_leaves;
    BasisTree &u_basis_tree = hmatrix.u_basis_tree;
    BasisTree &v_basis_tree = (hmatrix.sym ? u_basis_tree : hmatrix.v_basis_tree);

    if (dense)
    {
        for (int leaf = 0; leaf < num_dense_leaves; leaf++)
        {
            int tree_index = hmatrix.hnodes.dense_leaf_tree_index[leaf];
            int u_index = hmatrix.hnodes.node_u_index[tree_index];
            int v_index = hmatrix.hnodes.node_v_index[tree_index];

            int u_1 = u_basis_tree.node_start[u_index], v_1 = v_basis_tree.node_start[v_index];
            int u_2 = u_1 + u_basis_tree.node_len[u_index] - 1, v_2 = v_1 + v_basis_tree.node_len[v_index] - 1;
            H2Opus_Real *m = hmatrix.hnodes.getDenseMatrix(leaf);

            int ld = hmatrix.u_basis_tree.leaf_size;

            for (size_t i = u_1; i <= (size_t)u_2; i++)
            {
                for (size_t j = v_1; j <= (size_t)v_2; j++)
                {
                    int local_index = (i - u_1) + (j - v_1) * ld;
                    size_t dense_index = i + j * n;
                    matrix[dense_index] = m[local_index];
                }
            }
        }
    }

    // Now expand the basis trees
    int max_rank = hmatrix.u_basis_tree.level_data.getLargestRank();

    H2Opus_Real *u_matrices = (H2Opus_Real *)malloc(sizeof(H2Opus_Real) * hmatrix.n * max_rank * u_basis_tree.depth);
    H2Opus_Real *v_matrices = (H2Opus_Real *)malloc(sizeof(H2Opus_Real) * hmatrix.n * max_rank * v_basis_tree.depth);

    assert(u_matrices && v_matrices);

    expandBasis(u_basis_tree, 0, u_matrices, 0, hmatrix.n, max_rank);
    expandBasis(v_basis_tree, 0, v_matrices, 0, hmatrix.n, max_rank);

    for (int level = 0; level < hmatrix.hnodes.depth; level++)
    {
        int level_start, level_end;
        if (coupling_level != level && coupling_level != -1)
            continue;

        hmatrix.hnodes.getCouplingLevelRange(level, level_start, level_end);

        for (int node_index = level_start; node_index < level_end; node_index++)
        {
            int tree_index = hmatrix.hnodes.rank_leaf_tree_index[node_index];
            int u_index = hmatrix.hnodes.node_u_index[tree_index];
            int v_index = hmatrix.hnodes.node_v_index[tree_index];

            int u_1 = u_basis_tree.node_start[u_index], v_1 = v_basis_tree.node_start[v_index];
            // int u_2 = u_1 + u_basis_tree.node_len[u_index] - 1, v_2 = v_1 + v_basis_tree.node_len[v_index] - 1;

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
                    size_t dense_index = u_1 + j + (size_t)(v_1 + k) * n;
                    matrix[dense_index] = 0;
                    for (int l = 0; l < v_dim; l++)
                        matrix[dense_index] += u_matrix[j + l * hmatrix.n] * sv_matrix[l + k * v_dim];
                }
            }

            free(sv_matrix);
        }
    }

    free(u_matrices);
    free(v_matrices);
}

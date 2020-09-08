#ifndef __H2OPUS_BOX_ENTRY_GEN_H__
#define __H2OPUS_BOX_ENTRY_GEN_H__

#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/h2opus_eps.h>
#include <h2opus/util/kdtree.h>

template <class T, int hw, typename FunctionGen> struct BoxEntryGen
{
    FunctionGen &gen;

    inline void chebyshev_pts(TH2OpusKDTree<T, hw> &kdtree, int node_index, std::vector<T> &cheb_pts, int dim,
                              int slices)
    {
        for (int d = 0; d < dim; d++)
        {
            T a, b;
            kdtree.getBoundingBoxComponent(node_index, d, a, b);
            for (int i = 0; i < slices; i++)
            {
                T t = (T)(2 * i + 1) / (2 * slices);
                cheb_pts[i + d * slices] = 0.5 * (a + b) + 0.5 * (b - a) * cos(t * M_PI);
            }
        }
    }

    inline void lagrange_weights(std::vector<T> &pts, std::vector<T> &weights, int dim, int slices)
    {
        for (int d = 0; d < dim; d++)
            for (int i = 0; i < slices; i++)
                for (int j = 0; j < slices; j++)
                    if (i != j)
                        weights[i + d * slices] /= (pts[i + d * slices] - pts[j + d * slices]);
    }

    inline T lagrange_func(T x0, std::vector<T> &pts, int slices, int d)
    {
        T f = 1;
        T *x = &pts[d * slices];

        for (int i = 0; i < slices; i++)
            f *= (x0 - x[i]);

        return f;
    }

    void u_basis_leaf(T *leaf, int ld, TH2OpusKDTree<T, hw> &kdtree, int node_index, int slices)
    {
        basis_leaf(leaf, ld, kdtree, node_index, slices);
    }

    void v_basis_leaf(T *leaf, int ld, TH2OpusKDTree<T, hw> &kdtree, int node_index, int slices)
    {
        basis_leaf(leaf, ld, kdtree, node_index, slices);
    }

    void u_transfer_matrix(T *transfer, int ld, TH2OpusKDTree<T, hw> &kdtree, int cluster_index,
                           int parent_cluster_index, int slices, int parent_slices)
    {
        transfer_matrix(transfer, ld, kdtree, cluster_index, parent_cluster_index, slices, parent_slices);
    }

    void v_transfer_matrix(T *transfer, int ld, TH2OpusKDTree<T, hw> &kdtree, int cluster_index,
                           int parent_cluster_index, int slices, int parent_slices)
    {
        transfer_matrix(transfer, ld, kdtree, cluster_index, parent_cluster_index, slices, parent_slices);
    }

    void basis_leaf(T *leaf, int ld, TH2OpusKDTree<T, hw> &kdtree, int node_index, int slices)
    {
        int start, end;
        kdtree.getNodeLimits(node_index, start, end);

        H2OpusDataSet<T> *data_set = kdtree.getDataSet();
        int *index_map = kdtree.getIndexMap() + start;
        int num_pts = end - start;
        int dim = kdtree.getDim();
        int rank = pow(slices, dim);

        std::vector<T> weights(slices * dim, 1), cheb_pts(slices * dim), lx(num_pts * dim);

        chebyshev_pts(kdtree, node_index, cheb_pts, dim, slices);
        lagrange_weights(cheb_pts, weights, dim, slices);

        for (int i = 0; i < num_pts; i++)
        {
            int pt_index = index_map[i];
            for (int d = 0; d < dim; d++)
                lx[i + d * num_pts] = lagrange_func(data_set->getDataPoint(pt_index, d), cheb_pts, slices, d);
        }

        for (int i = 0; i < num_pts; i++)
        {
            for (int j = 0; j < rank; j++)
            {
                int slice_index = j;
                T entry = 1;
                for (int d = 0; d < dim; d++)
                {
                    int index = slice_index % slices;

                    T x0 = data_set->getDataPoint(index_map[i], d);
                    T lx0 = lx[i + d * num_pts];
                    T w = weights[index + d * slices];
                    T x = cheb_pts[index + d * slices];

                    if (fabs(x0 - x) > H2OpusEpsilon<T>::eps)
                        entry *= lx0 * w / (x0 - x);

                    slice_index /= slices;
                }

                leaf[i + j * ld] = (T)entry;
            }
        }
    }

    void transfer_matrix(T *transfer, int ld, TH2OpusKDTree<T, hw> &kdtree, int cluster_index, int parent_cluster_index,
                         int slices, int parent_slices)
    {
        int dim = kdtree.getDim();
        int parent_rank = pow(parent_slices, dim), rank = pow(slices, dim);
        std::vector<T> parent_weights(parent_slices * dim, 1), parent_cheb_pts(parent_slices * dim);
        std::vector<T> lx(slices * dim), cheb_pts(slices * dim);

        chebyshev_pts(kdtree, parent_cluster_index, parent_cheb_pts, dim, parent_slices);
        chebyshev_pts(kdtree, cluster_index, cheb_pts, dim, slices);
        lagrange_weights(parent_cheb_pts, parent_weights, dim, parent_slices);

        for (int i = 0; i < slices; i++)
            for (int d = 0; d < dim; d++)
                lx[i + d * slices] = lagrange_func(cheb_pts[i + d * slices], parent_cheb_pts, parent_slices, d);

        for (int i = 0; i < rank; i++)
        {
            for (int j = 0; j < parent_rank; j++)
            {
                int slice_index = i;
                int parent_slice_index = j;
                T entry = 1;
                for (int d = 0; d < dim; d++)
                {
                    int index = slice_index % slices;
                    int parent_index = parent_slice_index % parent_slices;

                    T x0 = cheb_pts[index + d * slices];
                    T lx0 = lx[index + d * slices];
                    T w_parent = parent_weights[parent_index + d * parent_slices];
                    T x_parent = parent_cheb_pts[parent_index + d * parent_slices];

                    if (fabs(x0 - x_parent) > H2OpusEpsilon<T>::eps)
                        entry *= lx0 * w_parent / (x0 - x_parent);

                    parent_slice_index /= parent_slices;
                    slice_index /= slices;
                }

                transfer[i + j * ld] = (T)entry;
            }
        }
    }

    void coupling_matrix(T *coupling, int ld, TH2OpusKDTree<T, hw> &u_kdtree, int u_cluster_index,
                         TH2OpusKDTree<T, hw> &v_kdtree, int v_cluster_index, int slices)
    {
        int dim = u_kdtree.getDim();
        int rank = pow(slices, dim);

        std::vector<T> u_cheb_pts(slices * dim), v_cheb_pts(slices * dim);

        std::vector<T> u_point(dim), v_point(dim);
        chebyshev_pts(u_kdtree, u_cluster_index, u_cheb_pts, dim, slices);
        chebyshev_pts(v_kdtree, v_cluster_index, v_cheb_pts, dim, slices);

        for (int i = 0; i < rank; i++)
        {
            int u_slice_index = i;
            for (int d = 0; d < dim; d++)
            {
                int ui = u_slice_index % slices;
                u_point[d] = (T)u_cheb_pts[ui + d * slices];
                u_slice_index /= slices;
            }

            for (int j = 0; j < rank; j++)
            {
                int v_slice_index = j;
                for (int d = 0; d < dim; d++)
                {
                    int vi = v_slice_index % slices;
                    v_point[d] = (T)v_cheb_pts[vi + d * slices];
                    v_slice_index /= slices;
                }

                coupling[i + j * ld] = gen(&u_point[0], &v_point[0]);
            }
        }
    }

    void dense_matrix(T *m, int ld, TH2OpusKDTree<T, hw> &u_kdtree, int u_cluster_index, TH2OpusKDTree<T, hw> &v_kdtree,
                      int v_cluster_index)
    {
        int dim = u_kdtree.getDim();
        H2OpusDataSet<T> *u_data_set = u_kdtree.getDataSet();
        H2OpusDataSet<T> *v_data_set = v_kdtree.getDataSet();

        int u_start, u_end, v_start, v_end;
        u_kdtree.getNodeLimits(u_cluster_index, u_start, u_end);
        v_kdtree.getNodeLimits(v_cluster_index, v_start, v_end);

        int rows = u_end - u_start, cols = v_end - v_start;
        int *u_index_map = u_kdtree.getIndexMap() + u_start;
        int *v_index_map = v_kdtree.getIndexMap() + v_start;

        std::vector<T> pt_i(dim), pt_j(dim);
        for (int i = 0; i < rows; i++)
        {
            int i_index = u_index_map[i];
            for (int d = 0; d < dim; d++)
                pt_i[d] = u_data_set->getDataPoint(i_index, d);

            for (int j = 0; j < cols; j++)
            {
                int j_index = v_index_map[j];
                for (int d = 0; d < dim; d++)
                    pt_j[d] = v_data_set->getDataPoint(j_index, d);
                m[i + j * ld] = gen(&pt_i[0], &pt_j[0]);
            }
        }
    }

    BoxEntryGen(FunctionGen &f_gen) : gen(f_gen)
    {
    }
};

#endif

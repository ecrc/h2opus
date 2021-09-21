#ifndef __FD2D_CORE_H
#define __FD2D_CORE_H

// Used to compute the diagonal term of the inner grid with a O(n) algorithm
// See test_fd_petsc.cpp
template <typename T> class OuterGrid
{
  public:
    T h, left, delta, a;
    int dimension;
    size_t outer_size, inner_size;
    size_t offset;
    size_t num_points;

    OuterGrid(int dim, size_t n, T a, T b, int nimages)
    {
        assert(dim == 2); // Not coded for dim == 3 yet
        this->dimension = dim;
        this->inner_size = n;
        this->outer_size = (2 * nimages + 1) * (n + 1) + 1;
        this->offset = (n + 1) + 1; // 0-index in inner grid is offset-index in outer grid
        this->num_points = this->outer_size * this->outer_size;
        this->h = (b - a) / (n + 1);
        this->a = a;
        this->delta = std::min(n * this->h, 10 * this->h);
        this->left = (a)-nimages * (b - a);
    }

    inline void get_point(int i, int j, T pt[])
    {
        pt[0] = this->left + i * this->h;
        pt[1] = this->left + j * this->h;
    }

    int getDimension() const
    {
        return dimension;
    }

    size_t getDataSetSize() const
    {
        return num_points;
    }

    T getDataPoint(size_t idx, int dim) const
    {
        T v;
        size_t i = idx % outer_size;
        size_t j = idx / outer_size;
        v = (dim ? this->left + j * this->h : this->left + i * this->h);
        return v;
    }

    std::vector<T> getCoordsV()
    {
        std::vector<T> pts;
        pts.resize(dimension * outer_size * outer_size);
        for (size_t j = 0; j < outer_size * outer_size; j++)
        {
            pts[2 * j] = getDataPoint(j, 0);
            pts[2 * j + 1] = getDataPoint(j, 1);
        }

        return pts;
    }

    template <typename IDX> void get_indices_interior(std::vector<IDX> &indices)
    {
        IDX nx = outer_size;
        for (IDX gj = offset; gj < (IDX)(offset + inner_size); gj++)
            for (IDX gi = offset; gi < (IDX)(offset + inner_size); gi++)
            {
                IDX idx = gj * nx + gi;
                indices.push_back(idx);
            }
        return;
    }
};

template <typename T> class PointCloud : public H2OpusDataSet<T>
{
  public:
    int dimension;
    size_t num_points;
    double h, a, b, l;
    int n, n2;
    OuterGrid<T> *ogrid;

    PointCloud()
    {
        this->dimension = 0;
        this->num_points = 0;
        this->h = 0;
        this->a = 0;
        this->b = 0;
        this->l = 0;
        this->n = 0;
        this->n2 = 0;
        this->ogrid = NULL;
    }

    void generateGrid(int dim, int grid_x, T min_x, T max_x)
    {
        assert(dim == 2 || dim == 3);
        T hx = (max_x - min_x) / (grid_x - 1);
        this->dimension = dim;
        this->num_points = pow(grid_x, dim);
        this->h = hx;
        this->a = min_x - hx;
        this->b = max_x + hx;
        this->l = min_x;
        this->n = grid_x;
        this->n2 = grid_x * grid_x;
        if (dim == 2)
        {
            this->ogrid = new OuterGrid<T>(2, n, a, b, 1);
        }
    }

    ~PointCloud()
    {
        delete ogrid;
    }

    int getDimension() const
    {
        return dimension;
    }

    size_t getDataSetSize() const
    {
        return num_points;
    }

    T getDataPoint(size_t idx, int dim) const
    {
        T r;
        if (dimension == 3)
        {
            size_t k = idx / n2;
            idx = idx % n2;
            size_t j = idx / n;
            size_t i = idx % n;
            switch (dim)
            {
            case 2:
                r = l + k * h;
                break;
            case 1:
                r = l + j * h;
                break;
            default:
                r = l + i * h;
                break;
            }
        }
        else
        {
            size_t j = idx / n;
            size_t i = idx % n;
            switch (dim)
            {
            case 1:
                r = l + j * h;
                break;
            default:
                r = l + i * h;
                break;
            }
        }
        return r;
    }

    std::vector<T> getCoordsV()
    {
        std::vector<T> pts;
        pts.resize(dimension * num_points);
        for (size_t j = 0; j < num_points; j++)
            for (int d = 0; d < dimension; d++)
                pts[dimension * j + d] = getDataPoint(j, d);
        return pts;
    }
};

template <typename T> __host__ __device__ inline T distance(const T pt_x[], const T pt_y[], int dim)
{
    T diff = 0;
    for (int d = 0; d < dim; d++)
        diff += (pt_x[d] - pt_y[d]) * (pt_x[d] - pt_y[d]);
    return sqrt(diff);
}

template <typename T> __host__ __device__ inline T beta(const T pt[])
{
    return 0.7;
}

template <typename T> __host__ __device__ inline T bump1d(T x, T center, T support)
{
    T y = (x - center) / (support / 2.0);
    return (abs(y) >= 1) ? 0.0 : exp(-1.0 / (1.0 - y * y)) * M_E;
}

template <typename T> __host__ __device__ inline T kappa(T pt[], int dim)
{
    return dim == 3 ? 1.0 + 1.0 * bump1d(pt[0], (T)0.0, (T)2.0) * bump1d(pt[1], (T)0.0, (T)2.0) *
                                bump1d(pt[2], (T)0.0, (T)2.0)
                    : 1.0 + 1.0 * bump1d(pt[0], (T)0.0, (T)2.0) * bump1d(pt[1], (T)0.0, (T)2.0);
}

template <typename T> __host__ __device__ inline T W(T x, T delta)
{
    T r = abs(x / delta);
    if (r >= 1)
        return 0;
    else
    {
        double f = 1 + pow(r, 4) * (-35 + 84 * r - 70 * r * r + 20 * r * r * r);
        return f;
    }
}

template <typename T> class FDGen
{
  public:
    int dim;
    T hs;

    FDGen(int dim, T h)
    {
        this->dim = dim;
        this->hs = pow(h, dim);
    }

    inline T kernel(T *pt_x, T *pt_y)
    {
        return hs * sqrt(kappa(pt_x, dim) * kappa(pt_y, dim)) /
               pow(distance(pt_x, pt_y, dim), dim + beta(pt_x) + beta(pt_y));
    }

    T operator()(T *pt_x, T *pt_y)
    {
        T dist = distance(pt_x, pt_y, dim);
        if (dist > 0)
        {
            return -kernel(pt_x, pt_y);
        }
        else // diagonal entry
        {
            return 0;
        }
    }

    // n^2 algorithm to compute the diagonal term
    T compute_diagonal(PointCloud<T> *pt_cloud, T *pt_x)
    {
        if (dim == 3 || !pt_cloud->ogrid)
            return 0;
        T v = 0;
        const T h = pt_cloud->h / 10;
        size_t n_outer = pt_cloud->ogrid->outer_size;
        for (size_t ki = 0; ki < n_outer; ki++)
        {
            for (size_t kj = 0; kj < n_outer; kj++)
            {
                T pt_j[2];
                pt_cloud->ogrid->get_point(ki, kj, pt_j);
                T d2 = distance(pt_x, pt_j, dim);
                if (d2 > h)
                {
                    v += kernel(pt_x, pt_j);
                }
            }
        }
        return v;
    }
};

// For debugging purposes
template <class T> class MatGen
{
  public:
    FDGen<T> gen;
    PointCloud<T> &cloud;
    int *index_map;
    int dim;

    MatGen(FDGen<T> &func_gen, PointCloud<T> &pt_cloud, int *map) : gen(func_gen), cloud(pt_cloud)
    {
        this->index_map = map;
        this->dim = pt_cloud.getDimension();
    }

    H2Opus_Real generateEntry(int i, int j)
    {
        int index_i = index_map[i], index_j = index_map[j];
        std::vector<H2Opus_Real> pt_i(dim), pt_j(dim);
        for (int d = 0; d < dim; d++)
        {
            pt_i[d] = cloud.getDataPoint(index_i, d);
            pt_j[d] = cloud.getDataPoint(index_j, d);
        }

        return gen(&pt_i[0], &pt_j[0]);
    }
};

// methods to construct S

#include "cubature.h"

int f2d_polar(unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval)
{
    double *xi = (double *)fdata;
    double delta = ((double *)fdata)[2];
    double r = x[0];

    fval[0] = 0.5 * r * W(r, delta) * pow(r, 2.0) / pow(r, 2 + 2 * beta(xi));
    if (fval[0] == INFINITY)
        fval[0] = 0;
    return 0;
}

template <typename T> class SMat
{
  private:
    FDGen<T> *gen;
    PointCloud<T> *pt_cloud;
    int dim;
    size_t n;
    size_t offset;
    T h, delta;

  public:
    SMat(FDGen<T> *func_gen, PointCloud<T> *pt_cloud_) : gen(func_gen), pt_cloud(pt_cloud_)
    {
        this->dim = gen->dim;
        assert(dim == pt_cloud->getDimension());
        this->n = pt_cloud->n;
        assert(pt_cloud->ogrid);
        this->offset = pt_cloud->ogrid->offset;
        this->delta = pt_cloud->ogrid->delta;
        this->h = pt_cloud->h;
    }

    inline T C1(size_t idx)
    {
        // Not coded for dim == 3 yet
        if (dim == 3)
            return 0;
        T C1i = 0.0;
        T xi[3], xk[3];

        xi[0] = pt_cloud->getDataPoint(idx, 0);
        xi[1] = pt_cloud->getDataPoint(idx, 1);
        size_t i = idx % n;
        size_t j = idx / n;
        size_t i0 = i + offset;
        size_t j0 = j + offset;
        size_t m = ceil(delta / h) - 1;

        for (size_t ki = i0 - m; ki <= i0 + m; ki++)
        {
            for (size_t kj = j0 - m; kj <= j0 + m; kj++)
            {
                if ((ki != i0) || (kj != j0))
                {
                    pt_cloud->ogrid->get_point(ki, kj, xk);
                    T rk = distance(xk, xi, dim);
                    T wk = W(rk, delta);
                    C1i += 0.5 * wk * (0.5 * pow(rk, 2)) / pow(rk, 2 + 2 * beta(xi));
                }
            }
        }
        return C1i;
    }

    inline T C2(size_t idx)
    {
        // Not coded for dim == 3 yet
        if (dim == 3)
            return 0;
        T C2i = 0.0;
        double fdata[3];
        fdata[0] = pt_cloud->getDataPoint(idx, 0);
        fdata[1] = pt_cloud->getDataPoint(idx, 1);
        fdata[2] = delta;
        double val, err;
        double xmin[1] = {0}, xmax[1] = {delta};

        hcubature(1, f2d_polar, fdata, 1, xmin, xmax, 1e6, 0, 1e-4, ERROR_INDIVIDUAL, &val, &err);
        C2i = -0.5 / (h * h) * (2 * M_PI * val);
        return C2i;
    }

    inline void generate_row(int idx, int &nz, int idxs[], T vals[])
    {
        nz = 0;
        // Not coded for dim == 3 yet
        if (dim == 3)
            return;
        T pt_i[2], pt_j[2];
        pt_i[0] = pt_cloud->getDataPoint(idx, 0);
        pt_i[1] = pt_cloud->getDataPoint(idx, 1);
        T east[2] = {pt_i[0] + h, pt_i[1]};
        T west[2] = {pt_i[0] - h, pt_i[1]};
        T north[2] = {pt_i[0], pt_i[1] + h};
        T south[2] = {pt_i[0], pt_i[1] - h};
        size_t i = idx % n;
        size_t j = idx / n;
        T c1c2 = C1(idx) + C2(idx);
        idxs[nz] = idx;
        vals[nz] = -sqrt(kappa(pt_i, 2)) *
                   (sqrt(kappa(east, 2)) + sqrt(kappa(west, 2)) + sqrt(kappa(north, 2)) + sqrt(kappa(south, 2))) * c1c2;
        nz++;
        if (i != 0)
        { // if west neighbor exists
            size_t J = (i - 1) + n * j;
            pt_j[0] = pt_cloud->getDataPoint(J, 0);
            pt_j[1] = pt_cloud->getDataPoint(J, 1);
            idxs[nz] = J;
            vals[nz] = sqrt(kappa(pt_i, 2)) * sqrt(kappa(pt_j, 2)) * c1c2;
            nz++;
        }
        if (j != 0)
        { // if south neighbor exists
            size_t J = i + n * (j - 1);
            pt_j[0] = pt_cloud->getDataPoint(J, 0);
            pt_j[1] = pt_cloud->getDataPoint(J, 1);
            idxs[nz] = J;
            vals[nz] = sqrt(kappa(pt_i, 2)) * sqrt(kappa(pt_j, 2)) * c1c2;
            nz++;
        }
        if (i + 1 != n)
        { // if east neighbor exists
            size_t J = (i + 1) + n * j;
            pt_j[0] = pt_cloud->getDataPoint(J, 0);
            pt_j[1] = pt_cloud->getDataPoint(J, 1);
            idxs[nz] = J;
            vals[nz] = sqrt(kappa(pt_i, 2)) * sqrt(kappa(pt_j, 2)) * c1c2;
            nz++;
        }
        if (j + 1 != n)
        { // if north neighbor exists
            size_t J = i + n * (j + 1);
            pt_j[0] = pt_cloud->getDataPoint(J, 0);
            pt_j[1] = pt_cloud->getDataPoint(J, 1);
            idxs[nz] = J;
            vals[nz] = sqrt(kappa(pt_i, 2)) * sqrt(kappa(pt_j, 2)) * c1c2;
            nz++;
        }
    }
};

#endif

#ifndef __EXAMPLE_PROBLEM_H__
#define __EXAMPLE_PROBLEM_H__

#include <thrust/random.h>
#include <h2opus.h>

#define DEFAULT_ETA 1.0

// This is an example of a custom data set class
template <class T> class PointCloud : public H2OpusDataSet<T>
{
  public:
    int dimension;
    size_t num_points;
    std::vector<std::vector<T>> pts;

    PointCloud(int dim, size_t num_pts)
    {
        this->dimension = dim;
        this->num_points = num_pts;

        pts.resize(dim);
        for (int i = 0; i < dim; i++)
            pts[i].resize(num_points);
    }

    size_t getDimension() const
    {
        return dimension;
    }

    size_t getDataSetSize() const
    {
        return num_points;
    }

    T getDataPoint(size_t idx, size_t dim) const
    {
        assert(dim < dimension && idx < num_points);
        return pts[dim][idx];
    }
};

template <class T> T getCorrelationLength(int dim)
{
    if (dim == 2)
        return 0.1;
    else
        return 0.2;
}

// This is an example of a functor that evaluates the kernel
template <class T> class FunctionGen
{
  private:
    int dim;

  public:
    FunctionGen(int dim)
    {
        this->dim = dim;
    }

    T operator()(T *pt_x, T *pt_y)
    {
        T diff = 0;
        for (int d = 0; d < dim; d++)
            diff += (pt_x[d] - pt_y[d]) * (pt_x[d] - pt_y[d]);
        T dist = sqrt(diff);
        return exp(-dist / getCorrelationLength<T>(dim));
    }
};

// Simple class to generate matrix entries in the h-ordering
template <class T> class MatGen
{
  public:
    FunctionGen<T> gen;
    PointCloud<T> cloud;
    int *index_map;
    int dim;

    MatGen(FunctionGen<T> &func_gen, PointCloud<T> &pt_cloud, int *map) : gen(func_gen), cloud(pt_cloud)
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

// Generate 1, 2, and 3D point grids
template <typename T> void generate1DGrid(PointCloud<T> &pt_cloud, int grid_x, T min_x, T max_x)
{
    T hx = (max_x - min_x) / (grid_x - 1);
    for (int i = 0; i < grid_x; i++)
        pt_cloud.pts[0][i] = min_x + i * hx;
}

template <typename T>
void generate2DGrid(PointCloud<T> &pt_cloud, int grid_x, int grid_y, T min_x, T max_x, T min_y, T max_y)
{
    T hx = (max_x - min_x) / (grid_x - 1);
    T hy = (max_y - min_y) / (grid_y - 1);

    for (size_t i = 0; i < (size_t)grid_x; i++)
    {
        for (size_t j = 0; j < (size_t)grid_y; j++)
        {
            pt_cloud.pts[0][i + j * grid_x] = min_x + i * hx;
            pt_cloud.pts[1][i + j * grid_x] = min_y + j * hy;
        }
    }
}

template <typename T>
void generate3DGrid(PointCloud<T> &pt_cloud, int grid_x, int grid_y, int grid_z, T min_x, T max_x, T min_y, T max_y,
                    T min_z, T max_z)
{
    T hx = (max_x - min_x) / (grid_x - 1);
    T hy = (max_y - min_y) / (grid_y - 1);
    T hz = (max_z - min_z) / (grid_z - 1);

    size_t pt_index = 0;

    for (size_t i = 0; i < (size_t)grid_x; i++)
    {
        for (size_t j = 0; j < (size_t)grid_y; j++)
        {
            for (size_t k = 0; k < (size_t)grid_z; k++)
            {
                pt_cloud.pts[0][pt_index] = min_x + i * hx;
                pt_cloud.pts[1][pt_index] = min_y + j * hy;
                pt_cloud.pts[2][pt_index] = min_z + k * hz;

                pt_index++;
            }
        }
    }
}

#endif

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

    PointCloud()
    {
        this->dimension = 0;
        this->num_points = 0;
    }

    PointCloud(int dim, size_t num_pts)
    {
        this->dimension = dim;
        this->num_points = num_pts;

        pts.resize(dim);
        for (int i = 0; i < dim; i++)
            pts[i].resize(num_points);
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
        assert(dim < dimension && idx < num_points);
        return pts[dim][idx];
    }

    std::vector<T> getCoords()
    {
        std::vector<T> outpts;
        outpts.resize(dimension * num_points);
        for (size_t j = 0; j < num_points; j++)
        {
            for (int d = 0; d < dimension; d++)
                outpts[dimension * j + d] = getDataPoint(j, d);
        }

        return outpts;
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
    T l;

  public:
    FunctionGen(int dim)
    {
        this->dim = dim;
        this->l = getCorrelationLength<T>(dim);
    }

    T operator()(T *pt_x, T *pt_y)
    {
        T diff = 0;
        for (int d = 0; d < dim; d++)
            diff += (pt_x[d] - pt_y[d]) * (pt_x[d] - pt_y[d]);
        T dist = sqrt(diff);
        return exp(-dist / l);
    }
};

// This is an example of a functor that evaluates the kernel
template <class T> class DiagGen
{
  private:
    int dim;

  public:
    DiagGen(int dim)
    {
        this->dim = dim;
    }

    T operator()(T *pt_x, T *pt_y)
    {
        T diff = 0;
        for (int d = 0; d < dim; d++)
            diff += (pt_x[d] - pt_y[d]) * (pt_x[d] - pt_y[d]);
        return diff > 0.0 ? 0.0 : pt_x[0];
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

template <typename T> void generateRandomSphereSurface(PointCloud<T> &pt_cloud, int num_pts, T r, unsigned int seed)
{
    thrust::minstd_rand seed_rng(seed);
    thrust::uniform_int_distribution<unsigned int> seed_dist;

    unsigned int seed_z = seed_dist(seed_rng);
    unsigned int seed_phi = seed_dist(seed_rng);

    thrust::minstd_rand z_rng(seed_z), phi_rng(seed_phi);
    thrust::uniform_real_distribution<T> z_dist(-r, r), phi_dist(0, 2 * M_PI);

    for (int i = 0; i < num_pts; i++)
    {
        T z = z_dist(z_rng), phi = phi_dist(phi_rng);
        T theta = asin(z / r);
        T x = r * cos(theta) * cos(phi);
        T y = r * cos(theta) * sin(phi);

        pt_cloud.pts[0][i] = x;
        pt_cloud.pts[1][i] = y;
        pt_cloud.pts[2][i] = z;
    }
}

template <typename T> void generateRandomSphere(PointCloud<T> &pt_cloud, int num_pts, T r, unsigned int seed)
{
    thrust::minstd_rand seed_rng(seed);
    thrust::uniform_int_distribution<unsigned int> seed_dist;

    unsigned int seed_v = seed_dist(seed_rng);
    unsigned int seed_theta = seed_dist(seed_rng);
    unsigned int seed_r = seed_dist(seed_rng);

    thrust::minstd_rand v_rng(seed_v), theta_rng(seed_theta), r_rng(seed_r);
    thrust::uniform_real_distribution<T> v_dist(0, 1), theta_dist(0, 2 * M_PI), r_dist(0, 1);

    for (int i = 0; i < num_pts; i++)
    {
        T v = v_dist(v_rng), theta = theta_dist(theta_rng), r2 = r_dist(r_rng);
        T phi = acos(2 * v - 1);
        r2 = r * pow(r2, (T)1.0 / 3);

        T x = r2 * sin(phi) * cos(theta);
        T y = r2 * sin(phi) * sin(theta);
        T z = r2 * cos(phi);

        pt_cloud.pts[0][i] = x;
        pt_cloud.pts[1][i] = y;
        pt_cloud.pts[2][i] = z;
    }
}

#endif

#ifndef __SPATIAL_STATISTICS_H__
#define __SPATIAL_STATISTICS_H__

#include <cmath>
#include <gsl/gsl_sf_bessel.h>

extern "C" double gsl_sf_gamma(const double);
extern "C" double gsl_sf_hyperg_1F1(double, double, double);

template <typename T> struct Spatial_Statistics
{
  private:
    T phi, nu;
    T con;
    int d;

  public:
    Spatial_Statistics(T phi, T nu, int dim)
    {
        this->phi = phi;
        this->nu = nu;
        // this->alpha = alpha;
        this->d = dim;

        con = pow(2, (nu - 1)) * tgamma(nu);
        con = 1.0 / con;

        // c1 = pow(h / sigma, d) / pow(sigma, alpha);
        // c2 = -pow(2.0, alpha) * gsl_sf_gamma((alpha + (T)d) / 2) / pow(M_PI, (T)d / 2) / gsl_sf_gamma((T)d / 2);
    }

    // T G(T x) const
    //{
    //    return c2 * gsl_sf_hyperg_1F1((alpha + (T)d) / 2, (T)d / 2, -x * x);
    //}

    T operator()(T *pt_x, T *pt_y) const
    {
        T diff = 0.0;
        for (int i = 0; i < d; i++)
            diff += (pt_x[i] - pt_y[i]) * (pt_x[i] - pt_y[i]);
        T dist = sqrt(diff);

        dist = 4.0 * sqrt(2.0 * nu) * (dist / phi);
        if (dist == 0.0)
        {
            return 1.0;
        }
        else
        {
            return con * pow(dist, nu) * gsl_sf_bessel_Knu(nu, dist);
        }
    }
};

template <typename T> struct MaternKernel
{
  private:
    T h, sigma, alpha;
    T c1, c2;
    int d;

  public:
    MaternKernel(T h, T sigma, T alpha, int dim)
    {
        this->h = h;
        this->sigma = sigma;
        this->alpha = alpha;
        this->d = dim;

        c1 = pow(h / sigma, d) / pow(sigma, alpha);
        c2 = -pow(2.0, alpha) * gsl_sf_gamma((alpha + (T)d) / 2.0) / pow(M_PI, (T)d / 2.0) / gsl_sf_gamma((T)d / 2.0);
    }

    T G(T x) const
    {
        return c2 * gsl_sf_hyperg_1F1((alpha + (T)d) / 2.0, (T)d / 2.0, -x * x);
    }

    T operator()(T *pt_x, T *pt_y) const
    {
        T diff = 0.0;
        for (int i = 0; i < d; i++)
            diff += (pt_x[i] - pt_y[i]) * (pt_x[i] - pt_y[i]);
        T dist = sqrt(diff);

        return c1 * G(dist / sigma);
    }
};

#endif

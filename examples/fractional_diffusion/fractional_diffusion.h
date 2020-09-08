#ifndef __FD_HELPER_H__
#define __FD_HELPER_H__

#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <fstream>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>

#include <h2opus.h>
#include "../common/example_util.h"
#include "../common/example_problem.h"
#include "../common/simple_wrappers.h"
#include "../common/hmatrix_samplers.h"

template <typename T> struct FractionalDiffusionFunction
{
  private:
    T h, sigma, alpha;
    T c1, c2;
    int d;

  public:
    FractionalDiffusionFunction(T h, T sigma, T alpha, int dim)
    {
        this->h = h;
        this->sigma = sigma;
        this->alpha = alpha;
        this->d = dim;

        c1 = pow(h / sigma, d) / pow(sigma, alpha);
        c2 = -pow(2.0, alpha) * gsl_sf_gamma((alpha + (T)d) / 2) / pow(M_PI, (T)d / 2) / gsl_sf_gamma((T)d / 2);
    }

    T G(T x) const
    {
        return c2 * gsl_sf_hyperg_1F1((alpha + (T)d) / 2, (T)d / 2, -x * x);
    }

    T operator()(T *pt_x, T *pt_y) const
    {
        T diff = 0;
        for (int i = 0; i < d; i++)
            diff += (pt_x[i] - pt_y[i]) * (pt_x[i] - pt_y[i]);
        T dist = sqrt(diff);

        return c1 * G(dist / sigma);
    }
};

#endif

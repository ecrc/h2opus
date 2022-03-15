#ifndef __FD_HELPER_H__
#define __FD_HELPER_H__

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
    Spatial_Statistics(T phi, T nu,  int dim)
    {
        this->phi = phi;
        this->nu = nu;
        //this->alpha = alpha;
        this->d = dim;

	con = pow(2,(nu-1)) * tgamma(nu);
        con = 1.0/con;

        //c1 = pow(h / sigma, d) / pow(sigma, alpha);
        //c2 = -pow(2.0, alpha) * gsl_sf_gamma((alpha + (T)d) / 2) / pow(M_PI, (T)d / 2) / gsl_sf_gamma((T)d / 2);
    }

    //T G(T x) const
    //{
    //    return c2 * gsl_sf_hyperg_1F1((alpha + (T)d) / 2, (T)d / 2, -x * x);
    //}

    T operator()(T *pt_x, T *pt_y) const
    {
        T diff = 0;
        for (int i = 0; i < d; i++)
            diff += (pt_x[i] - pt_y[i]) * (pt_x[i] - pt_y[i]);
        T dist = sqrt(diff);
	
	dist = 4 * sqrt(2*nu) * (dist/phi);
	if(dist==0){
		return 1;
	}else{
		return con * pow(dist, nu) * gsl_sf_bessel_Knu(nu, dist);
	}
    }
};

#endif

#include "../stimfit/stf.h"
#include "../stimfit/math/fit.h"
#include "../stimfit/math/funclib.h"
#include <gtest/gtest.h>
#include <cmath>

static std::vector< stf::storedFunc > funcLib = stf::GetFuncLib();

const static int nmax = 32768;
const static double tol = 0.1;


void par_test(double value, double expected, double tolerance) {
    EXPECT_NEAR(value, expected, abs(expected*tolerance));
}

// Tests fiting to a monoexponential function
TEST(fitlib_test, monoexponential) {

    double tau = 3000.0;

    Vector_double data(32768);
    for (int n = 0; n < data.size(); ++n) {
        data[n] = 1.0-exp(-n/tau);
    }

    // Respectively the scale factor for initial \mu,
    // stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2,
    // maxIter, maxPass
    Vector_double opts(6);
    opts[0]=5*1E-3; //default: 1E-03;
    opts[1]=1E-17; //default: 1E-17;
    opts[2]=1E-17; //default: 1E-17;
    opts[3]=1E-32; //default: 1E-17;
    opts[4]=64; //default: 64;
    opts[5]=16;

    /* Initial parameter guesses */
    Vector_double pars(3);
    pars[0] = -0.1;
    pars[1] = 3050.0;
    pars[2] = 1.1;

    std::string info;
    int warning;
    double chisqr = lmFit(data, 1.0, funcLib[0], opts, true /*use_scaling*/,
                          pars, info, warning );

    EXPECT_EQ(warning, 0);
    par_test(pars[0], -1.0, tol); /* Amplitude */
    par_test(pars[1], tau, tol);  /* Time constant */
    par_test(pars[2], 1.0, tol);  /* Baseline */


#if 0    
    int fd = open("debug.log", O_WRONLY|O_CREAT|O_TRUNC, 0660);
    assert(fd >= 0);
    int ret = dup2(fd, 1);
    assert(ret >= 0);
    std::cout << "chisqr = " << chisqr << std::endl;
    std::cout << "info = " << info << std::endl;
    std::cout << "warning = " << warning << std::endl;
    for (int n = 0; n < pars.size(); ++n) {
        std::cout << "Pars[" << n << "] = " << pars[n] << std::endl;
    }
    close(fd);
#endif
}

//=========================================================================
// Tests fitting to a monoexponential function, offset fixed to baseline 
// f(x; base, tau) = base - exp(-x/tau),
//=========================================================================
TEST(fitlib_test, monoexponential_offset2baseline){

    /* generate data */
    double tau = 3000.0;
    double base = -20.0;

    Vector_double data(nmax);
    for (int n = 0; n < data.size(); ++n) {
        data[n] = base - exp(-n/tau);
    }

    /* Initial parameters */
    Vector_double pars(3);
    pars[0] = -1.1;
    pars[1] = 3050.0;
    pars[2] = base; /* Has to be exactly base because this parameter is kept constant */

    // stopping thresholds 
    Vector_double opts(6);
    opts[0] = 5*1E-3; // scale factor for initial \mu,   default: 1E-03;
    opts[1] = 1E-17;  // ||J^T e||_inf, default: 1E-17;
    opts[2] = 1E-17;  // ||Dp||_2, default: 1E-17;
    opts[3] = 1E-32;  // ||e||_2,  default: 1E-17;
    opts[4] = 64;     // maxIter, default: 64;
    opts[5] = 16;     // maxPass

    std::string info;
    int warning;
    double chisqr = lmFit(data, 1.0, funcLib[1], opts, true,
        pars, info, warning);

#if 0
    int fd = open("debug.log", O_WRONLY|O_CREAT|O_TRUNC, 0660);
    assert(fd >= 0);
    int ret = dup2(fd, 1);
    assert(ret >= 0);
    std::cout << "chisqr = " << chisqr << std::endl;
    std::cout << "info = " << info << std::endl;
    std::cout << "warning = " << warning << std::endl;
    for (int n = 0; n < pars.size(); ++n) {
        std::cout << "Pars[" << n << "] = " << pars[n] << std::endl;
    }
    close(fd);
#endif
    EXPECT_EQ(warning,0); 
    par_test(pars[0], -1.0, tol); /* Amplitude */
    par_test(pars[1], tau, tol);  /* Time constant */
    par_test(pars[2], base, tol);  /* Baseline */
}

#include "../stimfit/stf.h"
#include "../stimfit/math/fit.h"
#include "../stimfit/math/funclib.h"
#include <gtest/gtest.h>
#include <cmath>
#include <fstream>


const static int nmax = 32768;
const static double tol = 0.1;

/* global variables to define our data */
const static int tmax = 100;   /* length of data in ms */
const static float dt = 1/20.0; /* sampling interval of data in ms */

/* list of available fitting functions */
const static std::vector< stf::storedFunc > funcLib = stf::GetFuncLib();

//=========================================================================
// 03 - Monoexponential function with delay, start fixed to baseline 
// param is an array of parameters, where
// param[0] is the baseline, 
// param[1] is the delay,
// param[2] is the time constant, 
// param[3] is the amplitude
//=========================================================================
Vector_double fexpde(const Vector_double &param){
    Vector_double mydata (int (tmax/dt));

    for (int n=0; n<mydata.size(); ++n){
        mydata[n] = stf::fexpde(n*dt, param);
    }
    
    return mydata;
}

//=========================================================================
// 05 - Biexponential function with offset to baseline
// param is an array of parameters, where
// param[0] is the baseline, 
// param[1] is the delay,
// param[2] is the slow time constant (e.g tau_h, tau_2)
// param[3] is the factor,
// param[4] is the fast time constant (e.g tau_m, tau_1)
//=========================================================================
Vector_double fexpbde(const Vector_double &param){
    Vector_double mydata (int(tmax/dt));
    
    for (int n=0; n<mydata.size(); ++n){
        mydata[n] = stf::fexpbde(n*dt, param);
    }
    
    return mydata;
}

//=========================================================================
// 09 - Alpha function with offset to baseline
// param is an array of parameters, where
// param[0] is the amplitude, 
// param[1] is the rate,
// param[2] is the offset 
//=========================================================================
Vector_double falpha(const Vector_double &param){
    Vector_double mydata (int(tmax/dt));
    
    for (int n=0; n<mydata.size(); ++n){
        mydata[n] = stf::falpha(n*dt, param);
    }
}

//#if 0
void savetxt(const Vector_double &mydata){

    std::ofstream output_file;
    output_file.open("array.out");

    for (int n=0; n<mydata.size(); ++n){
        output_file << mydata[n] << std::endl;
        std::cout << mydata[n] << std::endl;
    }

    output_file.close();
}

void debug_stdout(double chisqr, const std::string& info, int warning, \
    const Vector_double &pars){

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
}
//#endif

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

    data.clear();

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

    data.clear();
}

//=========================================================================
// Tests fitting to a monoexponential function with delay, start to base 
// Stimfit function with ID = 3
//=========================================================================
TEST(fitlib_test, id_03_monoexponential_with_delay){

    /* choose function parameters */
    Vector_double mypars(4);
    mypars[0] = 10.0; /* baseline */
    mypars[1] = 15.0; /* delay */
    mypars[2] = 17.0; /* time constant */
    mypars[3] = 90.0; /* amplitude */

    /* create a 100 ms trace with mypars */
    Vector_double data;
    data = fexpde(mypars);

#if 0
    savetxt(data);
#endif

    /* options for the implementation of the LM algorithm */
    Vector_double opts(6);
    opts[0] = 5*1E-3; opts[1] = 1E-17; opts[2] = 1E-17;
    opts[3] = 1E-32; opts[4] = 64; opts[5] = 16;

    /* Initial parameter guesses */
    Vector_double pars(4);
    pars[0] = mypars[0]; 
    pars[1] = 5.0;
    pars[2] = 50.0;
    pars[3] = 21.0;

    std::string info;
    int warning;
    double chisqr = lmFit(data, dt, funcLib[2], opts, true, pars, \
        info, warning );

    EXPECT_EQ(warning, 0);
    par_test(pars[0], mypars[0], tol); /* baseline */
    par_test(pars[1], mypars[1], tol); /* delay */
    par_test(pars[2], mypars[2], tol); /* time constant */
    par_test(pars[3], mypars[3], tol); /* amplitude */

#if 0
    debug_stdout(chisqr, info, warning, pars);
#endif
    data.clear();
}

//=========================================================================
// Tests fitting to an alpha function, 
// Stimfit function with ID = 9
//=========================================================================
TEST(fitlib_test, id_09_alpha){
    
    /* choose function parameters */
    Vector_double mypars(3);
    mypars[0] = 1500.0;    /* amplitude */
    mypars[1] = 0.5;       /* rate      */
    mypars[2] = 50.0;      /* offset    */

    /* create a 100 ms trace with mypars */
    Vector_double data;
    data = falpha(mypars);


//#if 0
    savetxt(data);
//#endif

    /* options for the implemenation of the LM algorithm */
    Vector_double opts(6);
    opts[0] = 5*1E-3; opts[1] = 1E-17; opts[2] = 1E-17;
    opts[3] = 1E-32; opts[4] = 64; opts[5] = 16;

    /* Initial parameter guesses */
    Vector_double pars(3);
    pars[0] = 1000.0;          /* amplitude */
    pars[1] = 0.10;            /* rate      */
    pars[2] = 8.0;             /* offset    */

    std::string info;
    int warning;

    double chisqr = lmFit(data, dt, funcLib[8], opts, true, pars, info,\
         warning );

    EXPECT_EQ(warning, 0);
    par_test(pars[0], mypars[0], tol);  /* amplitude */
    par_test(pars[1], mypars[1], tol);  /* rate      */
    par_test(pars[2], mypars[2], tol);  /* offset    */

//#if 0
    debug_stdout(chisqr, info, warning, pars);
//#endif

    data.clear();
}

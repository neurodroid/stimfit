#include "../stimfit/stf.h"
#include "../stimfit/math/fit.h"
#include "../stimfit/math/funclib.h"
#include <gtest/gtest.h>
#include <cmath>
#include <fstream>

#define EPSILON 0.00001f /*absolute tolerance value */

const static int nmax = 32768;
const static double tol = 0.1; /* param-relative tolerance value */

/* global variables to define our data */
const static int tmax = 100;   /* length of data in ms */
const static float dt = 1/20.0; /* sampling interval of data in ms */

/* list of available fitting functions */
const static std::vector< stf::storedFunc > funcLib = stf::GetFuncLib();

/* options for the implementation of the LM algorithm */
const Vector_double LM_opts(){

    Vector_double opts(6);
    opts[0]=5*1E-3;   // initial \mu, default: 1E-03;
    opts[1]=1E-17;    // stopping thr for ||J^T e||_inf, default: 1E-17;
    opts[2]=1E-17;    // stopping trh for ||Dp||_2, default: 1E-17;
    opts[3]=1E-32;    // stopping thr for ||e||_2, default: 1E-17;
    opts[4]=64;       // maximal number of iterations/pass, default: 64;
    opts[5]=16;       // maximal number of passes;
    
    return opts;
}

//=========================================================================
// Simple monoexponential function
// available for fitting to function 0 of Stimfit
// param is an array of parameters, where
// param[0] is the amplitude (peak-offset),
// param[1] is the time constant,
// param[2] is the end (offset)
//=========================================================================
Vector_double fexp_simple(const Vector_double &param){
    Vector_double mydata (int (tmax/dt));
    double amp  = param[0];
    double tau  = param[1];
    double end =  param[2]; 
    
    for (std::vector<int>::size_type n=0; n != mydata.size() ; ++n){
        mydata[n] =   amp*exp(-(n*dt)/tau) + end; 
    }
    return mydata;
}

//=========================================================================
// Monoexponential function with delay, start fixed to baseline 
// available for fitting to function 2 of Stimfit
// param is an array of parameters, where
// param[0] is the baseline, 
// param[1] is the delay,
// param[2] is the time constant, 
// param[3] is the amplitude
//=========================================================================
Vector_double fexpde(const Vector_double &param){
    Vector_double mydata (int (tmax/dt));

    for (std::vector<int>::size_type n=0; n != mydata.size(); ++n){
        mydata[n] = stf::fexpde(n*dt, param);
    }
    
    return mydata;
}

//=========================================================================
// Biexponential function with offset to baseline
// corresponding to fitting function 5 of Stimfit
// param is an array of parameters, where
// param[0] is the baseline, 
// param[1] is the delay,
// param[2] is the slow time constant (e.g tau_h, tau_2)
// param[3] is the factor,
// param[4] is the fast time constant (e.g tau_m, tau_1)
//=========================================================================
Vector_double fexpbde(const Vector_double &param){
    Vector_double mydata (int(tmax/dt));
    
    for (std::vector<int>::size_type n=0; n != mydata.size(); ++n){
        mydata[n] = stf::fexpbde(n*dt, param);
    }
    
    return mydata;
}

//=========================================================================
// Alpha function with offset to baseline
// corresponding to fitting function 9 of Stimfit
// param is an array of parameters, where
// param[0] is the amplitude, 
// param[1] is the rate,
// param[2] is the offset 
//=========================================================================
Vector_double falpha(const Vector_double &param){
    Vector_double mydata (int(tmax/dt));
    
    for (std::vector<int>::size_type n=0; n != mydata.size(); ++n){
        mydata[n] = stf::falpha(n*dt, param);
    }
    
    return mydata;
}

//=========================================================================
// Hodkin-Huxley-like sodium conductance function of the form:
// f(v;Na_bar, m, h, base) = Na_bar*m(v)^3*h(v) + base
// corresponding to fitting function 10 of Stimfit
// param is an array of parameters, where
// param[0] is the peak sodium conductance
// param[1] is the activation time constant (tau_m) 
// param[2] is the inactivation time constant (tau_h)
// param[3] is the offset
//=========================================================================
Vector_double fHH(const Vector_double &param){

    Vector_double mydata (int(tmax/dt));
    
    for (std::vector<int>::size_type n=0; n != mydata.size(); ++n){
        mydata[n] = stf::fHH(n*dt, param);
    }
    
    return mydata;
}

//#if 0
void savetxt(Vector_double &mydata){

    std::ofstream output_file;
    output_file.open("array.out");

    Vector_double::iterator it;

    for (it = mydata.begin(); it != mydata.end(); ++it){
        output_file << *it << std::endl;
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
    for (std::vector<int>::size_type n = 0; n != pars.size(); ++n) {
        std::cout << "Pars[" << n << "] = " << pars[n] << std::endl;
    }
    close(fd);
}
//#endif

void par_test(double value, double expected, double tolerance) {
    /* for very small expected values, the product of expected*tolerance
    would be smaller than the computer precission for a double????. 
    The computer returns abs(expected*tolerance) equals to zero, and 
    the test fails for that reason, a minimal tolerance value of EPSILON
    is given */
    /*if ( fabs(expected*tolerance) < EPSILON )
        EXPECT_NEAR(value, expected, EPSILON);
    else 
        EXPECT_NEAR(value, expected, fabs(expected*tolerance) );
    */
    EXPECT_NEAR(value, expected, fabs(expected*tolerance) );
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
// Tests fitting to a monoexponential function 
// Stimfit function with ID = 0 
//=========================================================================
TEST(fitlib_test, id_00_monoexponential){

    /* choose function parameters */
    Vector_double mypars(3);
    mypars[0] = 50.0;   /* amplitude */
    mypars[1] = 17.0;   /* time constant */
    mypars[2] = -20.0;  /* end  */


    /* create a 100 ms trace with mypars */
    Vector_double data;
    data = fexp_simple(mypars);

#if 0
    savetxt(data);
#endif

    /* options for the implementation of the LM algorithm */
    Vector_double opts;
    opts = LM_opts();

    /* Initial parameters guesses */
    Vector_double pars(3);
    pars[0] = 0.0;        /* Offset */
    pars[1] = 5.0;        /* Tau_0 */
    pars[2] = -35.0;      /* Amp_0 */

    std::string info;
    int warning;

    double chisqr = lmFit(data, dt, funcLib[0], opts, true, pars, info, warning );

    EXPECT_EQ(warning, 0);
    par_test(pars[0], mypars[0], tol);  /* Amp_0  */
    par_test(pars[1], mypars[1], tol);  /* Tau_0  */
    par_test(pars[2], mypars[2], tol);  /* Offset */

#if 0
    debug_stdout(chisqr, info, warning, pars);
#endif

    data.clear();

}

//=========================================================================
// Tests fitting to a monoexponential function, offset fixed to baseline
// Stimfit function with ID = 1 
//=========================================================================
TEST(fitlib_test, id_01_monoexponential_offsetfixed){

    /* choose function parameters */
    Vector_double mypars(3);
    mypars[0] = 80.0;   /* amplitude */
    mypars[1] = 34.0;   /* time constant */
    mypars[2] = -50.0;  /* end  */


    /* create a 100 ms trace with mypars */
    Vector_double data;
    data = fexp_simple(mypars);

#if 0
    savetxt(data);
#endif

    /* options for the implementation of the LM algorithm */
    Vector_double opts;
    opts = LM_opts();

    /* Initial parameters guesses */
    Vector_double pars(3);
    pars[0] = mypars[0];        /* Offset fixed to baseline */
    pars[1] = 5.0;              /* Tau_0 */
    pars[2] = -35.0;            /* Amp_0 */

    std::string info;
    int warning;

    double chisqr = lmFit(data, dt, funcLib[0], opts, true, pars, info, warning );

    EXPECT_EQ(warning, 0);
    par_test(pars[0], mypars[0], tol);  /* Amp_0  */
    par_test(pars[1], mypars[1], tol);  /* Tau_0  */
    par_test(pars[2], mypars[2], tol);  /* Offset */

#if 0
    debug_stdout(chisqr, info, warning, pars);
#endif

    data.clear();

}

//=========================================================================
// Tests fitting to a monoexponential function with delay, start to base 
// Stimfit function with ID = 2
//=========================================================================
TEST(fitlib_test, id_02_monoexponential_with_delay){

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
    Vector_double opts;
    opts = LM_opts();

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
// Tests fitting to a biexponential with delay, offset fixed to baseline
// Stimfit function with ID = 5
//=========================================================================
TEST(fitlib_test, id_05_biexponential_with_delay_offsetfixed){
   
    /* choose function parameters */
    Vector_double mypars(5);
    mypars[0] = 0.0;     /* baseline            */
    mypars[1] = 25.0;    /* Delay               */
    mypars[2] = 10.0;    /* fast time constant  */
    mypars[3] = 25.0;    /* Factor              */
    mypars[4] = 50.0;    /* slow time constant  */

    /* create a 100 ms trace with mypars */
    Vector_double data;
    data = fexpbde(mypars);

    /* options for the implementation of the LM algorithm */
    Vector_double opts;
    opts = LM_opts();

    /* Initial parameter guesses */
    Vector_double pars(5);
    pars[0] = mypars[0];      /* offset fixed to baseline! */
    pars[1] = 0.1;            /* Delay    */
    pars[2] = 19.925;         /* tau1    */
    pars[3] = 20.0;           /* Factor   */
    pars[4] = 24.9875;        /* tau2    */

    std::string info;
    int warning;

    double chisqr = lmFit(data, dt, funcLib[5], opts, true, pars, info, warning );

    EXPECT_EQ(warning, 0);
    par_test(pars[0], mypars[0], tol);  /* baseline */
    par_test(pars[1], mypars[1], tol);  /* delay */
    par_test(pars[2], mypars[2], tol);  /* short time constant */
    par_test(pars[3], mypars[3], tol);  /* Factor != amplitude */
    par_test(pars[4], mypars[4], tol);  /* long time constant */


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
    mypars[0] = 300.0;    /* amplitude */
    mypars[1] = 0.1;       /* rate      */
    mypars[2] = 50.0;      /* offset    */

    /* create a 100 ms trace with mypars */
    Vector_double data;
    data = falpha(mypars);


#if 0
    savetxt(data);
#endif

    /* options for the implementation of the LM algorithm */
    Vector_double opts;
    opts = LM_opts();

    /* Initial parameter guesses */
    Vector_double pars(3);
    pars[0] = 1101.72;       /* Q      */
    pars[1] = 0.2001;        /* rate   */
    pars[2] = 8.0;           /* Offset */

    std::string info;
    int warning;

    double chisqr = lmFit(data, dt, funcLib[9], opts, true, pars, info,\
         warning );

    EXPECT_EQ(warning, 0);
    par_test(pars[0], mypars[0], tol);  /* amplitude */
    par_test(pars[1], mypars[1], tol);  /* rate      */
    par_test(pars[2], mypars[2], tol);  /* offset    */

#if 0
    debug_stdout(chisqr, info, warning, pars);
#endif

    data.clear();

}
//=========================================================================
// Tests fitting to a HH-type sodium conductance, offset fixed to baseline
// Stimfit function with ID =10 
//=========================================================================
TEST(fitlib_test, id_10_HH_gNa_offsetfixed){

    /* choose function parameters */
    Vector_double mypars(4);
    mypars[0] = 120.0;   /* maximal sodium conductance   */
    mypars[1] = 130e-3;    /* activation time constat (tau_m_) in us    */
    mypars[2] = 728e-3;   /* inactivation time constant  in us */
    mypars[3] =   0.0;   /* offset                       */

    /* create a 100 ms trace with mypars */
    /* here the x-units and x-axis values are not relevant */
    Vector_double data;
    data = fHH(mypars);

//#if 0
    savetxt(data);
//#endif

    /* options for the implementation of the LM algorithm */
    Vector_double opts;
    opts = LM_opts();

    /* Initial parameter guesses */
    Vector_double pars(4);
    pars[0] = 1000.72;       /* gprime_na   */
    pars[1] = 0.2001;        /* tau_m       */
    pars[2] = 8.0;           /* tau_h       */
    pars[3] = mypars[3];      /* offset      */

    std::string info;
    int warning;

    double chisqr = lmFit(data, dt, funcLib[10], opts, true, pars, info,\
         warning );

    EXPECT_EQ(warning, 0);
    par_test(pars[0], mypars[0], tol);  /* gprime_na */
    par_test(pars[1], mypars[1], tol);  /* tau_m     */
    par_test(pars[2], mypars[2], tol);  /* tau_h     */
    par_test(pars[3], 0.0, tol);        /* offset    */

//#if 0
    debug_stdout(chisqr, info, warning, pars);
//#endif

    data.clear();

}

#include "../stimfit/stf.h"
#include "../stimfit/math/fit.h"
#include "../stimfit/math/funclib.h"
#include <gtest/gtest.h>
#include <cmath>
#include <fstream>


/* global variables to define our data */
const static int tmax = 100;   /* length of data in ms */
const static float dt = 1/100.0; /* sampling interval of data in ms */
//const static double tol = 0.001; /* param-relative tolerance value */
const static float tol = dt; /* 1 sampling interval */

/* list of available fitting functions, see /src/stimfit/math/funclib.cpp */
const static std::vector< stf::storedFunc > funcLib = stf::GetFuncLib();

/* Fitting options for the LM algorithm, see /src/stimfit/math/fit.h */
const Vector_double opts = stf::LM_default_opts();

//=========================================================================
// Simple monoexponential function
// available for fitting to function 0, and 1 of Stimfit
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
// Monoexponential function with delay, start fixed to baseline 
// available for fitting to function 3, 4, 6, 7 and 8 of Stimfit
// param is an array of parameters, where
// event terms (e.g param[0]) are the amplitudes, 
// odd terms (e.g param[1]) are the time constants, 
// param[last] is the offset
//=========================================================================
Vector_double fexp(const Vector_double &param){
    Vector_double mydata (int (tmax/dt));

    for (std::vector<int>::size_type n=0; n != mydata.size(); ++n){
        mydata[n] = stf::fexp(n*dt, param);
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

//=========================================================================
// Hodkin-Huxley-like sodium conductance function of the form:
// f(v;Na_bar, m, h, base) = Na_bar*m(v)*h(v) + base
// corresponding to fitting function 11 of Stimfit
// param is an array of parameters, where
// param[0] is the peak sodium conductance
// param[1] is the activation time constant (tau_m) 
// param[2] is the inactivation time constant (tau_h)
// param[3] is the offset
//=========================================================================
Vector_double fgnabiexp(const Vector_double &param){

    Vector_double mydata (int(tmax/dt));
    
    for (std::vector<int>::size_type n=0; n != mydata.size(); ++n){
        mydata[n] = stf::fgnabiexp(n*dt, param);
    }
    
    return mydata;
}

//=========================================================================
// A gaussian function of the form
// f(x, a, b, c) = a*exp((x-b)^2/2*c^2)
// corresponding to the fitting function 12 of Stimfit
// param in an array of parameters, where
// param[0] is the heigth of the Gaussian function (a)
// param[1] is the position of the peak (b)
// param[2] is the width of the gaussian (c)
//=========================================================================
Vector_double fgauss(const Vector_double &param){

    Vector_double mydata (int(tmax/dt));
    
    for (std::vector<int>::size_type n=0; n != mydata.size(); ++n){
        mydata[n] = stf::fgauss(n*dt, param);
    }
    
    return mydata;
}

//#if 0
void savetxt(const char *fname, Vector_double &mydata){

    std::ofstream output_file;
    output_file.open(fname);

    Vector_double::iterator it;

    for (it = mydata.begin(); it != mydata.end(); ++it){
        output_file << *it << std::endl;
    }

    output_file.close();
}

void debug_stdout(double chisqr, const std::string& info, int warning, 
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
    Vector_double opts = stf::LM_default_opts();

    /* Initial parameter guesses */
    Vector_double pars(3);
    pars[0] = -0.1;
    pars[1] = 3050.0;
    pars[2] = 1.1;

    std::string info;
    int warning;
    double chisqr = lmFit(data, 1.0, funcLib[0], opts, 
        true, /*use_scaling*/
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

    /* Initial parameters guesses */
    Vector_double pars(3);
    pars[0] = 0.0;        /* Offset */
    pars[1] = 5.0;        /* Tau_0 */
    pars[2] = -35.0;      /* Amp_0 */

    std::string info;
    int warning;

    double chisqr = lmFit(data, dt, funcLib[0], opts, 
        true, /* use_scaling */
        pars, info, warning );

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
    mypars[0] = 80.0;   /* amplitude     */
    mypars[1] = 34.0;   /* time constant */
    mypars[2] = -5.52;  /* offset        */


    /* create a 100 ms trace with mypars */
    Vector_double data;
    data = fexp_simple(mypars);

#if 0
    savetxt("/tmp/monoexp1.out", data);
#endif

    /* Initial parameters guesses */
    Vector_double pars(3);
    pars[0] = 35.5232;     /* Amp_0 */
    pars[1] = 14.6059;     /* Tau_0 */
    pars[2] = mypars[2];   /* Offset fixed to baseline */

    std::string info;
    int warning;

    double chisqr = lmFit(data, dt, funcLib[1], opts, 
        true, /* use_scaling */
        pars, info, warning );

    EXPECT_EQ(warning, 0);
    par_test(pars[0], mypars[0], tol);  /* Amp_0  */
    par_test(pars[1], mypars[1], tol);  /* Tau_0  */
    EXPECT_EQ(pars[2], mypars[2]);      /* Offset to baseline */

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

    /* Initial parameter guesses */
    Vector_double pars(4);
    pars[0] = mypars[0]; 
    pars[1] = 5.0;
    pars[2] = 50.0;
    pars[3] = 21.0;

    std::string info;
    int warning;
    double chisqr = lmFit(data, dt, funcLib[2], opts, 
        true, /*use_scaling*/
        pars, info, warning );

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
// Tests fitting to a biexponential 
// Stimfit function with ID = 3
//=========================================================================
TEST(fitlib_test, id_03_biexponential){

    /* choose function parameters */
    Vector_double mypars(5);
    mypars[0] = 9.0;    /* first amplitude      */
    mypars[1] = 2.0;    /* first time constant  */
    mypars[2] = 1.0;    /* second amplitude     */
    mypars[3] = 15.0;   /* second time constant */
    mypars[4] = 4.0;    /* baseline             */

    /* create a 100 ms trace with mypars */
    Vector_double data;
    data = fexp(mypars);

#if 0
    savetxt("/tmp/mybiexp3.out", data);
#endif

    /* Initial parameter guesses */
    Vector_double pars(5);
    pars[0] = 4.86764;    /* Amp_0   */
    pars[1] = 2.44482;    /* Tau_0   */
    pars[2] = 4.86764;    /* Amp_1   */
    pars[3] = 19.5586;    /* Tau_1   */
    pars[4] = 4.26472;    /* Offset  */

    std::string info;
    int warning;

    double chisqr = lmFit(data, dt, funcLib[3], opts,  
        true, /*use_scaling*/
        pars, info, warning );

    EXPECT_EQ(warning, 0);
    par_test(pars[0], mypars[0], tol);  /* Amp_0  */
    par_test(pars[1], mypars[1], tol);  /* Tau_0  */
    par_test(pars[2], mypars[2], tol);  /* Amp_1  */
    par_test(pars[3], mypars[3], tol);  /* Tau_1  */
    par_test(pars[4], mypars[4], tol);  /* Offset */


#if 0
    debug_stdout(chisqr, info, warning, pars);
#endif

    data.clear();
    
}

//=========================================================================
// Tests fitting to a biexponential offset fixed to baseline
// Stimfit function with ID = 4
//=========================================================================
TEST(fitlib_test, id_04_biexponential_offsetfixed){

    /* choose function parameters */
    Vector_double mypars(5);
    mypars[0] = 16.0;    /* first amplitude      */
    mypars[1] = 2.6;    /* first time constant  */
    mypars[2] = 12.0;    /* second amplitude     */
    mypars[3] = 15.0;   /* second time constant */
    mypars[4] = 4.0;    /* baseline             */

    /* create a 100 ms trace with mypars */
    Vector_double data;
    data = fexp(mypars);

#if 0
    savetxt("/tmp/mybiexp3.out", data);
#endif

    /* Initial parameter guesses */
    Vector_double pars(5);
    pars[0] = 4.86764;    /* Amp_0   */
    pars[1] = 2.44482;    /* Tau_0   */
    pars[2] = 4.86764;    /* Amp_1   */
    pars[3] = 19.5586;    /* Tau_1   */
    pars[4] = 4.0;        /* Offset  */

    std::string info;
    int warning;

    double chisqr = lmFit(data, dt, funcLib[4], opts, 
        true, /*use_scaling*/
        pars, info, warning );

    EXPECT_EQ(warning, 0);
    par_test(pars[0], mypars[0], tol);  /* Amp_0  */
    par_test(pars[1], mypars[1], tol);  /* Tau_0  */
    par_test(pars[2], mypars[2], tol);  /* Amp_1  */
    par_test(pars[3], mypars[3], tol);  /* Tau_1  */
    EXPECT_EQ(pars[4], mypars[4]);      /* Offset */


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

    /* Initial parameter guesses */
    Vector_double pars(5);
    pars[0] = mypars[0];      /* offset fixed to baseline! */
    pars[1] = 0.1;            /* Delay    */
    pars[2] = 19.925;         /* tau1    */
    pars[3] = 20.0;           /* Factor   */
    pars[4] = 24.9875;        /* tau2    */

    std::string info;
    int warning;

    double chisqr = lmFit(data, dt, funcLib[5], opts,  
        true, /*use_scaling*/
        pars, info, warning );

    EXPECT_EQ(warning, 0);
    EXPECT_EQ(pars[0], mypars[0]);      /* Offset fixed to baseline */
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
// Tests fitting to a triexponential 
// Stimfit function with ID = 6
//=========================================================================
TEST(fitlib_test, id_06_triexponential){

    /* choose function parameters */
    Vector_double mypars(7);
    mypars[0] = 5.76;   /* first amplitude      */
    mypars[1] = 3.37;   /* first time constant  */
    mypars[2] = 5.76;   /* second amplitude     */
    mypars[3] = 26.9;   /* second time constant */
    mypars[4] = 5.76;   /* third amplitude     */
    mypars[5] = 91.0;   /* third time constant */
    mypars[6] = 5.07;    /* baseline             */

    /* create a 100 ms trace with mypars */
    Vector_double data;
    data = fexp(mypars);

#if 0
    savetxt("/tmp/mytriexp6.out", data);
#endif

    /* Initial parameter guesses */
    Vector_double pars(7);
    pars[0] = 4.86764;    /* Amp_0   */
    pars[1] = 6.44482;    /* Tau_0   */
    pars[2] = 4.86764;    /* Amp_1   */
    pars[3] = 49.5586;    /* Tau_1   */
    pars[4] = 4.86764;    /* Amp_2   */
    pars[5] = 165.5586;   /* Tau_2   */
    pars[6] = 6.55319;    /* Offset  */

    std::string info;
    int warning;

    double chisqr = lmFit(data, dt, funcLib[6], opts,  
        true, /*use_scaling*/
        pars, info, warning );

    EXPECT_EQ(warning, 0);
    par_test(pars[0], mypars[0], tol);  /* Amp_0  */
    par_test(pars[1], mypars[1], tol);  /* Tau_0  */
    par_test(pars[2], mypars[2], tol);  /* Amp_1  */
    par_test(pars[3], mypars[3], tol);  /* Tau_1  */
    par_test(pars[4], mypars[4], tol);  /* Amp_2  */
    par_test(pars[5], mypars[5], tol);  /* Tau_2  */
    par_test(pars[6], mypars[6], tol);  /* Offset */


#if 0
    debug_stdout(chisqr, info, warning, pars);
#endif

    data.clear();
    
}


//=========================================================================
// Tests fitting to a triexponential free 
// Stimfit function with ID = 7
//=========================================================================
TEST(fitlib_test, id_07_triexponential_free){

    /* choose function parameters */
    Vector_double mypars(7);
    mypars[0] = 15.76;   /* first amplitude      */
    mypars[1] = 3.7;   /* first time constant  */
    mypars[2] = 19.76;   /* second amplitude     */
    mypars[3] = 21.9;   /* second time constant */
    mypars[4] = 2.76;   /* third amplitude     */
    mypars[5] = 42.2;   /* third time constant */
    mypars[6] = -45.22;    /* baseline             */

    /* create a 100 ms trace with mypars */
    Vector_double data;
    data = fexp(mypars);

#if 0
    savetxt("/tmp/mytriexp7.out", data);
#endif

    /* Initial parameter guesses */
    Vector_double pars(7);
    pars[0] = 4.86764;    /* Amp_0   */
    pars[1] = 6.44482;    /* Tau_0   */
    pars[2] = 4.86764;    /* Amp_1   */
    pars[3] = 49.5586;    /* Tau_1   */
    pars[4] = 4.86764;    /* Amp_2   */
    pars[5] = 65.5586;    /* Tau_2   */
    pars[6] = -34.242;    /* Offset fixed to baseline */

    std::string info;
    int warning;

    double chisqr = lmFit(data, dt, funcLib[7], opts,  
        true, /*use_scaling*/
        pars, info, warning );

    EXPECT_EQ(warning, 0);
    par_test(pars[0], mypars[0], tol);  /* Amp_0  */
    par_test(pars[1], mypars[1], tol);  /* Tau_0  */
    par_test(pars[2], mypars[2], tol);  /* Amp_1  */
    par_test(pars[3], mypars[3], tol);  /* Tau_1  */
    par_test(pars[4], mypars[4], tol);  /* Amp_2  */
    par_test(pars[5], mypars[5], tol);  /* Tau_2  */
    par_test(pars[6], mypars[6], tol);  /* Offset */


#if 0
    debug_stdout(chisqr, info, warning, pars);
#endif

    data.clear();
    
}

//=========================================================================
// Tests fitting to a triexponential with offset fixed to baseline
// Stimfit function with ID = 8
//=========================================================================
TEST(fitlib_test, id_08_triexponential_offsetfixed){

    /* choose function parameters */
    Vector_double mypars(7);
    mypars[0] = 5.76;   /* first amplitude      */
    mypars[1] = 3.37;   /* first time constant  */
    mypars[2] = 9.76;   /* second amplitude     */
    mypars[3] = 21.9;   /* second time constant */
    mypars[4] = 5.76;   /* third amplitude     */
    mypars[5] = 41.0;   /* third time constant */
    mypars[6] = -5.12;    /* baseline             */

    /* create a 100 ms trace with mypars */
    Vector_double data;
    data = fexp(mypars);

#if 0
    savetxt("/tmp/mytriexp8.out", data);
#endif

    /* Initial parameter guesses */
    Vector_double pars(7);
    pars[0] = 4.86764;    /* Amp_0   */
    pars[1] = 6.44482;    /* Tau_0   */
    pars[2] = 4.86764;    /* Amp_1   */
    pars[3] = 49.5586;    /* Tau_1   */
    pars[4] = 4.86764;    /* Amp_2   */
    pars[5] = 165.5586;   /* Tau_2   */
    pars[6] = mypars[6];  /* Offset fixed to baseline */

    std::string info;
    int warning;

    double chisqr = lmFit(data, dt, funcLib[8], opts,  
        true, /*use_scaling*/
        pars, info, warning );

    EXPECT_EQ(warning, 0);
    par_test(pars[0], mypars[0], tol);  /* Amp_0  */
    par_test(pars[1], mypars[1], tol);  /* Tau_0  */
    par_test(pars[2], mypars[2], tol);  /* Amp_1  */
    par_test(pars[3], mypars[3], tol);  /* Tau_1  */
    par_test(pars[4], mypars[4], tol);  /* Amp_2  */
    par_test(pars[5], mypars[5], tol);  /* Tau_2  */
    EXPECT_EQ(pars[6], mypars[6]);      /* Offset */


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
    mypars[1] = 5.7;      /* rate      */
    mypars[2] = 50.0;     /* offset    */

    /* create a 100 ms trace with mypars */
    Vector_double data;
    data = falpha(mypars);


#if 0
    savetxt(data);
#endif

    /* Initial parameter guesses */
    Vector_double pars(3);
    pars[0] = 350.0;       /* Q      */
    pars[1] = 5.7;         /* rate   */
    pars[2] = 0.0;         /* Offset */

    std::string info;
    int warning;

    double chisqr = lmFit(data, dt, funcLib[9], opts, 
        true, /*use_scaling*/
        pars, info, warning );

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
    mypars[1] = 130e-3;  /* activation time constat (tau_m_) in us    */
    mypars[2] = 728e-3;  /* inactivation time constant  in us */
    mypars[3] =   0.0;   /* offset                       */

    /* create a 100 ms trace with mypars */
    /* here the x-units and x-axis values are not relevant */
    Vector_double data;
    data = fHH(mypars);

#if 0
    savetxt("/tmp/gnabar.out", data);
#endif

    /* Initial parameter guesses */
    Vector_double pars(4);
    pars[0] = 1000.72;       /* gprime_na   */
    pars[1] = 0.2001;        /* tau_m       */
    pars[2] = 8.0;           /* tau_h       */
    pars[3] = mypars[3];      /* offset fixed to baseline     */

    std::string info;
    int warning;

    double chisqr = lmFit(data, dt, funcLib[10], opts, 
        true, /* use_scaling */
        pars, info, warning );

    EXPECT_EQ(warning, 0);
    par_test(pars[0], mypars[0], tol);  /* gprime_na                  */
    par_test(pars[1], mypars[1], tol);  /* tau_m                      */
    par_test(pars[2], mypars[2], tol);  /* tau_h                      */
    EXPECT_EQ(pars[3], mypars[3]);      /* offset fixed to baseline   */

#if 0
    debug_stdout(chisqr, info, warning, pars);
#endif

    data.clear();

}

//=========================================================================
// Tests fitting to a HH-type SINGLE sodium conductance, offset fixed to 
// baseline, Stimfit function with ID =11 
//=========================================================================
TEST(fitlib_test, id_11_HH_gNa_biexpoffsetfixed){

    /* choose function parameters */
    Vector_double mypars(4);
    mypars[0] = 120.0;   /* maximal sodium conductance              */
    mypars[1] = 1.3;    /* activation time constat (tau_m_) in us   */
    mypars[2] = 5.2;    /* inactivation time constant  in us        */
    mypars[3] = 10.0;   /* offset                                   */

    /* create a 100 ms trace with mypars */
    Vector_double data;
    data = fgnabiexp(mypars);

#if 0
    savetxt("/tmp/gnabar_single.out", data);
#endif

    /* Initial parameter guesses */
    Vector_double pars(4);
    pars[0] = 123.284;      /* gprime_na                     */
    pars[1] = 2.625;        /* tau_m                         */
    pars[2] = 15.75;        /* tau_h                         */
    pars[3] = mypars[3];    /* offset fixed to baseline      */

    std::string info;
    int warning;

    double chisqr = lmFit(data, dt, funcLib[11], opts, 
        true, /*use_scaling*/
        pars, info, warning );

    EXPECT_EQ(warning, 0);
    par_test(pars[0], mypars[0], tol);  /* gprime_na */
    par_test(pars[1], mypars[1], tol);  /* tau_m     */
    par_test(pars[2], mypars[2], tol);  /* tau_h     */
    EXPECT_EQ(pars[3], mypars[3]);      /* offset fixed to baseline */

#if 0
    debug_stdout(chisqr, info, warning, pars);
#endif

    data.clear();

}

//=========================================================================
// Tests fitting to a gaussian distribution
// Stimfit function with ID =12
//=========================================================================
TEST(fitlib_test, id_12_fgaussian){

    /* choose function parameters */
    Vector_double mypars(3);
    mypars[0] = 1.5;  /* height */
    mypars[1] = 5.0;  /* peak   */
    mypars[2] = 4.5;  /* width  */

    /* create a trace with mypars */
    Vector_double data;
    data = fgauss(mypars);

//#if 0
    savetxt("/tmp/mygaussian.out", data);
//#endif 

    /* Initial parameter guesses */
    Vector_double pars(3);
    pars[0] = 1.72;  /* amplitude   */
    pars[1] = 5.5;   /* mean        */
    pars[2] = 2.0;   /* width       */

    std::string info;
    int warning;

    double chisqr = lmFit(data, dt, funcLib[12], opts, 
        true, /*use_scaling*/
        pars, info, warning );

    EXPECT_EQ(warning, 0);
    par_test(pars[0], mypars[0], tol);  /* amplitude */
    par_test(pars[1], mypars[1], tol);  /* peak     */
    par_test(pars[2], mypars[2], tol);  /* witdth     */

//#if 0
    debug_stdout(chisqr, info, warning, pars);
//#endif
    //data.clear();

}

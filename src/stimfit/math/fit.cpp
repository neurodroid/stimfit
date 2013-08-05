// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#include "./fit.h"
#include "./levmar/levmar.h"

#include <float.h>
#include <cmath>
#include <boost/algorithm/minmax_element.hpp>

namespace stf {
// C-style functions for Lourakis' routines:
void c_func_lour(double *p, double* hx, int m, int n, void *adata);
void c_jac_lour(double *p, double *j, int m, int n, void *adata);

// Helper functions for lmFit to store the function at global scope:
void saveFunc(stf::Func func);
void saveJac(stf::Jac jac);

// A struct that will be passed as a pointer to
// Lourakis' C-functions. It is used to:
// (1) specify which parameters are to be fitted, and
// (2) pass the constant parameters
// (3) the sampling interval
struct fitInfo {
    fitInfo(const std::deque<bool>& fit_p_arg,
            const Vector_double& const_p_arg,
            double dt_arg)
        :   fit_p(fit_p_arg), const_p(const_p_arg),
            dt(dt_arg)
    {}

    // Specifies for each parameter whether the client
    // wants to fit it (true) or to keep it constant (false)
    std::deque<bool> fit_p;

    // A valarray containing the parameters that
    // will be kept constant:
    Vector_double const_p;

    // sampling interval
    double dt;
};
}

// Functions stored at global scope to be called by c_func_lour
// and c_jac_lour
stf::Func func_lour;
stf::Jac jac_lour; 

void stf::saveFunc(stf::Func func) {
    func_lour=func;
}

void stf::saveJac(stf::Jac jac) {
    jac_lour=jac;
}

void stf::c_func_lour(double *p, double* hx, int m, int n, void *adata) {
    // m: the number of parameters that are to be fitted
    // adata: pointer to a struct that (1) specifies which parameters are to be fitted
    //		  and (2) contains the constant parameters
    fitInfo *fInfo=static_cast<fitInfo*>(adata);
    // total number of parameters, including constants:
    int tot_p=(int)fInfo->fit_p.size();
    // all parameters, including constants:
    Vector_double p_f(tot_p);
    for (int n_tp=0, n_p=0, n_f=0;n_tp<tot_p;++n_tp) {
        // if the parameter needs to be fitted...
        if (fInfo->fit_p[n_tp]) {
            // ... take it from *p, ...
            p_f[n_tp] = p[n_p++];
        } else {
            // ... otherwise, take it from the fInfo struct:
            p_f[n_tp] = fInfo->const_p[n_f++];
        }
    }
    for (int n_x=0;n_x<n;++n_x) {
        hx[n_x]=func_lour( (double)n_x*fInfo->dt, p_f);
    }	
}

void stf::c_jac_lour(double *p, double *jac, int m, int n, void *adata) {
    // m: the number of parameters that are to be fitted
    // adata: pointer to a struct that (1) specifies which parameters are to be fitted
    //		  and (2) contains the constant parameters
    fitInfo *fInfo=static_cast<fitInfo*>(adata);
    // total number of parameters, including constants:
    int tot_p=(int)fInfo->fit_p.size();
    // all parameters, including constants:
    Vector_double p_f(tot_p);
    for (int n_tp=0,n_p=0,n_f=0;n_tp<tot_p;++n_tp) {
        // if the parameter needs to be fitted...
        if (fInfo->fit_p[n_tp]) {
            // ... take it from *p, ...
            p_f[n_tp] = p[n_p++];
        } else {
            // ... otherwise, take it from the fInfo struct:
            p_f[n_tp] = fInfo->const_p[n_f++];
        }
    }
    for (int n_x=0,n_j=0;n_x<n;++n_x) {
        // jac_f will calculate the derivatives of all parameters,
        // including the constants...
        Vector_double jac_f(jac_lour((double)n_x*fInfo->dt,p_f));
        // ... but we only need the derivatives of the non-constants...
        for (int n_tp=0;n_tp<tot_p;++n_tp) {
            // ... hence, we will eliminate the derivatives of the constants:
            if (fInfo->fit_p[n_tp]) {
                jac[n_j++]=jac_f[n_tp];
            }
        }
    }
}

Vector_double stf::get_scale(Vector_double& data, double oldx) {
    Vector_double xyscale(4);
    std::pair<Vector_double::const_iterator, Vector_double::const_iterator> minmax;
    minmax = boost::minmax_element(data.begin(), data.end());
    double ymin = *minmax.first;
    double ymax = *minmax.second;
    double amp = ymax-ymin;
    data = stfio::vec_scal_mul(data, 1.0/amp);
    data = stfio::vec_scal_minus(data, ymin/amp);

    xyscale[0] = 1.0/(data.size()*oldx);
    xyscale[1] = 0;
    xyscale[2] = 1.0/amp;
    xyscale[3] = ymin/amp;
    
    return xyscale;
}

double stf::lmFit( const Vector_double& data, double dt,
                   const stf::storedFunc& fitFunc, const Vector_double& opts,
                   bool use_scaling,
                   Vector_double& p, std::string& info, int& warning )
{
    // Basic range checking:
    if (fitFunc.pInfo.size()!=p.size()) {
        std::string msg("Error in stf::lmFit()\n"
                "function parameters (p_fit) and parameters entered (p) have different sizes");
        throw std::runtime_error(msg);
    }
    if ( opts.size() != 6 ) {
        std::string msg("Error in stf::lmFit()\n"
                "wrong number of options");
        throw std::runtime_error(msg);
    }

    bool constrained = false;
    std::vector< double > constrains_lm_lb( fitFunc.pInfo.size() );
    std::vector< double > constrains_lm_ub( fitFunc.pInfo.size() );

    bool can_scale = use_scaling;
    
    for ( unsigned n_p=0; n_p < fitFunc.pInfo.size(); ++n_p ) {
        if ( fitFunc.pInfo[n_p].constrained ) {
            constrained = true;
            constrains_lm_lb[n_p] = fitFunc.pInfo[n_p].constr_lb;
            constrains_lm_ub[n_p] = fitFunc.pInfo[n_p].constr_ub;
        } else {
            constrains_lm_lb[n_p] = -DBL_MAX;
            constrains_lm_ub[n_p] = DBL_MAX;
        }
        if ( can_scale ) {
            if (fitFunc.pInfo[n_p].scale == stf::noscale) {
                can_scale = false;
            }
        }
    }

    // Store the functions at global scope:
    saveFunc(fitFunc.func);
    saveJac(fitFunc.jac);

    double info_id[LM_INFO_SZ];
    Vector_double data_ptr(data);
    Vector_double xyscale(4);
    if (can_scale) {
        xyscale = get_scale(data_ptr, dt);
    }
    
    // The parameters need to be separated into two parts:
    // Those that are to be fitted and those that the client wants
    // to keep constant. Since there is no native support to
    // do so in Lourakis' routines, the workaround is a little
    // tricky, making (ab)use of the *void pointer:

    // number of parameters that need to be fitted:
    int n_fitted=0;
    for ( unsigned n_p=0; n_p < fitFunc.pInfo.size(); ++n_p ) {
        n_fitted += fitFunc.pInfo[n_p].toFit;
    }
    // parameters that need to be fitted:
    Vector_double p_toFit(n_fitted);
    std::deque<bool> p_fit_bool( fitFunc.pInfo.size() );
    // parameters that are held constant:
    Vector_double p_const( fitFunc.pInfo.size()-n_fitted );
    for ( unsigned n_p=0, n_c=0, n_f=0; n_p < fitFunc.pInfo.size(); ++n_p ) {
        if (fitFunc.pInfo[n_p].toFit) {
            p_toFit[n_f++] = p[n_p];
            if (can_scale) {
                p_toFit[n_f-1] = fitFunc.pInfo[n_p].scale(p_toFit[n_f-1], xyscale[0],
                                                          xyscale[1], xyscale[2], xyscale[3]);
            }
        } else {
            p_const[n_c++] = p[n_p];
            if (can_scale) {
                p_const[n_c-1] = fitFunc.pInfo[n_p].scale(p_const[n_c-1], xyscale[0],
                                                          xyscale[1], xyscale[2], xyscale[3]);
            }
        }
        p_fit_bool[n_p] = fitFunc.pInfo[n_p].toFit;
    }
    // size * dt_new = 1 -> dt_new = 1.0/size
    double dt_finfo = dt;
    if (can_scale)
        dt_finfo = 1.0/data_ptr.size();

    fitInfo fInfo( p_fit_bool, p_const, dt_finfo );

    // make l-value of opts:
    Vector_double opts_l(5);
    for (std::size_t n=0; n < 4; ++n) opts_l[n] = opts[n];
    opts_l[4] = -1e-6;
    int it = 0;
    if (p_toFit.size()!=0 && data_ptr.size()!=0) {
        double old_info_id[LM_INFO_SZ];

        // initialize with initial parameter guess:
        Vector_double old_p_toFit(p_toFit);

#ifdef _DEBUG
        std::string optsMsg;
        optsMsg << wxT("\nopts: ");
        for (std::size_t n_p=0; n_p < opts.size(); ++n_p)
            optsMsg << opts[n_p] << wxT("\t");
        optsMsg << wxT("\n") << wxT("data_ptr[") << data_ptr.size()-1 << wxT("]=") << data_ptr[data_ptr.size()-1] << wxT("\n");
        optsMsg << wxT("constrains_lm_lb: "); 
        for (std::size_t n_p=0; n_p < constrains_lm_lb.size(); ++n_p) 
            optsMsg << constrains_lm_lb[n_p] << wxT("\t");
        optsMsg << wxT("\n") << wxT("constrains_lm_ub: "); 
        for (std::size_t n_p=0; n_p < constrains_lm_ub.size(); ++n_p) 
            optsMsg << constrains_lm_ub[n_p] << wxT("\t");
        optsMsg << wxT("\n\n");
        std::cout << optsMsg;
#endif

        while ( 1 ) {
#ifdef _DEBUG
            std::ostringstream paramMsg;
            paramMsg << wxT("Pass: ") << it << wxT("\t");
            paramMsg << wxT("p_toFit: ");
            for (std::size_t n_p=0; n_p < p_toFit.size(); ++n_p)
                paramMsg << p_toFit[n_p] << wxT("\t");
            paramMsg << wxT("\n");
            std::cout << paramMsg.str().c_str();
#endif

            if ( !fitFunc.hasJac ) {
                if ( !constrained ) {
                    dlevmar_dif( c_func_lour, &p_toFit[0], &data_ptr[0], n_fitted, 
                            (int)data.size(), (int)opts[4], &opts_l[0], info_id,
                            NULL, NULL, &fInfo );
                } else {
                    dlevmar_bc_dif( c_func_lour, &p_toFit[0], &data_ptr[0], n_fitted, 
                            (int)data.size(), &constrains_lm_lb[0], &constrains_lm_ub[0], NULL,
                            (int)opts[4], &opts_l[0], info_id, NULL, NULL, &fInfo );
                }
            } else {
                if ( !constrained ) {
                    dlevmar_der( c_func_lour, c_jac_lour, &p_toFit[0], &data_ptr[0], 
                            n_fitted, (int)data.size(), (int)opts[4], &opts_l[0], info_id,
                            NULL, NULL, &fInfo );                
                } else {
                    dlevmar_bc_der( c_func_lour,  c_jac_lour, &p_toFit[0], 
                            &data_ptr[0], n_fitted, (int)data.size(), &constrains_lm_lb[0], 
                            &constrains_lm_ub[0], NULL, (int)opts[4], &opts_l[0], info_id,
                            NULL, NULL, &fInfo );
                }
            }
            it++;
            if ( info_id[1] != info_id[1] ) {
                // restore previous parameters if new chisqr is NaN:
                p_toFit = old_p_toFit;
            } else {
                double dchisqr = (info_id[0] - info_id[1]) / info_id[1]; // (old chisqr - new chisqr) / new_chisqr
            
                if ( dchisqr < 0 ) {
                    // restore previous results and exit if new chisqr is larger:
                    for ( int n_i = 0; n_i < LM_INFO_SZ; ++n_i )  info_id[n_i] = old_info_id[n_i];
                    p_toFit = old_p_toFit;
                    break;
                }
                if ( dchisqr < 1e-5 ) {
                    // Keep current results and exit if change in chisqr is below threshold
                    break;
                }
                // otherwise, store results and continue iterating:
                for ( int n_i = 0; n_i < LM_INFO_SZ; ++n_i ) old_info_id[n_i] = info_id[n_i];
                old_p_toFit = p_toFit;
            }
            if ( it >= opts[5] )
                // Exit if maximal number of iterations is reached
                break;
            // decrease initial step size for next iteration:
            opts_l[0] *= 1e-4;
        }
    } else {
        std::runtime_error e("Array of size zero in lmFit");
        throw e;
    }

    // copy back the fitted parameters to p:
    for ( unsigned n_p=0, n_f=0, n_c=0; n_p<fitFunc.pInfo.size(); ++n_p ) {
        if (fitFunc.pInfo[n_p].toFit) {
            p[n_p] = p_toFit[n_f++];
        } else {
            p[n_p] = p_const[n_c++];
        }
        if (can_scale) {
            p[n_p] = fitFunc.pInfo[n_p].unscale(p[n_p], xyscale[0],
                                                xyscale[1], xyscale[2], xyscale[3]);
        }
    }
    
    std::ostringstream str_info;
    str_info << "Passes: " << it;
    str_info << "\nIterations during last pass: " << info_id[5];
    str_info << "\nStopping reason during last pass:";
    switch ((int)info_id[6]) {
     case 1:
         str_info << "\nStopped by small gradient of squared error.";
         warning = 0;
         break;
     case 2:
         str_info << "\nStopped by small rel. parameter change.";
         warning = 0;
         break;
     case 3:
         str_info << "\nReached max. number of iterations. Restart\n"
                  << "with smarter initial parameters and / or with\n"
                  << "increased initial scaling factor and / or with\n"
                  << "increased max. number of iterations.";
         warning = 3;
         break;
     case 4:
         str_info << "\nSingular matrix. Restart from current parameters\n"
                  << "with increased initial scaling factor.";
         warning = 4;
         break;
     case 5:
         str_info << "\nNo further error reduction is possible.\n"
                  << "Restart with increased initial scaling factor.";
         warning = 5;
         break;
     case 6:
         str_info << "\nStopped by small squared error.";
         warning = 0;
         break;
     case 7:
         str_info << "\nStopped by invalid (i.e. NaN or Inf) \"func\" values.\n";
         str_info << "This is a user error.";
         warning = 7;
         break;
     default:
         str_info << "\nUnknown reason for stopping the fit.";
         warning = -1;
    }
    if (use_scaling && !can_scale) {
        str_info << "\nCouldn't use scaling because one or more "
                 << "of the parameters don't allow it.";
    }
    info=str_info.str();
    return info_id[1];
}

double stf::flin(double x, const Vector_double& p) { return p[0]*x + p[1]; }

//! Dummy function to be passed to stf::storedFunc for linear functions.
void stf::flin_init(const Vector_double& data, double base, double peak,
        double RTLoHI, double HalfWidth, double dt, Vector_double& pInit )
{ }

stf::storedFunc stf::initLinFunc() {
    std::vector< stf::parInfo > linParInfo(2);
    linParInfo[0] = stf::parInfo("Slope", true);
    linParInfo[1] = stf::parInfo("Y intersect", true);
    return stf::storedFunc("Linear function", linParInfo,
            stf::flin, stf::flin_init, stf::nojac, false, stf::defaultOutput);
}

 /* options for the implementation of the LM algorithm */
Vector_double stf::LM_default_opts() {

    Vector_double opts(6);
    //opts[0]=5*1E-3;   // initial \mu, default: 1E-03;
    opts[0]=1E-3;   // initial \mu, default: 1E-03;
    opts[1]=1E-17;    // stopping thr for ||J^T e||_inf, default: 1E-17;
    opts[2]=1E-17;    // stopping trh for ||Dp||_2, default: 1E-17;
    opts[3]=1E-32;    // stopping thr for ||e||_2, default: 1E-17;
    opts[4]=64;       // maximal number of iterations/pass, default: 64;
    opts[5]=16;       // maximal number of passes;
    
    return opts;
}

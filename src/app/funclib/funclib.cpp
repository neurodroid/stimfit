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

#include <float.h>

#include "./../../core/fitlib.h"
#include "./../../core/measlib.h"
#include "./funclib.h"

std::vector< stf::storedFunc > stf::GetFuncLib() {
    std::vector< stf::storedFunc > funcList;
    
    // Monoexponential function, free fit:
    std::vector<parInfo> parInfoMExp=getParInfoExp(1);
    funcList.push_back(stf::storedFunc(
                                       wxT("Monoexponential"),parInfoMExp,fexp,fexp_init,fexp_jac,true));

    // Monoexponential function, offset fixed to baseline:
    parInfoMExp[2].toFit=false;
    funcList.push_back(stf::storedFunc(
                                       wxT("Monoexponential, offset fixed to baseline"),parInfoMExp,fexp,fexp_init,fexp_jac,true));

    // Monoexponential function, starting with a delay, start fixed to baseline:
    std::vector<parInfo> parInfoMExpDe(4);
    parInfoMExpDe[0].toFit=false; parInfoMExpDe[0].desc=wxT("Baseline");
    parInfoMExpDe[1].toFit=true; parInfoMExpDe[1].desc=wxT("Delay");
    parInfoMExpDe[2].toFit=true; parInfoMExpDe[2].desc=wxT("tau");
    parInfoMExpDe[3].toFit=true; parInfoMExpDe[3].desc=wxT("Peak");
    funcList.push_back(stf::storedFunc(
                                       wxT("Monoexponential with delay, start fixed to baseline"),parInfoMExpDe,fexpde,fexpde_init,nojac,false));

    // Biexponential function, free fit:
    std::vector<parInfo> parInfoBExp=getParInfoExp(2);
    funcList.push_back(stf::storedFunc(
                                       wxT("Biexponential"),parInfoBExp,fexp,fexp_init,fexp_jac,true,outputWTau));

    // Biexponential function, offset fixed to baseline:
    parInfoBExp[4].toFit=false;
    funcList.push_back(stf::storedFunc(
                                       wxT("Biexponential, offset fixed to baseline"),parInfoBExp,fexp,fexp_init,fexp_jac,true,outputWTau));

    // Biexponential function, starting with a delay, start fixed to baseline:
    std::vector<parInfo> parInfoBExpDe(5);
    parInfoBExpDe[0].toFit=false; parInfoBExpDe[0].desc=wxT("Baseline");
    parInfoBExpDe[1].toFit=true;  parInfoBExpDe[1].desc=wxT("Delay");
    parInfoBExpDe[1].constrained = true; parInfoBExpDe[1].constr_lb = 0.0; parInfoBExpDe[1].constr_ub = DBL_MAX;
    parInfoBExpDe[2].toFit=true;  parInfoBExpDe[2].desc=wxT("tau1");
    parInfoBExpDe[3].toFit=true;  parInfoBExpDe[3].desc=wxT("Factor");
    parInfoBExpDe[4].toFit=true;  parInfoBExpDe[4].desc=wxT("tau2");
    funcList.push_back(stf::storedFunc(
                                       wxT("Biexponential with delay, start fixed to baseline, delay constrained to > 0"),parInfoBExpDe,fexpbde,fexpbde_init,nojac,false));

    // Triexponential function, free fit:
    std::vector<parInfo> parInfoTExp=getParInfoExp(3);
    funcList.push_back(stf::storedFunc(
                                       wxT("Triexponential"),parInfoTExp,fexp,fexp_init,fexp_jac,true,outputWTau));

    // Triexponential function, free fit, different initialization:
    funcList.push_back(stf::storedFunc(
                                       wxT("Triexponential, initialize for PSCs/PSPs"),parInfoTExp,fexp,fexp_init2,fexp_jac,true,outputWTau));

    // Triexponential function, offset fixed to baseline:
    parInfoTExp[6].toFit=false;
    funcList.push_back(stf::storedFunc(
                                       wxT("Triexponential, offset fixed to baseline"),parInfoTExp,fexp,fexp_init,fexp_jac,true,outputWTau));

    // Alpha function:
    std::vector<parInfo> parInfoAlpha(3);
    parInfoAlpha[0].toFit=true; parInfoAlpha[0].desc=wxT("Q");
    parInfoAlpha[1].toFit=true; parInfoAlpha[1].desc=wxT("rate");
    parInfoAlpha[2].toFit=true; parInfoAlpha[2].desc=wxT("Offset");
    funcList.push_back(stf::storedFunc(
                                       wxT("Alpha function"), parInfoAlpha,falpha,falpha_init,falpha_jac,true));

    // HH gNa function:
    std::vector<parInfo> parInfoHH(4);
    parInfoHH[0].toFit=true; parInfoHH[0].desc=wxT("gprime_na");
    parInfoHH[1].toFit=true; parInfoHH[1].desc=wxT("tau_m");
    parInfoHH[2].toFit=true; parInfoHH[2].desc=wxT("tau_h");
    parInfoHH[3].toFit=false; parInfoHH[3].desc=wxT("offset");
    funcList.push_back(stf::storedFunc(
                                       wxT("Hodgkin-Huxley g_Na function, offset fixed to baseline"), parInfoHH, fHH, fHH_init, nojac, false));

    // power of 1 gNa function:
    funcList.push_back(stf::storedFunc(
                                       wxT("power of 1 g_Na function, offset fixed to baseline"), parInfoHH, fgnabiexp, fgnabiexp_init, nojac, false));
    return funcList;
}

std::valarray<double> stf::nojac(double x, const std::valarray<double>& p) {
    return std::valarray<double>(0);
}

double stf::fexp(double x, const std::valarray<double>& p) {
    double sum=0.0;
    for (std::size_t n_p=0;n_p<p.size()-1;n_p+=2) {
        double e=exp(-x/p[n_p+1]);
        sum+=p[n_p]*e;
    }
    return sum+p[p.size()-1];
}

std::valarray<double> stf::fexp_jac(double x, const std::valarray<double>& p) {
    std::valarray<double> jac(p.size());
    for (std::size_t n_p=0;n_p<p.size()-1;n_p+=2) {
        double e=exp(-x/p[n_p+1]);
        jac[n_p]=e;
        jac[n_p+1]=p[n_p]*x*e/(p[n_p+1]*p[n_p+1]);
    }
    jac[p.size()-1]=1.0;
    return jac;
}

void stf::fexp_init(const std::valarray<double>& data, double base, double peak, double dt, std::valarray<double>& pInit ) {
    // Find out direction:
    bool increasing = data[0] < data[data.size()-1];
    double floor = (increasing ? (data.max()+1.0e-9) : (data.min()-1.0e-9));
    std::valarray<double> peeled(data-floor);
    if (increasing) peeled *= -1.0;
    peeled = log(peeled);

    // linear fit on log-transformed data:
    std::valarray<double> x(data.size());
    for (std::size_t n_x = 0; n_x < x.size(); ++n_x) {
        x[n_x] = (double)n_x * dt;
    }
    double m=0, c=0;
    stf::linFit(x,peeled,m,c);
    
    double tau_mean = -1.0 / m;

    int n_exp=(int)pInit.size()/2;

    // Distribute taus:
    for (int n_p = 0; n_p < (int)pInit.size()-2; n_p+=2) {
        int n_term = n_p/2 + 1;
        double frac = pow((double)n_term,3) / pow((((double)n_exp+1.0)/2.0),3);
        // Calculate tau and amplitude:
        pInit[n_p+1] = tau_mean * frac;
    }
    // Estimate amps:
    double amp_total = data[0]-data[data.size()-1];
    for (int n_p = 0; n_p < (int)pInit.size()-2; n_p+=2) {
        pInit[n_p] = amp_total / n_exp;
    }
    // offset:
    pInit[pInit.size()-1] = data[data.size()-1];

}

void stf::fexp_init2(const std::valarray<double>& data, double base, double peak, double dt, std::valarray<double>& pInit ) {
    int n_exp=(int)pInit.size()/2;
    for (std::size_t n_p=0;n_p<pInit.size()-1;n_p+=2) {
        // use inverse amplitude for last term:
        int sign=1;
        if ((int)n_p==n_exp*2-2) {
            sign=-1;
        }
        pInit[n_p]=(double)sign/(double)n_exp*fabs(peak-base);
        pInit[n_p+1]=1.0/((double)n_p+2.0)/((double)n_p+2.0)*(double)data.size()*dt;
    }
    pInit[pInit.size()-1]=peak;
}

double stf::fexpde(double x, const std::valarray<double>& p) {
    if (x<p[1]) {
        return p[0];
    } else {
        double e1=exp((p[1]-x)/p[2]);
        // normalize the amplitude so that the peak really is the peak:
        return (p[0]-p[3])*e1 + p[3];
    }
}

#if 0
std::valarray<double> stf::fexpde_jac(double x, const std::valarray<double>& p) {
    std::valarray<double> jac(4);
    if (x<p[3]) {
        jac[0]=1.0;
        jac[1]=0.0;
        jac[2]=0.0;
        jac[3]=0.0;
    } else {
        double e=exp((p[3]-x)/p[1]);
        jac[0]=e;
        jac[1]=(p[0]-p[2])*(p[3]-x)*(-1.0/(p[1]*p[1]))*e;
        jac[2]=-e+1.0;
        jac[3]=(p[0]-p[2])*(1.0/p[1])*e;
    }
    return jac;
}
#endif 

void stf::fexpde_init(const std::valarray<double>& data, double base, double peak, double dt, std::valarray<double>& pInit ) {
    // Find the peak position in data:
    double maxT;
    stf::peak( data, base, 0, data.size(), 1, stf::both, maxT );

    pInit[0]=base;
    pInit[1]=0.0;
    pInit[2]=0.5 * maxT * dt;
    pInit[3]=peak;
}

double stf::fexpbde(double x, const std::valarray<double>& p) {
    if (x<p[1]) {
        return p[0];
    } else {
        // double tpeak = p[4]*p[2]*log(p[4]/p[2])/(p[4]-p[2]);
        // double adjust = 1.0/((1.0-exp(-tpeak/p[4]))-(1.0-exp(-tpeak/p[2])));
        double e1=exp((p[1]-x)/p[2]);
        double e2=exp((p[1]-x)/p[4]);

        return p[3]*e1 - p[3]*e2 + p[0];
    }
}

#if 0
std::valarray<double> stf::fexpbde_jac(double x, const std::valarray<double>& p) {
    std::valarray<double> jac(5);
    if (x<p[1]) {
        jac[0]=1.0;
        jac[1]=0.0;
        jac[2]=0.0;
        jac[3]=0.0;
        jac[4]=0.0;
    } else {
        double tpeak = p[4]*p[2]*log(p[4]/p[2])/(p[4]-p[2]);
        double adjust = 1.0/((1.0-exp(-tpeak/p[4]))-(1.0-exp(-tpeak/p[2])));
        double e1=exp((p[1]-x)/p[2]);
        double e2=exp((p[1]-x)/p[4]);
        jac[0]=1.0;
        jac[1]=adjust*p[3]/p[2] * e1 - adjust*p[3]/p[4] * e2;
        jac[2]=adjust*p[3]*(p[1]-x)*(-1.0/(p[2]*p[2]))*e1;
        jac[3]=adjust*e1-adjust*e2;
        jac[4]=adjust*p[3]*(p[1]-x)*(1.0/(p[4]*p[4]))*e2;
    }
    return jac;
}
#endif

void stf::fexpbde_init(const std::valarray<double>& data, double base, double peak, double dt, std::valarray<double>& pInit ) {
    // Find the peak position in data:
    double maxT = stf::whereis( data, peak );
    // stf::peak( data, base, 0, data.size(), 1, stf::both, maxT );

    
    if ( maxT == 0 ) maxT = data.size() * 0.05;
    pInit[0]=base;
    pInit[1]=0.01;
    pInit[2]=3 * maxT * dt;
    pInit[4]=0.5 * maxT * dt;
    double tpeak = pInit[4]*pInit[2]*log(pInit[4]/pInit[2])/(pInit[4]-pInit[2]);
    double adjust = 1.0/((1.0-exp(-tpeak/pInit[4]))-(1.0-exp(-tpeak/pInit[2])));
    pInit[3]=adjust*(peak-base);
}

double stf::falpha(double x, const std::valarray<double>& p) {
    double e=exp(-p[1]*x);
    return p[0]*p[1]*p[1]*x*e+p[2]; 
}

std::valarray<double> stf::falpha_jac(double x, const std::valarray<double>& p) {
    std::valarray<double> jac(3);
    double e=exp(-p[1]*x);
    jac[0]=p[1]*p[1]*x*e;
    jac[1]=p[0]*x*p[1]*(2*e-x*p[1]*e);
    jac[2]=1.0;
    return jac;
}

void stf::falpha_init(const std::valarray<double>& data, double base, double peak, double dt, std::valarray<double>& pInit ) {
        pInit[0]=(peak-base)*data.size()*dt;
        pInit[1]=1.0/(data.size()*dt/20.0);
        pInit[2]=base;
}

double stf::fHH(double x, const std::valarray<double>& p) {
    // p[0]: gprime_na
    // p[1]: tau_m
    // p[2]: tau_h
    // p[3]: offset
    double e1 = exp(-x/p[1]);
    double e2 = exp(-x/p[2]);
    return p[0] * (1-e1)*(1-e1)*(1-e1) * e2 + p[3];
}

double stf::fgnabiexp(double x, const std::valarray<double>& p) {
    // p[0]: gprime_na
    // p[1]: tau_m
    // p[2]: tau_h
    // p[3]: offset
    double e1 = exp(-x/p[1]);
    double e2 = exp(-x/p[2]);
    return p[0] * (1-e1) * e2 + p[3];
}

void stf::fHH_init(const std::valarray<double>& data, double base, double peak, double dt, std::valarray<double>& pInit ) {
    // Find the peak position in data:
    double maxT = stf::whereis( data, peak );
    // stf::peak( data, base, 0, data.size(), 1, stf::both, maxT );
    
    if ( maxT == 0 ) maxT = data.size() * 0.05;
    // double tpeak = p[1]*log((3.0*p[2])/p[1]+1.0);

    // p[0]: gprime_na
    // p[1]: tau_m
    // p[2]: tau_h
    // p[3]: offset
    pInit[1]=0.5 * maxT * dt;
    pInit[2]=3 * maxT * dt;
    double norm = (27.0*pow(pInit[2],3)*exp(-(pInit[1]*log((3.0*pInit[2]+pInit[1])/pInit[1]))/pInit[2])) / 
                  (27.0*pow(pInit[2],3)+27.0*pInit[1]*pInit[2]*pInit[2]+9.0*pInit[1]*pInit[1]*pInit[2]+pow(pInit[1],3));
    pInit[0]=(peak-base)/norm;
    pInit[3]=base;
}

void stf::fgnabiexp_init(const std::valarray<double>& data, double base, double peak, double dt, std::valarray<double>& pInit ) {
    // Find the peak position in data:
    double maxT = stf::whereis( data, peak );
    // stf::peak( data, base, 0, data.size(), 1, stf::both, maxT );
    
    if ( maxT == 0 ) maxT = data.size() * 0.05;
    // p[0]: gprime_na
    // p[1]: tau_m
    // p[2]: tau_h
    // p[3]: offset
    pInit[1]=0.5 * maxT * dt;
    pInit[2]=3 * maxT * dt;
    double tpeak = pInit[1]*log(pInit[2]/pInit[1]+1);
    double norm = (1-exp(-tpeak/pInit[1]))*exp(-tpeak/pInit[2]);
    pInit[0]=(peak-base)/norm;
    pInit[3]=base;
}

std::vector<stf::parInfo> stf::getParInfoExp(int n_exp) {
    std::vector<parInfo> retParInfo(n_exp*2+1);
    for (int n_e=0;n_e<n_exp*2;n_e+=2) {
        retParInfo[n_e].toFit=true;
        retParInfo[n_e+1].toFit=true;
        retParInfo[n_e].desc << wxT("Amp_") << (int)n_e/2;
        retParInfo[n_e+1].desc <<  wxT("Tau_") << (int)n_e/2;
    }
    retParInfo[n_exp*2].toFit=true;
    retParInfo[n_exp*2].desc=wxT("Offset");
    return retParInfo;
}

stf::Table stf::outputWTau(
    const std::valarray<double>& pars,
    const std::vector<parInfo>& parsInfo,
    double chisqr
) {
    Table output(pars.size()+1,1);
    // call default version:
    try  {
        output=defaultOutput(pars,parsInfo,chisqr);
    }
    catch (...) {
        throw;
    }
    // add weighted tau:
    // sum up amplitude terms:
    double sumAmp=0.0;
    for (std::size_t n_p=0;n_p<pars.size()-1;n_p+=2) {
        sumAmp+=pars[n_p];
    }
    // weight taus by their respective amplitudes:
    double sumTau=0.0;
    for (std::size_t n_p=0;n_p<pars.size()-1;n_p+=2) {
        sumTau+=(pars[n_p]/sumAmp)*pars[n_p+1];
    }
    // print:
    output.AppendRows(1);
    try {
        output.SetRowLabel(pars.size()+1, wxT("Weighted tau"));
        output.at(pars.size()+1,0)=sumTau;
    }
    catch (...) {
        throw;
    }
    return output;
}

std::size_t stf::whereis(const std::valarray<double>& data, double value) {
    if (data.size()==0) return 0;
    bool fromtop=false;
    // coming from top or bottom?
    if (data[0]>value) {
        fromtop=true;
    }
    for (std::size_t n=0;n<data.size();++n) {
        if (fromtop) {
            if (data[n] <= value ) {
                return n;
            }
        } else {
            if (data[n] >= value) {
                return n;
            }
        }
    }
    return 0;
}

stf::Table stf::defaultOutput(
	const std::valarray<double>& pars,
	const std::vector<stf::parInfo>& parsInfo,
    double chisqr
) {
	if (pars.size()!=parsInfo.size()) {
		throw std::out_of_range("index out of range in stf::defaultOutput");
	}
	Table output(pars.size()+1,1);
	try {
		output.SetColLabel(0,wxT("Best-fit value"));
		for (std::size_t n_p=0;n_p<pars.size(); ++n_p) {
			output.SetRowLabel(n_p,parsInfo[n_p].desc);
			output.at(n_p,0)=pars[n_p];
		}
        output.SetRowLabel(pars.size(),wxT("SSE"));
        output.at(pars.size(),0)=chisqr;
	}
	catch (...) {
		throw;
	}
	return output;
}


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

// core.cpp
// Some definitions of functions declared in the stf:: namespace
// last revision: 07-23-2006
// C. Schmidt-Hieber

#include <cmath>
#include <limits>

#ifndef _STIMFIT_H_
#include "core.h"
#include "./filelib/asciilib.h"
#include "./filelib/cfslib.h"
#include "./filelib/hdf5lib.h"
#include "./filelib/abflib.h"
#include "./filelib/atflib.h"
#include "./filelib/axglib.h"
#include "./filelib/hekalib.h"
#if 0
#include "./filelib/sonlib.h"
#endif
#endif

stf::filetype
stf::findType(const wxString& ext) {
    if (ext==wxT("*.dat;*.cfs")) return stf::cfs;
    else if (ext==wxT("*.abf")) return stf::abf;
    else if (ext==wxT("*.axgd;*.axgx")) return stf::axg;
    else if (ext==wxT("*.h5")) return stf::hdf5;
    else if (ext==wxT("*.atf")) return stf::atf;
    else if (ext==wxT("*.dat")) return stf::heka;
    else if (ext==wxT("*.smr")) return stf::son;
    else return stf::ascii;
}

bool stf::importFile(
        const wxString& fName,
        stf::filetype type,
        Recording& ReturnData,
        const stf::txtImportSettings& txtImport,
        bool progress,
        wxWindow* parent
) {
    try {
        switch (type) {
        case stf::cfs: {
            stf::importCFSFile(fName, ReturnData, progress);
            break;
        }
        case stf::hdf5: {
            stf::importHDF5File(fName, ReturnData, progress);
            break;
        }
        case stf::abf: {
            stf::importABFFile(fName, ReturnData, progress);
            break;
        }
        case stf::atf: {
            stf::importATFFile(fName, ReturnData, progress);
            break;
        }
        case stf::axg: {
            stf::importAXGFile(fName, ReturnData, progress, parent);
            break;
        }
        case stf::heka: {
            stf::importHEKAFile(fName, ReturnData, progress);
            break;
        }
#ifndef MODULE_ONLY
#if 0
        case stf::son: {
            stf::SON::importSONFile(fName,ReturnData);
            break;
        }
#endif
        case stf::ascii: {
            stf::importASCIIFile( fName, txtImport.hLines, txtImport.ncolumns,
                    txtImport.firstIsTime, txtImport.toSection, ReturnData );
            if (!txtImport.firstIsTime) {
                ReturnData.SetXScale(1.0/txtImport.sr);
            }
            if (ReturnData.size()>0)
                ReturnData[0].SetYUnits(txtImport.yUnits);
            if (ReturnData.size()>1)
                ReturnData[1].SetYUnits(txtImport.yUnitsCh2);
            ReturnData.SetXUnits( txtImport.xUnits );
            break;
        }
#endif
        default:
            throw std::runtime_error("Unknown file type");
        }
    }
    catch (...) {
        throw;
    }
    return true;
}

bool stf::exportFile(const wxString& fName, stf::filetype type, const Recording& Data,
                     bool progress)
{
    try {
        switch (type) {
        case stf::hdf5: {
            stf::exportHDF5File(fName, Data);
            break;
        }
        default:
            throw std::runtime_error("Only hdf5 is supported for writing at present.");
        }
    }
    catch (...) {
        throw;
    }
    return true;
}

    Vector_double stf::vec_scal_plus(const Vector_double& vec, double scalar) {
        Vector_double ret_vec(vec.size(), scalar);
        std::transform(vec.begin(), vec.end(), ret_vec.begin(), ret_vec.begin(), std::plus<double>());
        return ret_vec;
    }

    Vector_double stf::vec_scal_minus(const Vector_double& vec, double scalar) {
        Vector_double ret_vec(vec.size(), scalar);
        std::transform(vec.begin(), vec.end(), ret_vec.begin(), ret_vec.begin(), std::minus<double>());
        return ret_vec;
    }

    Vector_double stf::vec_scal_mul(const Vector_double& vec, double scalar) {
        Vector_double ret_vec(vec.size(), scalar);
        std::transform(vec.begin(), vec.end(), ret_vec.begin(), ret_vec.begin(), std::multiplies<double>());
        return ret_vec;
    }

    Vector_double stf::vec_scal_div(const Vector_double& vec, double scalar) {
        Vector_double ret_vec(vec.size(), scalar);
        std::transform(vec.begin(), vec.end(), ret_vec.begin(), ret_vec.begin(), std::divides<double>());
        return ret_vec;
    }

    Vector_double stf::vec_vec_plus(const Vector_double& vec1, const Vector_double& vec2) {
        Vector_double ret_vec(vec1.size());
        std::transform(vec1.begin(), vec1.end(), vec2.begin(), ret_vec.begin(), std::plus<double>());
        return ret_vec;
    }

    Vector_double stf::vec_vec_minus(const Vector_double& vec1, const Vector_double& vec2) {
        Vector_double ret_vec(vec1.size());
        std::transform(vec1.begin(), vec1.end(), vec2.begin(), ret_vec.begin(), std::minus<double>());
        return ret_vec;
    }

    Vector_double stf::vec_vec_mul(const Vector_double& vec1, const Vector_double& vec2) {
        Vector_double ret_vec(vec1.size());
        std::transform(vec1.begin(), vec1.end(), vec2.begin(), ret_vec.begin(), std::multiplies<double>());
        return ret_vec;
    }

    Vector_double stf::vec_vec_div(const Vector_double& vec1, const Vector_double& vec2) {
        Vector_double ret_vec(vec1.size());
        std::transform(vec1.begin(), vec1.end(), vec2.begin(), ret_vec.begin(), std::divides<double>());
        return ret_vec;
    }


#ifndef MODULE_ONLY
#include <wx/wxprec.h>
#include <wx/progdlg.h>
#include <wx/filename.h>

wxString stf::noPath(const wxString& fName) {
    return wxFileName(fName).GetFullName();
}

// LU decomposition from lapack
#ifdef __cplusplus
extern "C" {
#endif
    extern int dgetrs_(char *trans, int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
    extern int dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
#ifdef __cplusplus
}
#endif

double stf::fgauss(double x, const Vector_double& pars) {
    double y=0.0, fac=0.0, ex=0.0, arg=0.0;
    int npars=static_cast<int>(pars.size());
    for (int i=0; i < npars-1; i += 3) {
        arg=(x-pars[i+1])/pars[i+2];
        ex=exp(-arg*arg);
        fac=pars[i]*ex*2.0*arg;
        y += pars[i] * ex;
    }
    return y;
}

double stf::fboltz(double x, const Vector_double& pars) {
    double arg=(pars[0]-x)/pars[1];
    double ex=exp(arg);
    return 1/(1+ex);
}

double stf::fbessel(double x, int n) {
    double sum=0.0;
    for (int k=0;k<=n;++k) {
        int fac1=stf::fac(2*n-k);
        int fac2=stf::fac(n-k);
        int fac3=stf::fac(k);
        sum+=fac1/(fac2*fac3)*pow(x,k)/pow2(n-k);
    }
    return sum;
}

double stf::fbessel4(double x, const Vector_double& pars) {
    // normalize so that attenuation is -3dB at cutoff:
    return fbessel(0,4)/fbessel(x*0.355589/pars[0],4);
}

double stf::fgaussColqu(double x, const Vector_double& pars) {
    return exp(-0.3466*(x/pars[0])*(x/pars[0]));
}

int stf::fac(int arg) {
    if (arg<=1) {
        return 1;
    } else {
        return arg*fac(arg-1);
    }
}

#ifndef TEST_MINIMAL
Vector_double
stf::filter( const Vector_double& data, std::size_t filter_start,
        std::size_t filter_end, const Vector_double &a, int SR,
        Func func, bool inverse ) {
    if (data.size()<=0 || filter_start>=data.size() || filter_end > data.size()) {
        std::out_of_range e("subscript out of range in stf::filter()");
        throw e;
    }
    std::size_t filter_size=filter_end-filter_start+1;
    Vector_double data_return(filter_size);
    double SI=1.0/SR; //the sampling interval

    double *in;
    //fftw_complex is a double[2]; hence, out is an array of
    //double[2] with out[n][0] being the real and out[n][1] being
    //the imaginary part.
    fftw_complex *out;
    fftw_plan p1, p2;

    //memory allocation as suggested by fftw:
    in =(double *)fftw_malloc(sizeof(double) * filter_size);
    out=(fftw_complex *)fftw_malloc(sizeof(fftw_complex) * ((int)(filter_size/2)+1));

    // calculate the offset (a straight line between the first and last points):
    double offset_0=data[filter_start];
    double offset_1=data[filter_end]-offset_0;
    double offset_step=offset_1 / (filter_size-1);

    //fill the input array with data removing the offset:
    for (std::size_t n_point=0;n_point<filter_size;++n_point) {
        in[n_point]=data[n_point+filter_start]-(offset_0 + offset_step*n_point);
    }

    //plan the fft and execute it:
    p1 =fftw_plan_dft_r2c_1d((int)filter_size,in,out,FFTW_ESTIMATE);
    fftw_execute(p1);

    for (std::size_t n_point=0; n_point < (unsigned int)(filter_size/2)+1; ++n_point) {
        //calculate the frequency (in kHz) which corresponds to the index:
        double f=n_point / (filter_size*SI);
        double rslt= (!inverse? func(f,a) : 1.0-func(f,a));
        out[n_point][0] *= rslt;
        out[n_point][1] *= rslt;
    }

    //do the reverse fft:
    p2=fftw_plan_dft_c2r_1d((int)filter_size,out,in,FFTW_ESTIMATE);
    fftw_execute(p2);

    //fill the return array, adding the offset, and scaling by filter_size
    //(because fftw computes an unnormalized transform):
    data_return.resize(filter_size);
    for (std::size_t n_point=0; n_point < filter_size; ++n_point) {
        data_return[n_point]=(in[n_point]/filter_size + offset_0 + offset_step*n_point);
    }
    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
    fftw_free(in);fftw_free(out);
    return data_return;
}

Vector_double
stf::spectrum(
        const std::vector<std::complex<double> >& data,
        long K,
        double& f_n
) {
    // Variable names according to:
    // Welch, P.D. (1967). IEEE Transaction on Audio and Electroacoustics 15(2):70-37

    // First, perform "small" spectrum estimates of the
    // segments. Therefore, we need to split the original
    // data into equal-sized, overlapping segments. The overlap
    // should be half of the segment's size.

    // Check size of array:
    if (data.size()==0) {
        throw std::runtime_error("Exception:\nArray of size 0 in stf::spectrum");
    }
    if (K<=0) {
        throw std::runtime_error("Exception:\nNumber of segments <=0 in stf::spectrum");
    }
    double step_size=(double)data.size()/(double)(K+1);
    // Segment size:
    long L=stf::round(step_size*2.0);
    if (L<=0) {
        throw std::runtime_error("Exception:\nSegment size <=0 in stf::spectrum");
    }
    long spec_size=long(L/2)+1;
    double offset=0.0;
    fftw_complex* X=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*L);
    fftw_complex* A=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*L);
    // plan the fft once:
    fftw_plan p1=fftw_plan_dft_1d(L,X,A,FFTW_FORWARD,FFTW_ESTIMATE);
    Vector_double P(spec_size, 0.0);

    // Window function summed, squared and normalized:
    double U=0.0;
    for (long j=0;j<L;++j) {
        U+=SQR(stf::window(j,L));
    }
    // This should be normalized by L; however,
    // this can be omitted due to the later multi-
    // plication with 1/(L*U), which will then get 1/U.

    for (long k=0;k<K;++k) {
        // Fill the segment, applying the window function:
        for (long j=0;j<L;++j) {
            X[j][0]=data[(long)offset+j].real()*window(j,L);
            X[j][1]=data[(long)offset+j].imag()*window(j,L);
        }

        // Transform the data:
        fftw_execute(p1);
        // Instead of normalizing A right here, we will do this after summing up.

        // Add segment periodogram to spectrum (the intermediate variable I
        // from Welch's paper is not needed because we add the periodograms
        // directly to P here).
        // Treat the 0-component separately (because there is no corresponding
        // negative part):
        P[0]+=SQR(A[0][0])+SQR(A[0][1]);
        for (long i_out=1;i_out<spec_size;++i_out) {
            // Add corresponding negative and positive frequencies to
            // the same position in the spectrum:
            P[i_out]+=(SQR(A[i_out][0])+SQR(A[i_out][1])+
                    SQR(A[L-i_out][0])+SQR(A[L-i_out][1]));
            // This should be multiplied by L/U at every step; however,
            // we can do this once all values have been summed up.
        }
        // If this is the second-last loop, calculate the offset from the end:
        if (k!=K-2) {
            offset+=step_size;
        } else {
            offset=data.size()-L;
        }
    }
    // Do the multiplication and the normalization that we omitted above:
    P = stf::vec_scal_div(P,U);
    // Average:
    P = stf::vec_scal_div(P,K);

    // Use FFTW's deallocation routines:
    fftw_destroy_plan(p1);
    fftw_free(X);fftw_free(A);
    // frequency stepsize of P:
    f_n=1.0/L;
    return P;
}
#endif

Vector_double
stf::detectionCriterion(const Vector_double& data, const Vector_double& templ)
{
    wxProgressDialog progDlg( wxT("Template matching"), wxT("Starting template matching"),
            100, NULL, wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL | wxPD_CAN_SKIP );
    bool skipped=false;
    // variable names are taken from Clements & Bekkers (1997) as long
    // as they don't interfere with C++ keywords (such as "template")
    Vector_double detection_criterion(data.size()-templ.size());
    // avoid redundant computations:
    double sum_templ_data=0.0, sum_templ=0.0, sum_templ_sqr=0.0, sum_data=0.0, sum_data_sqr=0.0;
    for (int n_templ=0; n_templ<(int)templ.size();++n_templ) {
        sum_templ_data+=templ[n_templ]*data[0+n_templ];
        sum_data+=data[0+n_templ];
        sum_data_sqr+=data[0+n_templ]*data[0+n_templ];
        sum_templ+=templ[n_templ];
        sum_templ_sqr+=templ[n_templ]*templ[n_templ];
    }
    double y_old=0.0;
    double y2_old=0.0;
    int progCounter=0;
    double progFraction=(data.size()-templ.size())/100;
    for (unsigned n_data=0; n_data<data.size()-templ.size(); ++n_data) {
        if (n_data/progFraction>progCounter) {
            progDlg.Update( (int)((double)n_data/(double)(data.size()-templ.size())*100.0),
                    wxT("Calculating detection criterion"), &skipped );
            if (skipped) {
                detection_criterion.resize(0);
                return detection_criterion;
            }
            progCounter++;
        }
        if (n_data!=0) {
            sum_templ_data=0.0;
            // The product has to be computed in full length:
            for (int n_templ=0; n_templ<(int)templ.size();++n_templ) {
                sum_templ_data+=templ[n_templ]*data[n_data+n_templ];
            }
            // The new value that will be added is:
            double y_new=data[n_data+templ.size()-1];
            double y2_new=data[n_data+templ.size()-1]*data[n_data+templ.size()-1];
            sum_data+=y_new-y_old;
            sum_data_sqr+=y2_new-y2_old;
        }
        // The first value that was added (and will have to be subtracted during
        // the next loop):
        y_old=data[n_data+0];
        y2_old=data[n_data+0]*data[n_data+0];

        double scale=(sum_templ_data-sum_templ*sum_data/templ.size())/
        (sum_templ_sqr-sum_templ*sum_templ/templ.size());
        double offset=(sum_data-scale*sum_templ)/templ.size();
        double sse=sum_data_sqr+scale*scale*sum_templ_sqr+templ.size()*offset*offset
        -2.0*(scale*sum_templ_data
                +offset*sum_data-scale*offset*sum_templ);
        double standard_error=sqrt(sse/(templ.size()-1));
        detection_criterion[n_data]=(scale/standard_error);
    }
    return detection_criterion;
}

std::vector<int>
stf::peakIndices(const Vector_double& data,
        double threshold,
        int minDistance)
{
    // to avoid unnecessary copying, we first reserve quite
    // a bit of space for the vector:
    std::vector<int> peakInd;
    peakInd.reserve(data.size());
    for (unsigned n_data=0; n_data<data.size(); ++n_data) {
        // check whether the data point is above threshold...
        int llp=n_data;
        int ulp=n_data+1;
        if (data[n_data]>threshold) {
            // ... and if so, find the data point where the threshold
            // is crossed again in the opposite direction, ...
            for (;;) {
                if (n_data>data.size()-1) {
                    ulp=(int)data.size()-1;
                    break;
                }
                n_data++;
                if (data[n_data]<threshold && (int)n_data-ulp>minDistance) {
                    // ... making this the upper limit of the peak window:
                    ulp=n_data;
                    break;
                }
            }
            // Now, find the peak within the window:
            double max=-1e8;
            int peakIndex=llp;
            for (int n_p=llp; n_p<=ulp; ++n_p) {
                if (data[n_p]>max) {
                    max=data[n_p];
                    peakIndex=n_p;
                }
            }
            peakInd.push_back(peakIndex);
        }
    }
    // Trim peakInd's reserved memory:
    std::vector<int>(peakInd.begin(),peakInd.end()).swap(peakInd);
    return peakInd;
}

Vector_double
stf::linCorr(const Vector_double& data, const Vector_double& templ)
{
    wxProgressDialog progDlg( wxT("Template matching"), wxT("Starting template matching"),
            100, NULL, wxPD_SMOOTH | wxPD_AUTO_HIDE | wxPD_APP_MODAL | wxPD_CAN_SKIP );
    bool skipped = false;
    // the template has to be smaller than the data waveform:
    if (data.size()<templ.size()) {
        throw std::runtime_error("Template larger than data in stf::crossCorr");
    }
    if (data.size()==0 || templ.size()==0) {
        throw std::runtime_error("Array of size 0 in stf::crossCorr");
    }
    Vector_double Corr(data.size()-templ.size());

    // Optimal scaling & offset:
    // avoid redundant computations:
    double sum_templ_data=0.0, sum_templ=0.0, sum_templ_sqr=0.0, sum_data=0.0, sum_data_sqr=0.0;
    for (int n_templ=0; n_templ<(int)templ.size();++n_templ) {
        sum_templ_data+=templ[n_templ]*data[0+n_templ];
        sum_data+=data[0+n_templ];
        sum_data_sqr+=data[0+n_templ]*data[0+n_templ];
        sum_templ+=templ[n_templ];
        sum_templ_sqr+=templ[n_templ]*templ[n_templ];
    }
    double y_old=0.0;
    double y2_old=0.0;
    int progCounter=0;
    double progFraction=(data.size()-templ.size())/100;
    for (unsigned n_data=0; n_data<data.size()-templ.size(); ++n_data) {
        if (n_data/progFraction>progCounter) {
            progDlg.Update( (int)((double)n_data/(double)(data.size()-templ.size())*100.0),
                    wxT("Calculating correlation coefficient"), &skipped );
            if (skipped) {
                Corr.resize(0);
                return Corr;
            }
            progCounter++;
        }
        if (n_data!=0) {
            sum_templ_data=0.0;
            // The product has to be computed in full length:
            for (int n_templ=0; n_templ<(int)templ.size();++n_templ) {
                sum_templ_data+=templ[n_templ]*data[n_data+n_templ];
            }
            // The new value that will be added is:
            double y_new=data[n_data+templ.size()-1];
            double y2_new=data[n_data+templ.size()-1]*data[n_data+templ.size()-1];
            sum_data+=y_new-y_old;
            sum_data_sqr+=y2_new-y2_old;
        }
        // The first value that was added (and will have to be subtracted during
        // the next loop):
        y_old=data[n_data+0];
        y2_old=data[n_data+0]*data[n_data+0];

        double scale=(sum_templ_data-sum_templ*sum_data/templ.size())/
        (sum_templ_sqr-sum_templ*sum_templ/templ.size());
        double offset=(sum_data-scale*sum_templ)/templ.size();

        // Now that the optimal template has been found,
        // compute the correlation between data and optimal template.
        // The correlation coefficient is computed in a way that avoids
        // numerical instability; therefore, the sum of squares
        // computed above can't be re-used.
        // Get the means:
        double mean_data=sum_data/templ.size();
        double sum_optTempl=sum_templ*scale+offset*templ.size();
        double mean_optTempl=sum_optTempl/templ.size();

        // Get SDs:
        double sd_data=0.0;
        double sd_templ=0.0;
        for (int i=0;i<(int)templ.size();++i) {
            sd_data+=SQR(data[i+n_data]-mean_data);
            sd_templ+=SQR(templ[i]*scale+offset-mean_optTempl);
        }
        sd_data=sqrt(sd_data/templ.size());
        sd_templ=sqrt(sd_templ/templ.size());

        // Get correlation:
        double r=0.0;
        for (int i=0;i<(int)templ.size();++i) {
            r+=(data[i+n_data]-mean_data)*(templ[i]*scale+offset-mean_optTempl);
        }
        r/=((templ.size()-1)*sd_data*sd_templ);
        Corr[n_data]=r;
    }
    return Corr;
}

wxString stf::sectionToString(const Section& section) {
    wxString retString;
    retString << (int)section.size() << wxT("\n");
    for (int n=0;n<(int)section.size();++n) {
        retString << section.GetXScale()*n << wxT("\t") << section[n] << wxT("\n");
    }
    return retString;
}

wxString stf::CreatePreview(const wxString& fName) {
    ifstreamMan ASCIIfile( fName );
    // Stop reading if we are either at the end or at line 100:
    wxString preview;
	ASCIIfile.myStream.ReadAll( &preview );
    return preview;
}

double stf::integrate_simpson(
        const Vector_double& input,
        std::size_t i1,
        std::size_t i2,
        double x_scale
) {

    // Use composite Simpson's rule to approximate the definite integral of f from a to b
    // check for out-of-range:
    if (i2>=input.size() || i1>=i2) {
        throw std::out_of_range( "integration interval out of range in stf::integrate_simpson" );
    }
    bool even = std::div((int)i2-(int)i1,2).rem==0;

    // use Simpson's rule for the even part:
    if (!even)
        i2--;
    std::size_t n=i2-i1;
    double a=i1*x_scale;
    double b=i2*x_scale;

    double sum_2=0.0, sum_4=0.0;
    for (std::size_t j = 1; j <= n/2; ++j) {
        if (j<n/2)
            sum_2+=input[i1+2*j];
        sum_4+=input[i1+2*j-1];
    }
    double sum=input[i1] + 2*sum_2 + 4*sum_4 + input[i2];
    sum *= (b-a)/(double)n;
    sum /= 3;

    // if uneven, add the last interval by trapezoidal integration:
    if (!even) {
        i2++;
        a = (i2-1)*x_scale;
        b = i2*x_scale;
        sum += (b-a)/2 * (input[i2]+input[i2-1]);
    }
    return sum;
}

double stf::integrate_trapezium(
        const Vector_double& input,
        std::size_t i1,
        std::size_t i2,
        double x_scale
) {
    if (i2>=input.size() || i1>=i2) {
        throw std::out_of_range( "integration interval out of range in stf::integrate_simpson" );
    }
    double a = i1 * x_scale;
    double b = i2 * x_scale;

    double sum=input[i1]+input[i2];
    for (std::size_t n=i1+1; n<i2; ++n) {
        sum += 2*input[n];
    }
    sum *= (b-a)/2/(i2-i1);
    return sum;
}

int
stf::linsolv( int m, int n, int nrhs, Vector_double& A,
        Vector_double& B)
{
#ifndef TEST_MINIMAL
    if (A.size()<=0) {
        throw std::runtime_error("Matrix A has size 0 in stf::linsolv");
    }

    if (B.size()<=0) {
        throw std::runtime_error("Matrix B has size 0 in stf::linsolv");
    }

    if (A.size()!= std::size_t(m*n)) {
        throw std::runtime_error("Size of matrix A is not m*n");
    }

    /* Arguments to dgetrf_
     *  ====================
     *
     *  M       (input) INTEGER
     *          The number of rows of the matrix A.  M >= 0.
     *
     *  N       (input) INTEGER
     *          The number of columns of the matrix A.  N >= 0.
     *
     *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
     *          On entry, the M-by-N matrix to be factored.
     *          On exit, the factors L and U from the factorization
     *          A = P*L*U; the unit diagonal elements of L are not stored.
     *
     *  LDA     (input) INTEGER
     *          The leading dimension of the array A.  LDA >= max(1,M).
     *
     *  IPIV    (output) INTEGER array, dimension (min(M,N))
     *          The pivot indices; for 1 <= i <= min(M,N), row i of the
     *          matrix was interchanged with row IPIV(i).
     *
     *  INFO    (output) INTEGER
     *          = 0:  successful exit
     *          < 0:  if INFO = -i, the i-th argument had an illegal value
     *          > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
     *                has been completed, but the factor U is exactly
     *                singular, and division by zero will occur if it is used
     *                to solve a system of equations.
     */

    int lda_f = m;
    std::size_t ipiv_size = (m < n) ? m : n;
    std::vector<int> ipiv(ipiv_size);
    int info=0;

    dgetrf_(&m, &n, &A[0], &lda_f, &ipiv[0], &info);
    if (info<0) {
        wxString error_msg;
        error_msg << wxT("Argument ") << -info << wxT(" had an illegal value in LAPACK's dgetrf_");
		throw std::runtime_error( std::string(error_msg.char_str()));
    }
    if (info>0) {
        throw std::runtime_error("Singular matrix in LAPACK's dgetrf_; would result in division by zero");
    }


    /* Arguments to dgetrs_
     *  ====================
     *
     *  TRANS   (input) CHARACTER*1
     *          Specifies the form of the system of equations:
     *          = 'N':  A * X = B  (No transpose)
     *          = 'T':  A'* X = B  (Transpose)
     *          = 'C':  A'* X = B  (Conjugate transpose = Transpose)
     *
     *  N       (input) INTEGER
     *          The order of the matrix A.  N >= 0.
     *
     *  NRHS    (input) INTEGER
     *          The number of right hand sides, i.e., the number of columns
     *          of the matrix B.  NRHS >= 0.
     *
     *  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
     *          The factors L and U from the factorization A = P*L*U
     *          as computed by DGETRF.
     *
     *  LDA     (input) INTEGER
     *          The leading dimension of the array A.  LDA >= max(1,N).
     *
     *  IPIV    (input) INTEGER array, dimension (N)
     *          The pivot indices from DGETRF; for 1<=i<=N, row i of the
     *          matrix was interchanged with row IPIV(i).
     *
     *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
     *          On entry, the right hand side matrix B.
     *          On exit, the solution matrix X.
     *
     *  LDB     (input) INTEGER
     *          The leading dimension of the array B.  LDB >= max(1,N).
     *
     *  INFO    (output) INTEGER
     *          = 0:  successful exit
     *          < 0:  if INFO = -i, the i-th argument had an illegal value
     */
    char trans='N';
    dgetrs_(&trans, &m, &nrhs, &A[0], &m, &ipiv[0], &B[0], &m, &info);
    if (info<0) {
        wxString error_msg;
        error_msg << wxT("Argument ") << -info << wxT(" had an illegal value in LAPACK's dgetrs_");
        throw std::runtime_error(std::string(error_msg.char_str()));
    }
#endif
    return 0;
}

stf::Table::Table(std::size_t nRows,std::size_t nCols) :
values(nRows,std::vector<double>(nCols,1.0)),
    empty(nRows,std::deque<bool>(nCols,false)),
    rowLabels(nRows, wxT("\0")),
    colLabels(nCols, wxT("\0"))
    {}

stf::Table::Table(const std::map< wxString, double >& map)
: values(map.size(),std::vector<double>(1,1.0)), empty(map.size(),std::deque<bool>(1,false)),
rowLabels(map.size(),wxT("\0")), colLabels(1,wxT("Results"))
{
    std::map< wxString, double >::const_iterator cit;
    wxs_it it1 = rowLabels.begin();
    std::vector< std::vector<double> >::iterator it2 = values.begin();
    for (cit = map.begin();
         cit != map.end() && it1 != rowLabels.end() && it2 != values.end();
         cit++)
    {
        (*it1) = cit->first;
        it2->at(0) = cit->second;
        it1++;
        it2++;
    }
}

double stf::Table::at(std::size_t row,std::size_t col) const {
    try {
        return values.at(row).at(col);
    }
    catch (...) {
        throw;
    }
}

double& stf::Table::at(std::size_t row,std::size_t col) {
    try {
        return values.at(row).at(col);
    }
    catch (...) {
        throw;
    }
}

bool stf::Table::IsEmpty(std::size_t row,std::size_t col) const {
    try {
        return empty.at(row).at(col);
    }
    catch (...) {
        throw;
    }
}

void stf::Table::SetEmpty(std::size_t row,std::size_t col,bool value) {
    try {
        empty.at(row).at(col)=value;
    }
    catch (...) {
        throw;
    }
}

void stf::Table::SetRowLabel(std::size_t row,const wxString& label) {
    try {
        rowLabels.at(row)=label;
    }
    catch (...) {
        throw;
    }
}

void stf::Table::SetColLabel(std::size_t col,const wxString& label) {
    try {
        colLabels.at(col)=label;
    }
    catch (...) {
        throw;
    }
}

const wxString& stf::Table::GetRowLabel(std::size_t row) const {
    try {
        return rowLabels.at(row);
    }
    catch (...) {
        throw;
    }
}

const wxString& stf::Table::GetColLabel(std::size_t col) const {
    try {
        return colLabels.at(col);
    }
    catch (...) {
        throw;
    }
}

void stf::Table::AppendRows(std::size_t nRows_) {
    std::size_t oldRows=nRows();
    rowLabels.resize(oldRows+nRows_);
    values.resize(oldRows+nRows_);
    empty.resize(oldRows+nRows_);
    for (std::size_t nRow = 0; nRow < oldRows + nRows_; ++nRow) {
        values[nRow].resize(nCols());
        empty[nRow].resize(nCols());
    }
}

#endif

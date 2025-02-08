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
// Some definitions of functions declared in the stfnum:: namespace
// last revision: 07-23-2006
// C. Schmidt-Hieber

#include <cmath>
#include <limits>
#include <algorithm>

#include "stfnum.h"
#include "fit.h"
#include "funclib.h"

int isnan(double x) { return x != x; }
int isinf(double x) { return !isnan(x) && isnan(x - x); }

stfnum::Table::Table(std::size_t nRows,std::size_t nCols) :
values(nRows,std::vector<double>(nCols,1.0)),
    empty(nRows,std::deque<bool>(nCols,false)),
    rowLabels(nRows, "\0"),
    colLabels(nCols, "\0")
    {}

stfnum::Table::Table(const std::map< std::string, double >& map)
: values(map.size(),std::vector<double>(1,1.0)), empty(map.size(),std::deque<bool>(1,false)),
rowLabels(map.size(), "\0"), colLabels(1, "Results")
{
    std::map< std::string, double >::const_iterator cit;
    sst_it it1 = rowLabels.begin();
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

double stfnum::Table::at(std::size_t row,std::size_t col) const {
    try {
        return values.at(row).at(col);
    }
    catch (...) {
        throw;
    }
}

double& stfnum::Table::at(std::size_t row,std::size_t col) {
    try {
        return values.at(row).at(col);
    }
    catch (...) {
        throw;
    }
}

bool stfnum::Table::IsEmpty(std::size_t row,std::size_t col) const {
    try {
        return empty.at(row).at(col);
    }
    catch (...) {
        throw;
    }
}

void stfnum::Table::SetEmpty(std::size_t row,std::size_t col,bool value) {
    try {
        empty.at(row).at(col)=value;
    }
    catch (...) {
        throw;
    }
}

void stfnum::Table::SetRowLabel(std::size_t row,const std::string& label) {
    try {
        rowLabels.at(row)=label;
    }
    catch (...) {
        throw;
    }
}

void stfnum::Table::SetColLabel(std::size_t col,const std::string& label) {
    try {
        colLabels.at(col)=label;
    }
    catch (...) {
        throw;
    }
}

const std::string& stfnum::Table::GetRowLabel(std::size_t row) const {
    try {
        return rowLabels.at(row);
    }
    catch (...) {
        throw;
    }
}

const std::string& stfnum::Table::GetColLabel(std::size_t col) const {
    try {
        return colLabels.at(col);
    }
    catch (...) {
        throw;
    }
}

void stfnum::Table::AppendRows(std::size_t nRows_) {
    std::size_t oldRows=nRows();
    rowLabels.resize(oldRows+nRows_);
    values.resize(oldRows+nRows_);
    empty.resize(oldRows+nRows_);
    for (std::size_t nRow = 0; nRow < oldRows + nRows_; ++nRow) {
        values[nRow].resize(nCols());
        empty[nRow].resize(nCols());
    }
}

double stfnum::fboltz(double x, const Vector_double& pars) {
    double arg=(pars[0]-x)/pars[1];
    double ex=exp(arg);
    return 1/(1+ex);
}

double stfnum::fbessel(double x, int n) {
    double sum=0.0;
    for (int k=0;k<=n;++k) {
        int fac1=stfnum::fac(2*n-k);
        int fac2=stfnum::fac(n-k);
        int fac3=stfnum::fac(k);
        sum+=fac1/(fac2*fac3)*pow(x,k)/pow2(n-k);
    }
    return sum;
}

double stfnum::fbessel4(double x, const Vector_double& pars) {
    // normalize so that attenuation is -3dB at cutoff:
    return fbessel(0,4)/fbessel(x*0.355589/pars[0],4);
}

double stfnum::fgaussColqu(double x, const Vector_double& pars) {
    return exp(-0.3466*(x/pars[0])*(x/pars[0]));
}

int stfnum::fac(int arg) {
    if (arg<=1) {
        return 1;
    } else {
        return arg*fac(arg-1);
    }
}

Vector_double
stfnum::filter( const Vector_double& data, std::size_t filter_start,
        std::size_t filter_end, const Vector_double &a, int SR,
        stfnum::Func func, bool inverse ) {
    if (data.size()<=0 || filter_start>=data.size() || filter_end > data.size()) {
        std::out_of_range e("subscript out of range in stfnum::filter()");
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
stfnum::detectionCriterion(const Vector_double& data, const Vector_double& templ, stfio::ProgressInfo& progDlg)
{
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
    double progFraction=(data.size()-templ.size())/100.0;
    for (unsigned n_data=0; n_data<data.size()-templ.size(); ++n_data) {
        if (n_data/progFraction>progCounter) {
            progDlg.Update( (int)((double)n_data/(double)(data.size()-templ.size())*100.0),
                            "Calculating detection criterion", &skipped );
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
        double sse=sum_data_sqr+scale*scale*sum_templ_sqr+templ.size()*offset*offset -
            2.0*(scale*sum_templ_data +
                 offset*sum_data-scale*offset*sum_templ);
        double standard_error=sqrt(sse/(templ.size()-1));
        detection_criterion[n_data]=(scale/standard_error);
    }
    return detection_criterion;
}

std::vector<int>
stfnum::peakIndices(const Vector_double& data, double threshold,
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
                if (n_data>data.size()-2) {
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
stfnum::linCorr(const Vector_double& data, const Vector_double& templ, stfio::ProgressInfo& progDlg)
{
    bool skipped = false;
    // the template has to be smaller than the data waveform:
    if (data.size()<templ.size()) {
        throw std::runtime_error("Template larger than data in stfnum::crossCorr");
    }
    if (data.size()==0 || templ.size()==0) {
        throw std::runtime_error("Array of size 0 in stfnum::crossCorr");
    }
    Vector_double Corr(data.size()-templ.size());

    // Optimal scaling & offset:
    // avoid redundant computations:
    double sum_templ_data=0.0, sum_templ=0.0, sum_templ_sqr=0.0, sum_data=0.0;
    for (int n_templ=0; n_templ<(int)templ.size();++n_templ) {
        sum_templ_data+=templ[n_templ]*data[0+n_templ];
        sum_data+=data[0+n_templ];
        sum_templ+=templ[n_templ];
        sum_templ_sqr+=templ[n_templ]*templ[n_templ];
    }
    double y_old=0.0;
    int progCounter=0;
    double progFraction=(data.size()-templ.size())/100.0;
    for (unsigned n_data=0; n_data<data.size()-templ.size(); ++n_data) {
        if (n_data/progFraction>progCounter) {
            progDlg.Update( (int)((double)n_data/(double)(data.size()-templ.size())*100.0),
                            "Calculating correlation coefficient", &skipped );
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
            sum_data+=y_new-y_old;
        }
        // The first value that was added (and will have to be subtracted during
        // the next loop):
        y_old=data[n_data+0];

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

double stfnum::integrate_simpson(
        const Vector_double& input,
        std::size_t i1,
        std::size_t i2,
        double x_scale
) {

    // Use composite Simpson's rule to approximate the definite integral of f from a to b
    // check for out-of-range:
    if (i2>=input.size() || i1>=i2) {
        throw std::out_of_range( "integration interval out of range in stfnum::integrate_simpson" );
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

double stfnum::integrate_trapezium(
        const Vector_double& input,
        std::size_t i1,
        std::size_t i2,
        double x_scale
) {
    if (i2>=input.size() || i1>=i2) {
        throw std::out_of_range( "integration interval out of range in stfnum::integrate_trapezium" );
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

// LU decomposition from lapack
#ifdef __cplusplus
extern "C" {
#endif
    extern int dgetrs_(char *trans, int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
    extern int dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
#ifdef __cplusplus
}
#endif

int
stfnum::linsolv( int m, int n, int nrhs, Vector_double& A,
              Vector_double& B)
{
#ifndef TEST_MINIMAL
    if (A.size()<=0) {
        throw std::runtime_error("Matrix A has size 0 in stfnum::linsolv");
    }

    if (B.size()<=0) {
        throw std::runtime_error("Matrix B has size 0 in stfnum::linsolv");
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
        std::ostringstream error_msg;
        error_msg << "Argument " << -info << " had an illegal value in LAPACK's dgetrf_";
		throw std::runtime_error( std::string(error_msg.str()));
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
        std::ostringstream error_msg;
        error_msg << "Argument " << -info << " had an illegal value in LAPACK's dgetrs_";
        throw std::runtime_error(error_msg.str());
    }
#endif
    return 0;
}

Vector_double stfnum::quad(const Vector_double& data, std::size_t begin, std::size_t end) {

    // Solve quadratic equations relating 3 sample points a time
    
    int n_intervals=std::div((int)end-(int)begin,2).quot;
    
    Vector_double quad_p(n_intervals*3);
    
    int n_q=0;
    if (begin-end>1) {
        for (int n=begin; n<(int)end-1; n+=2) {
            Vector_double A(9);
            Vector_double B(3);
    
            // solve linear equation system:
            // use column-major order (Fortran)
            A[0]=(double)n*(double)n;
            A[1]=((double)n+1.0)*((double)n+1.0);
            A[2]=((double)n+2.0)*((double)n+2.0);
            A[3]=(double)n;
            A[4]=(double)n+1.0;
            A[5]=(double)n+2.0;
            A[6]=1.0;
            A[7]=1.0;
            A[8]=1.0;
            B[0]=data[n];
            B[1]=data[n+1];
            B[2]=data[n+2];
            try {
                stfnum::linsolv(3,3,1,A,B);
            }
            catch (...) {
                throw;
            }
            quad_p[n_q++]=B[0];
            quad_p[n_q++]=B[1];
            quad_p[n_q++]=B[2];
        }
    }
    return quad_p;
}

Vector_double stfnum::nojac(double x, const Vector_double& p) {
    return Vector_double(0);
}

double stfnum::noscale(double param, double xscale, double oldx, double yscale, double yoff) {
    return param;
}

stfnum::Table stfnum::defaultOutput(
	const Vector_double& pars,
	const std::vector<stfnum::parInfo>& parsInfo,
    double chisqr
) {
	if (pars.size()!=parsInfo.size()) {
		throw std::out_of_range("index out of range in stfnum::defaultOutput");
	}
        stfnum::Table output(pars.size()+1,1);
	try {
		output.SetColLabel(0,"Best-fit value");
		for (std::size_t n_p=0;n_p<pars.size(); ++n_p) {
			output.SetRowLabel(n_p,parsInfo[n_p].desc);
			output.at(n_p,0)=pars[n_p];
		}
        output.SetRowLabel(pars.size(),"SSE");
        output.at(pars.size(),0)=chisqr;
	}
	catch (...) {
		throw;
	}
	return output;
}

std::map<double, int>
stfnum::histogram(const Vector_double& data, int nbins) {

    if (nbins==-1) {
        nbins = int(data.size()/100.0);
    }

    double fmax = *std::max_element(data.begin(), data.end());
    double fmin = *std::min_element(data.begin(), data.end());
    fmax += (fmax-fmin)*1e-9;

    double bin = (fmax-fmin)/nbins;

    std::map<double,int> histo;
    for (int nbin=0; fmin + nbin*bin < fmax; ++nbin) {
        histo[fmin + nbin*bin] = 0;
    }
    for (std::size_t npoint=0; npoint < data.size(); ++npoint) {
        int nbin = int((data[npoint]-fmin) / bin);
        histo[fmin + nbin*bin]++;
    }
    return histo;
}

Vector_double
stfnum::deconvolve(const Vector_double& dataIn, const Vector_double& templ,
                int SR, double hipass, double lopass, stfio::ProgressInfo& progDlg)
{
	// Normalize data
    double fmax = *std::max_element(dataIn.begin(), dataIn.end());
    double fmin = *std::min_element(dataIn.begin(), dataIn.end());
    Vector_double data = stfio::vec_scal_minus(dataIn, fmin);
    data = stfio::vec_scal_div(data, fmax-fmin);

    bool skipped = false;
    progDlg.Update( 0, "Starting deconvolution...", &skipped );
    if (data.size()<=0 || templ.size() <=0 || templ.size() > data.size()) {
        std::out_of_range e("subscript out of range in stfnum::filter()");
        throw e;
    }
    /* pad templ */
    double* in_templ_padded =(double *)fftw_malloc(sizeof(double) * data.size());
    std::copy(templ.begin(), templ.end(), in_templ_padded);
    if (templ.size() < data.size()) {
        for (size_t kp=templ.size(); kp<data.size(); ++kp)
            in_templ_padded[kp] = 0;
    }

    Vector_double data_return(data.size());
    if (skipped) {
        data_return.resize(0);
        return data_return;
    }

    //fftw_complex is a double[2]; hence, out is an array of
    //double[2] with out[n][0] being the real and out[n][1] being
    //the imaginary part.
    fftw_plan p_data, p_templ, p_inv;

    //memory allocation as suggested by fftw:
    double* in_data =(double *)fftw_malloc(sizeof(double) * data.size());
    std::copy(data.begin(), data.end(), in_data);
    fftw_complex* out_data = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * ((int)(data.size()/2)+1));

    //plan the ffts and execute them:
    p_data =fftw_plan_dft_r2c_1d((int)data.size(), in_data, out_data,
                                 FFTW_ESTIMATE);
    fftw_execute(p_data);
    if (isnan(out_data[0][0]) || isinf(out_data[0][0])) {
        data_return.resize(0);
        throw std::runtime_error("Unstable fft; try again avoiding any test pulses (if present)");
    }
    fftw_complex* out_templ_padded = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * ((int)(data.size()/2)+1));
    p_templ =fftw_plan_dft_r2c_1d((int)data.size(),
                                  in_templ_padded, out_templ_padded, FFTW_ESTIMATE);
    fftw_execute(p_templ);

    double SI=1.0/SR; //the sampling interval
    progDlg.Update( 25, "Performing deconvolution...", &skipped );
    if (skipped) {
        data_return.resize(0);
        return data_return;
    }

    Vector_double f_c(1);
    for (std::size_t n_point=0; n_point < (unsigned int)(data.size()/2)+1; ++n_point) {
        /* highpass filter */
        double f = n_point / (data.size()*SI);

        double rslt_hi = 1.0;
        if (hipass > 0) {
            f_c[0] = hipass;
            rslt_hi = 1.0-fgaussColqu(f, f_c);
        }

        /* lowpass filter */
        double rslt_lo = 1.0;
        if (lopass > 0) {
            f_c[0] = lopass;
            rslt_lo= fgaussColqu(f, f_c);
        }

        /* do the division in place */
        double a = out_data[n_point][0];
        double b = out_data[n_point][1];
        double c = out_templ_padded[n_point][0];
        double d = out_templ_padded[n_point][1];
        double mag2 = c*c + d*d;
        out_data[n_point][0] = rslt_hi * rslt_lo * (a*c + b*d)/mag2;
        out_data[n_point][1] = rslt_hi * rslt_lo * (b*c - a*d)/mag2;
    }

    //do the reverse fft:
    p_inv = fftw_plan_dft_c2r_1d((int)data.size(),out_data, in_data, FFTW_ESTIMATE);
    fftw_execute(p_inv);

    //fill the return array, adding the offset, and scaling by data.size()
    //(because fftw computes an unnormalized transform):
    for (std::size_t n_point=0; n_point < data.size(); ++n_point) {
        data_return[n_point]= in_data[n_point]/data.size();
    }

    fftw_destroy_plan(p_data);
    fftw_destroy_plan(p_templ);
    fftw_destroy_plan(p_inv);

    fftw_free(in_data);
    fftw_free(out_data);
    fftw_free(in_templ_padded);
    fftw_free(out_templ_padded);

    progDlg.Update( 50, "Computing data histogram...", &skipped );
    if (skipped) {
        data_return.resize(0);
        return data_return;
    }
    int nbins =  500; //int(data_return.size()/500.0);
    std::map<double, int> histo = histogram(data_return, nbins);
    double max_value = -1;
    double max_time = 0;
    double maxhalf_time = 0;
    Vector_double histo_fit(0);
    for (std::map<double,int>::const_iterator it=histo.begin();
         it != histo.end(); ++it) {
        if (it->second > max_value) {
            max_value = it->second;
            max_time = it->first;
        }
        histo_fit.push_back(it->second);
#ifdef _STFDEBUG
        std::cout << it->first << "\t" << it->second << std::endl;
#endif
    }
    for (std::map<double,int>::const_iterator it=histo.begin();
         it != histo.end(); ++it) {
        if (it->second > 0.5*max_value) {
            maxhalf_time = it->first;
            break;
        }
    }
    maxhalf_time = fabs(max_time-maxhalf_time);
    progDlg.Update( 75, "Fitting Gaussian...", &skipped );
    if (skipped) {
        data_return.resize(0);
        return data_return;
    }
    
    /* Fit Gaussian to histogram */
    double interval = (++histo.begin())->first-histo.begin()->first;
    if (maxhalf_time==0) {
        maxhalf_time = interval;
    }
    /* Initial parameter guesses */
    Vector_double pars(3);
    pars[0] = max_value;
    pars[1] = (max_time - histo.begin()->first);
    pars[2] = maxhalf_time *sqrt(2.0)/2.35482;
#ifdef _STFDEBUG    
    std::cout << "nbins: " << nbins << std::endl;
    std::cout << "initial values:" << std::endl;
    for (std::size_t np=0; np<pars.size(); ++np) {
        std::cout << pars[np] << std::endl;
    }
#endif

    Vector_double opts = LM_default_opts();
    std::vector< stfnum::storedFunc > funcLib = stfnum::GetFuncLib();
    std::string info;
    int warning;
#ifdef _STFDEBUG
    double chisqr =
#endif
        lmFit(histo_fit, interval, funcLib[funcLib.size()-2], opts, true,
              pars, info, warning );
#ifdef _STFDEBUG
    std::cout << chisqr << "\t" << interval << std::endl;
    std::cout << "final values:" << std::endl;
    for (std::size_t np=0; np<pars.size(); ++np) {
        std::cout << pars[np] << std::endl;
    }
#endif
    double sigma = pars[2]/sqrt(2.0);
    /* return data in terms of sigma */
    for (std::size_t n_point=0; n_point < data.size(); ++n_point) {
        data_return[n_point] /= sigma;
    }
    progDlg.Update( 100, "Done.", &skipped );
    return data_return;
}

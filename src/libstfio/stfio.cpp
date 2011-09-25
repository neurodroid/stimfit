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

/*! \file stfio.c
 *  \author Christoph Schmidt-Hieber
 *  \date 2011-09-25
 *  \brief General functions for libstfio
 * 
 * 
 *  Implements some general functions for libstfio
 */

#include "stfio.h"

#include <sstream>

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
stfio::linsolv( int m, int n, int nrhs, Vector_double& A,
        Vector_double& B)
{
#ifndef TEST_MINIMAL
    if (A.size()<=0) {
        throw std::runtime_error("Matrix A has size 0 in stfio::linsolv");
    }

    if (B.size()<=0) {
        throw std::runtime_error("Matrix B has size 0 in stfio::linsolv");
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
        throw std::runtime_error(std::string(error_msg.str()));
    }
#endif
    return 0;
}

stfio::Table::Table(std::size_t nRows,std::size_t nCols) :
values(nRows,std::vector<double>(nCols,1.0)),
    empty(nRows,std::deque<bool>(nCols,false)),
    rowLabels(nRows, "\0"),
    colLabels(nCols, "\0")
    {}

stfio::Table::Table(const std::map< std::string, double >& map)
: values(map.size(),std::vector<double>(1,1.0)), empty(map.size(),std::deque<bool>(1,false)),
rowLabels(map.size(), "\0"), colLabels(1, "Results")
{
    std::map< std::string, double >::const_iterator cit;
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

double stfio::Table::at(std::size_t row,std::size_t col) const {
    try {
        return values.at(row).at(col);
    }
    catch (...) {
        throw;
    }
}

double& stfio::Table::at(std::size_t row,std::size_t col) {
    try {
        return values.at(row).at(col);
    }
    catch (...) {
        throw;
    }
}

bool stfio::Table::IsEmpty(std::size_t row,std::size_t col) const {
    try {
        return empty.at(row).at(col);
    }
    catch (...) {
        throw;
    }
}

void stfio::Table::SetEmpty(std::size_t row,std::size_t col,bool value) {
    try {
        empty.at(row).at(col)=value;
    }
    catch (...) {
        throw;
    }
}

void stfio::Table::SetRowLabel(std::size_t row,const std::string& label) {
    try {
        rowLabels.at(row)=label;
    }
    catch (...) {
        throw;
    }
}

void stfio::Table::SetColLabel(std::size_t col,const std::string& label) {
    try {
        colLabels.at(col)=label;
    }
    catch (...) {
        throw;
    }
}

const std::string& stfio::Table::GetRowLabel(std::size_t row) const {
    try {
        return rowLabels.at(row);
    }
    catch (...) {
        throw;
    }
}

const std::string& stfio::Table::GetColLabel(std::size_t col) const {
    try {
        return colLabels.at(col);
    }
    catch (...) {
        throw;
    }
}

void stfio::Table::AppendRows(std::size_t nRows_) {
    std::size_t oldRows=nRows();
    rowLabels.resize(oldRows+nRows_);
    values.resize(oldRows+nRows_);
    empty.resize(oldRows+nRows_);
    for (std::size_t nRow = 0; nRow < oldRows + nRows_; ++nRow) {
        values[nRow].resize(nCols());
        empty[nRow].resize(nCols());
    }
}

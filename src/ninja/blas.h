/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  BLAS and MKL interfaces for compile time dispatching to various
 *           math calculations (OpenBLAS, CUDA, etc.)
 * Author:   Kyle Shannon <kyle@pobox.com>
 *
 ******************************************************************************
 * * THIS SOFTWARE WAS DEVELOPED AT THE ROCKY MOUNTAIN RESEARCH STATION (RMRS)
 * MISSOULA FIRE SCIENCES LABORATORY BY EMPLOYEES OF THE FEDERAL GOVERNMENT
 * IN THE COURSE OF THEIR OFFICIAL DUTIES. PURSUANT TO TITLE 17 SECTION 105
 * OF THE UNITED STATES CODE, THIS SOFTWARE IS NOT SUBJECT TO COPYRIGHT
 * PROTECTION AND IS IN THE PUBLIC DOMAIN. RMRS MISSOULA FIRE SCIENCES
 * LABORATORY ASSUMES NO RESPONSIBILITY WHATSOEVER FOR ITS USE BY OTHER
 * PARTIES,  AND MAKES NO GUARANTEES, EXPRESSED OR IMPLIED, ABOUT ITS QUALITY,
 * RELIABILITY, OR ANY OTHER CHARACTERISTIC.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#ifndef NINJA_BLAS_H
#define NINJA_BLAS_H

#include <assert.h>
#include <math.h>

#include <omp.h>

#include <cpl_multiproc.h>

#if defined(NINJA_OPEN_BLAS)
#include <cblas.h>
#elif defined(NINJA_CUDA_BLAS)
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse_v2.h"
#endif

#ifndef OFFSET
#define OFFSET(N, incX) ((incX) > 0 ?  0 : ((N) - 1) * (-(incX))) //for cblas_dscal
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void * ninja_matrix_malloc(size_t n);

void * ninja_matrix_finalize(void *p, size_t n);

void ninja_blas_dscal(const int N, const double alpha, double *X, const int incX);

void ninja_blas_dcopy(const int N, const double *X, const int incX, double *Y, const int incY);

double ninja_blas_ddot(const int N, const double *X, const int incX, const double *Y, const int incY);

void ninja_blas_daxpy(const int N, const double alpha, const double *X,
        const int incX, double *Y, const int incY);

double ninja_blas_dnrm2(const int N, const double *X, const int incX);

void ninja_dcsrmv(char *transa, int *m, int *k, double *alpha, char *matdescra, double *val, int *indx, int *pntrb, int *pntre, double *x, double *beta, double *y);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* NINJA_BLAS_H */

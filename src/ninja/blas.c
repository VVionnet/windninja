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


#include "blas.h"

#if defined(NINJA_CUDA_BLAS)
#define CUDA_MALLOC(p, n) cudaMallocHost(p, n)
#define CUDA_FREE(p) cudaFree(p)
#define CUDA_SIZE sizeof(double)
static cublasHandle_t blasHandle = 0;
static cusparseHandle_t sparseHandle = 0;
#endif

void * ninja_matrix_malloc(size_t n) {
#if defined(NINJA_CUDA_BLAS)
    void *p;
    CUDA_MALLOC(p, n);
#else
    return malloc(n);
#endif
}

void * ninja_matrix_finalize(void *p, size_t n) {
#if defined(NINJA_CUDA_BLAS)
    void *dst;
    cublasGetVector(n, CUDA_SIZE, p, 1, dst, 1);
    return dst;
#else
    return p;
#endif
}

void ninja_blas_dscal(const int N, const double alpha, double *X, const int incX) {
    assert(incX == 1);
#if defined(NINJA_OPEN_BLAS)
    cblas_dscal(N, alpha, X, 1);
#elif defined(NINJA_CUDA_BLAS)
    cublasStatus_t stat;
    if(!blasHandle) {
        stat = cublasCreate(&blasHandle);
        assert(stat == CUBLAS_STATUS_SUCCESS);
    }
    double *dx;
    CUDA_MALLOC((void**)&dx, CUDA_SIZE * N);
    cublasSetVector(N, CUDA_SIZE, X, 1, dx, 1);

    stat = cublasDscal(blasHandle, N, &alpha, dx, 1);
    assert(stat == CUBLAS_STATUS_SUCCESS);
    cublasGetVector(N, CUDA_SIZE, dx, 1, X, 1);
    CUDA_FREE(dx);
#else
     int i, ix;
     ix = OFFSET(N, incX);
     for (i = 0; i < N; i++) {
         X[ix] *= alpha;
         ix    += incX;
     }
#endif
}

void ninja_blas_dcopy(const int N, const double *X, const int incX, double *Y, const int incY) {
    assert(incX == 1 && incY == 1);
#if defined(NINJA_CUDA_BLAS) && defined(_NOT_DEFINED)
    cudaMemcpy(Y, X, CUDA_SIZE * N, cudaMemcpyDeviceToDevice);
#else
    memcpy(Y, X, sizeof(double) * N );
#endif
}

double ninja_blas_ddot(const int N, const double *X, const int incX, const double *Y, const int incY) {
    assert(incX == 1 && incY == 1);
#if defined(NINJA_OPEN_BLAS)
    cblas_ddot(...);
#elif defined(NINJA_CUDA_BLAS)
    cublasStatus_t stat;
    if(!blasHandle) {
        stat = cublasCreate(&blasHandle);
        assert(stat == CUBLAS_STATUS_SUCCESS);
    }
    double *dx, *dy;
    CUDA_MALLOC((void**)&dx, CUDA_SIZE * N);
    CUDA_MALLOC((void**)&dy, CUDA_SIZE * N);
    cublasSetVector(N, CUDA_SIZE, X, 1, dx, 1);
    cublasSetVector(N, CUDA_SIZE, Y, 1, dy, 1);

    double dp;
    stat = cublasDdot(blasHandle, N, dx, 1, dy, 1, &dp);
    assert(stat == CUBLAS_STATUS_SUCCESS);
    CUDA_FREE(dx);
    CUDA_FREE(dy);
    return dp;
#else
    double val=0.0;
    int i;
    #pragma omp parallel for reduction(+:val)
    for(i=0;i<N;i++) {
        val += X[i]*Y[i];
    }
    return val;
#endif
}
/*
** Performs the calculation Y = Y + alpha * X where N is the size of the
** matrices X an Y.  incX and incY must be 1, the function signature keeps the
** increment variables to remain consistent with CBLAS.
*/
void ninja_blas_daxpy(const int N, const double alpha, const double *X,
        const int incX, double *Y, const int incY) {
    assert(incX == 1 && incY == 1);
#if defined(NINJA_OPEN_BLAS)
    cblas_daxpy(N, alpha, X, incX, Y, incY);
#elif defined(NINJA_CUDA_BLAS)
    cublasStatus_t stat;
    if(!blasHandle) {
        stat = cublasCreate(&blasHandle);
        assert(stat == CUBLAS_STATUS_SUCCESS);
    }
    double *dx, *dy;
    CUDA_MALLOC((void**)&dx, CUDA_SIZE * N);
    CUDA_MALLOC((void**)&dy, CUDA_SIZE * N);
    cublasSetVector(N, CUDA_SIZE, X, 1, dx, 1);
    cublasSetVector(N, CUDA_SIZE, Y, 1, dy, 1);

    stat = cublasDaxpy(blasHandle, N, &alpha, dx, 1, dy, 1);
    assert(stat == CUBLAS_STATUS_SUCCESS);
    cublasGetVector(N, CUDA_SIZE, dy, 1, Y, 1);
    CUDA_FREE(dx);
    CUDA_FREE(dy);
    //cublasDestroy(blasHandle);
#else
    int i;
    #pragma omp parallel for
    for(int i = 0; i < N; i++) {
        Y[i] += alpha*X[i];
    }
#endif
}

double ninja_blas_dnrm2(const int N, const double *X, const int incX) {
    assert(incX == 1);
#if defined(NINJA_OPEN_BLAS)
    cblas_dnrm2(...);
#elif defined(NINJA_CUDA_BLAS)
    cublasStatus_t stat;
    if(!blasHandle) {
        stat = cublasCreate(&blasHandle);
        assert(stat == CUBLAS_STATUS_SUCCESS);
    }
    double *dx;
    CUDA_MALLOC((void**)&dx, CUDA_SIZE * N);
    cublasSetVector(N, CUDA_SIZE, X, 1, dx, 1);

    double norm;
    stat = cublasDnrm2(blasHandle, N, dx, 1, &norm);
    assert(stat == CUBLAS_STATUS_SUCCESS);
    CUDA_FREE(dx);
    return norm;
#else
    double val=0.0;
    int i;
    //#pragma omp parallel for reduction(+:val)
    for(i=0;i<N;i++) {
        val += X[i]*X[i];
    }
    return sqrt(val);
#endif
}

void ninja_dcsrmv(char *transa, int *m, int *k, double *alpha, char *matdescra, double *val, int *indx, int *pntrb, int *pntre, double *x, double *beta, double *y) {
#if defined(NINJA_OPEN_BLAS)
#elif defined(NINJA_CUDA_BLAS) && defined(_NOT_DEFINED) // Not implemented
    cusparseStatus_t stat;
    if(!sparseHandle) {
        stat = cusparseCreate(&sparseHandle);
        assert(stat == CUSPARSE_STATUS_SUCCESS);
    }
    /*
    cusparseDcsrmv(cusparseHandle_t handle, cusparseOperation_t transA, 
        int m, int n, int nnz, const double          *alpha, 
        const cusparseMatDescr_t descrA, 
        const double          *csrValA, 
        const int *csrRowPtrA, const int *csrColIndA,
        const double          *x, const double          *beta, 
        double          *y)
    */
#else
    int i,j,N;
    N=*m;
    #pragma omp parallel private(i,j)
    {
        #pragma omp for
        for(i=0;i<N;i++) {
            y[i]=0.0;
        }
        #pragma omp for
        for(i=0;i<N;i++) {
            y[i] += val[pntrb[i]]*x[i];	// diagonal
            for(j=pntrb[i]+1;j<pntre[i];j++)
            {
                y[i] += val[j]*x[indx[j]];
            }
        }
    }

    for(i=0;i<N;i++) {
        for(j=pntrb[i]+1;j<pntre[i];j++) {
            y[indx[j]] += val[j]*x[i];
        }
    }
#endif
}

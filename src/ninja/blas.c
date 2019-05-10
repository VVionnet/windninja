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
    CUDA_MALLOC(&p, n);
    return p;
#else
    return malloc(n);
#endif
}

void * ninja_matrix_finalize(void *p, size_t n) {
#if defined(NINJA_CUDA_BLAS)
    void *dst;
    dst = malloc(CUDA_SIZE * n);
    cudaMemcpy(dst, p, CUDA_SIZE * n, cudaMemcpyDeviceToHost);
    CUDA_FREE(p);
    return dst;
#else
    return p;
#endif
}

void ninja_matrix_free(void *p) {
    free(p);
}

void ninja_matrix_memset(void *p, int c, size_t n) {
#if defined(NINJA_CUDA_BLAS)
    cudaMemset(p, c, n);
#else
    memset(p, c, n);
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
    stat = cublasDscal(blasHandle, N, &alpha, X, 1);
    assert(stat == CUBLAS_STATUS_SUCCESS);
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
    double dp;
    stat = cublasDdot(blasHandle, N, X, 1, Y, 1, &dp);
    assert(stat == CUBLAS_STATUS_SUCCESS);
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
    stat = cublasDaxpy(blasHandle, N, &alpha, X, 1, Y, 1);
    assert(stat == CUBLAS_STATUS_SUCCESS);
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
    double norm;
    stat = cublasDnrm2(blasHandle, N, X, 1, &norm);
    assert(stat == CUBLAS_STATUS_SUCCESS);
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

void ninja_blas_sub(const int N, double *R, const int incR, const double *b, const int incB) {
    int i = 0;
    for(i = 0; i < N; i++) {
      R[i] = b[i] - R[i];
    }
}

void ninja_dcsrmv(char *transa, int *m, int *k, double *alpha, char *matdescra,
        double *val, int *indx, int *pntrb, int *pntre, double *x, double *beta, double *y) {
#if defined(NINJA_OPEN_BLAS)
#elif defined(NINJA_CUDA_BLAS) && defined(_NOT_)
    cusparseStatus_t stat;
    if(!sparseHandle) {
        stat = cusparseCreate(&sparseHandle);
        assert(stat == CUSPARSE_STATUS_SUCCESS);
    }
    int N = *m;
    int nnz = pntre[N-1] - pntrb[0];
    // malloc on device: csrValA, csrRowPtrA, csrColIndA, x, beta, y
    double *dcsrValA, *dcsrRowPtrA, *dcsrColIndA, *dx, *dy;
    stat = CUDA_MALLOC((void**)&dcsrValA, CUDA_SIZE * nnz);
    assert(stat == CUSPARSE_STATUS_SUCCESS);
    stat = CUDA_MALLOC((void**)&dcsrRowPtrA, CUDA_SIZE * (N + 1));
    assert(stat == CUSPARSE_STATUS_SUCCESS);
    stat = CUDA_MALLOC((void**)&dcsrColIndA, CUDA_SIZE * nnz);
    assert(stat == CUSPARSE_STATUS_SUCCESS);
    stat = CUDA_MALLOC((void**)&dx, CUDA_SIZE * N);
    assert(stat == CUSPARSE_STATUS_SUCCESS);
    stat = CUDA_MALLOC((void**)&dy, CUDA_SIZE * N);
    assert(stat == CUSPARSE_STATUS_SUCCESS);


    stat = cudaMemcpy(dcsrValA, val, CUDA_SIZE * nnz, cudaMemcpyHostToDevice);
    assert(stat == CUSPARSE_STATUS_SUCCESS);
    stat = cudaMemcpy(dcsrRowPtrA, pntrb, CUDA_SIZE * (N + 1), cudaMemcpyHostToDevice);
    assert(stat == CUSPARSE_STATUS_SUCCESS);
    stat = cudaMemcpy(dcsrColIndA, indx, CUDA_SIZE * nnz, cudaMemcpyHostToDevice);
    assert(stat == CUSPARSE_STATUS_SUCCESS);
    stat = cudaMemcpy(dx, x, CUDA_SIZE * N, cudaMemcpyHostToDevice);
    assert(stat == CUSPARSE_STATUS_SUCCESS);
    stat = cudaMemcpy(dy, y, CUDA_SIZE * N, cudaMemcpyHostToDevice);
    assert(stat == CUSPARSE_STATUS_SUCCESS);

    cusparseMatDescr_t descrA;
    stat = cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
    assert(stat == CUSPARSE_STATUS_SUCCESS);
    stat = cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    assert(stat == CUSPARSE_STATUS_SUCCESS);
    stat = cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_UPPER);
    assert(stat == CUSPARSE_STATUS_SUCCESS);
    stat = cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_UNIT);
    assert(stat == CUSPARSE_STATUS_SUCCESS);

    stat = cusparseDcsrmv(sparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            *m,
            *k,
            nnz,
            alpha,
            descrA,
            val,
            pntrb,
            pntre,
            x,
            beta,
            y);
    printf("%d\n", stat);
    assert(stat == CUSPARSE_STATUS_SUCCESS);

    CUDA_FREE(dcsrValA);
    CUDA_FREE(dcsrRowPtrA);
    CUDA_FREE(dcsrColIndA);
    CUDA_FREE(dx);
    CUDA_FREE(dy);
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
            y[i] += val[pntrb[i]]*x[i];    // diagonal
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

void ninja_dcsrsv(char *transa, int *m, double *alpha, char *matdescra, double *val, int *indx, int *pntrb, int *pntre, double *x, double *y) {
    // My version of the mkl_dcsrsv() function; solves val*y=x
    // Only works for my specific settings
    //        Case 1:
    //            transa='t';            //solve using transpose y := alpha*inv(A')*x
    //            matdescra[0]='t';    //triangular
    //            matdescra[1]='u';    //upper triangle
    //            matdescra[2]='u';    //unit diagonal
    //            matdescra[3]='c';    //zero based indexing
    //        Case 2:
    //            transa='n';            //solve using regular matrix (not transpose) y := alpha*inv(A)*x
    //            matdescra[0]='t';    //triangular
    //            matdescra[1]='u';    //upper triangle
    //            matdescra[2]='n';    //not unit diagonal
    //            matdescra[3]='c';    //zero based indexing
    int i, j;

    //Case 1:
    if(*transa=='t' && matdescra[0]=='t' && matdescra[1]=='u' && matdescra[2]=='u' && matdescra[3]=='c')
    {
        for(i=0; i<*m; i++)
            y[i] = x[i];
        for(i=0; i<*m; i++)
        {
                            // normally would have x[i]/diagonal of val[i,i] here, but val[i,i] is unit (=1)
            for(j=pntrb[i]; j<pntre[i]; j++)
            {
                y[indx[j]] -=  y[i]*val[j];
            }
        }
    //Case 2:
    }else if(*transa=='n' && matdescra[0]=='t' && matdescra[1]=='u' && matdescra[2]=='n' && matdescra[3]=='c')
    {
        for(i=*m-1; i>=0; i--)    //loop up rows
            y[i] = x[i];
        y[*m-1] /= val[pntrb[*m-1]];
        for(i=*m-2; i>=0; i--)    //loop up rows
        {
            for(j=pntrb[i]+1; j<pntre[i]; j++)    //don't include diagonal in loop
                y[i] -= val[j]*y[indx[j]];
            y[i] /= val[pntrb[i]];
        }
    }
}



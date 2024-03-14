#define __STDC_FORMAT_MACROS
#include <cstdlib>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <inttypes.h>


void matrix_vector_product_omp(double* a, double *b, double *c, int m, int n)
{
    #pragma omp parallel num_threads(40)
        {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++) {
            c[i] = 0.0;
            for (int j = 0; j < n; j++){
                c[i] += a[i * n + j] * b[j];
            }
        }
    }
}

void parallel_init(double* a, double *b, double *c, int m, int n){
    #pragma omp parallel num_threads(40)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i < ub; i++) {
            for (int j = 0; j < n; j++){
                a[i * n + j] = i + j;
            }
        }
        for (int j = 0; j < n; j++){
            b[j] = j;
        }
    }
    
}

void run_parallel(int m, int n)
{
    double* a;
    double* b;
    double* c;
    a = (double*)malloc(sizeof(double) * m * n);
    b = (double*)malloc(sizeof(double) * n);
    c = (double*)malloc(sizeof(double) * m);
    
    double t1 = omp_get_wtime ();

    parallel_init(a, b, c, m, n);

    t1 = omp_get_wtime () - t1;
    printf("Elapsed time (parallel): %.2f sec.\n", t1);
    double t;
    t = omp_get_wtime ();
    matrix_vector_product_omp(a, b, c, m, n);
    t = omp_get_wtime () - t;
    printf("Elapsed time (parallel): %.2f sec.\n", t);
    free(a);
    free(b);
    free(c);
}

int main(int argc, char **argv)
{   
    int m = 20000, n = 20000;
    printf("Matrix-vector product (c[m] = a[m, n] * b[n]; m = %d, n = %d)\n", m, n);
    printf("Memory used: %" PRIu64 " MiB\n", ((m * n + m + n) * sizeof(double)) >> 20);
    // run_serial();
    run_parallel(m, n);

return 0;
}
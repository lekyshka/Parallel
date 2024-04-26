#include <iostream>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include </opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include/nvtx3/nvToolsExt.h>
// #include <nvToolsExt.h>

#define OFFSET(x, y, m) (((x)*(m)) + (y))

void initialize(double *A, double *Anew, int n)
{
    memset(A, 0, n * n * sizeof(double));
    memset(Anew, 0, n *n * sizeof(double));
    // for(int i = 0; i < n * n; i++){
    //     A[i] = 0.0;
    //     Anew[i] = 0.0;
    // }
    double corners[4] = {10, 20, 30, 20};
    A[0] = corners[0];
    A[n - 1] = corners[1];
    A[n * n - 1] = corners[2];
    A[n * (n - 1)] = corners[3];
    double step = (corners[1] - corners[0]) / (n - 1);


    for (int i = 1; i < n - 1; i ++) {
        A[i] = corners[0] + i * step;
        A[n * i] = corners[0] + i * step;
        A[(n-1) + n * i] = corners[1] + i * step;
        A[n * (n-1) + i] = corners[3] + i * step;
        Anew[i] = corners[0] + i * step;
        Anew[n * i] = corners[0] + i * step;
        Anew[(n-1) + n * i] = corners[1] + i * step;
        Anew[n * (n-1) + i] = corners[3] + i * step;
    }
    #pragma acc enter data copyin(A[:n*n],Anew[:n*n])
}

double calcNext(double *A, double *Anew, int n)
{
    double error = 0.0;

    #pragma acc parallel loop reduction(max:error) present(A,Anew)
    for( int j = 1; j < n-1; j++) {
        #pragma acc loop
        for( int i = 1; i < n-1; i++ ) {
            // Anew[OFFSET(j, i, n)] = ( A[OFFSET(j, i+1, n)] + A[OFFSET(j, i-1, n)]
            //                             + A[OFFSET(j-1, i, n)] + A[OFFSET(j+1, i,n)])*0.25;
            // error = fmax( error, fabs(Anew[OFFSET(j, i, n)] - A[OFFSET(j, i , n)]));

            Anew[i * n + j] = ( A[i * n + j - 1] + A[(i - 1) * n + j]
                                        + A[(i + 1) * n + j] + A[i * n + j + 1])*0.25;
            error = fmax(error, Anew[i * n + j] - A[i * n + j]);


        }
    }


    return error;
}
        
void swap(double *A, double *Anew, int n)
{
    #pragma acc parallel loop present(A,Anew)
    for( int j = 1; j < n-1; j++)
    {
        #pragma acc loop
        for( int i = 1; i < n-1; i++ )
        {
            A[OFFSET(j, i, n)] = Anew[OFFSET(j, i, n)];    
        }
    }
}

void deallocate(double *A, double *Anew)
{
    #pragma acc exit data delete(A,Anew)
    free(A);
    free(Anew);
}


int main(int argc, char* argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "Produce help message")
        ("precision", po::value<double>()->default_value(0.000001), "Set precision")
        ("grid-size", po::value<int>()->default_value(10), "Set grid size")
        ("iterations", po::value<int>()->default_value(1000000), "Set number of iterations");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
    }

    double precision = vm["precision"].as<double>();
    int n = vm["grid-size"].as<int>();
    int iter_max= vm["iterations"].as<int>();

    double error = 1.0;

    double *A = (double *)malloc(sizeof(double) * n * n);
    double *Anew = (double *)malloc(sizeof(double) * n * n);

    initialize(A, Anew, n);

    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, n);

    double st = omp_get_wtime();
    int iter = 0;

    while (error > precision && iter < iter_max)
    {
        error = calcNext(A, Anew, n);

        swap(A, Anew, n);

        if (iter % 100 == 0)
            printf("%5d, %0.6f\n", iter, error);

        iter++;

    }

    double runtime = omp_get_wtime() - st;
    printf("%5d, %0.6f\n", iter, error);

    printf(" total: %f s\n", runtime);
    #pragma acc update host(A[:n*n]) 
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%0.6f ", A[i*n+j]); 
        }
        printf("\n");
    }

    deallocate(A, Anew);
 
    return 0;
}

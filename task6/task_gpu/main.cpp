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
    }
    std::memcpy(Anew, A, n * n * sizeof(double));
    #pragma acc enter data copyin(A[:n*n],Anew[:n*n])
}

double calcNext(double *A, double *Anew, int n)
{
    double error = 0.0;

    #pragma acc parallel loop reduction(max:error) present(A,Anew)
    for( int j = 1; j < n-1; j++) {
        #pragma acc loop
        for( int i = 1; i < n-1; i++ ) {
            Anew[OFFSET(j, i, n)] = ( A[OFFSET(j, i+1, n)] + A[OFFSET(j, i-1, n)]
                                        + A[OFFSET(j-1, i, n)] + A[OFFSET(j+1, i,n)])*0.25;
            error = fmax( error, fabs(Anew[OFFSET(j, i, n)] - A[OFFSET(j, i , n)]));
        }
    }

    return error;
}


void deallocate(double *A, double *Anew)
{
    #pragma acc exit data delete(A,Anew)
    delete[] A;
    delete[] Anew;
}


int main(int argc, char* argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "Produce help message")
        ("precision", po::value<double>()->default_value(0.000001), "Set precision")
        ("grid-size", po::value<int>()->default_value(1024), "Set grid size")
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

    double* A = new double[n * n];
    double* Anew = new double[n * n];

    initialize(A, Anew, n);

    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, n);

    double st = omp_get_wtime();
    int iter = 0;
    #pragma acc data copyin(error)
    while (error > precision && iter < iter_max)
    {
       if(iter % 100 == 0){
            error = 0.0;
            #pragma acc update device(error) async(1)
        }
        #pragma acc parallel loop reduction(max:error) present(A,Anew)
        for( int j = 1; j < n-1; j++) {
            #pragma acc loop
            for( int i = 1; i < n-1; i++ ) {
                Anew[OFFSET(j, i, n)] = ( A[OFFSET(j, i+1, n)] + A[OFFSET(j, i-1, n)]
                                            + A[OFFSET(j-1, i, n)] + A[OFFSET(j+1, i,n)])*0.25;
                error = fmax( error, fabs(Anew[OFFSET(j, i, n)] - A[OFFSET(j, i , n)]));
            }
        }
       
        double* temp = A;
        A = Anew;
        Anew = temp;


        if (iter % 100 == 0)
        	#pragma acc update host(error) async(1)	
	        #pragma acc wait(1)
            printf("%5d, %0.6f\n", iter, error);

        iter++;

    }

    double runtime = omp_get_wtime() - st;
    printf("%5d, %0.6f\n", iter, error);

    printf(" total: %f s\n", runtime);


    deallocate(A, Anew);
 
    return 0;
}

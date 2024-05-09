#include <iostream>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>

#define OFFSET(x, y, m) (((x)*(m)) + (y))

void initialize(std::unique_ptr<double[]> &A, std::unique_ptr<double[]> &Anew, int n)
{
    memset(A.get(), 0, n * n * sizeof(double));

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
    std::memcpy(Anew.get(), A.get(), n * n * sizeof(double));

    
}

void deallocate(double *A, double *Anew)
{

    A = nullptr;
    Anew = nullptr;

}


int main(int argc, char* argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "Produce help message")
        ("precision", po::value<double>()->default_value(0.000001), "Set precision")
        ("grid-size", po::value<int>()->default_value(512), "Set grid size")
        ("iterations", po::value<int>()->default_value(1000000), "Set number of iterations");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
    }
    po::notify(vm);



    double precision = vm["precision"].as<double>();
    int n = vm["grid-size"].as<int>();
    int iter_max= vm["iterations"].as<int>();

    double error = 1.0;

    std::unique_ptr<double[]> A_ptr(new double[n*n]);
    std::unique_ptr<double[]> Anew_ptr(new double[n*n]);
    initialize(std::ref(A_ptr), std::ref(Anew_ptr), n);

    double* A = A_ptr.get();
    double* Anew = Anew_ptr.get();

    

    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, n);

    auto start = std::chrono::high_resolution_clock::now();
    int iter = 0;
    {
         // nvtxRangePushA("Main loop");
        while (error > precision && iter < iter_max)
        {
        if(iter % 100 == 0){
                error = 0.0;
                #pragma acc parallel loop independent collapse(2) vector vector_length(256) gang num_gangs(n) reduction(max:error)
                for( int j = 1; j < n-1; j++) {
                    for( int i = 1; i < n-1; i++ ) {
                        Anew[OFFSET(j, i, n)] = ( A[OFFSET(j, i+1, n)] + A[OFFSET(j, i-1, n)]
                                                    + A[OFFSET(j-1, i, n)] + A[OFFSET(j+1, i,n)])*0.25;

                        error = fmax( error, fabs(Anew[OFFSET(j, i, n)] - A[OFFSET(j, i , n)]));
                            
                    }
                }
                printf("%5d, %0.6f\n", iter, error);
            }
            else{
                #pragma acc parallel loop independent collapse(2) vector vector_length(256) gang num_gangs(1024)
                for( int j = 1; j < n-1; j++) {
                    for( int i = 1; i < n-1; i++ ) {
                        Anew[OFFSET(j, i, n)] = ( A[OFFSET(j, i+1, n)] + A[OFFSET(j, i-1, n)]
                                                    + A[OFFSET(j-1, i, n)] + A[OFFSET(j+1, i,n)])*0.25;
                            
                    }
                }
            }

        
            double* temp = A;
            A = Anew;
            Anew = temp;

            iter++;

        }
        // nvtxRangePop();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> runtime = end - start;
        printf("%5d, %0.6f\n", iter, error);

        printf(" total: %f s\n", runtime);
    }
   
    return 0;
}

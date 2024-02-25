#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <malloc.h>
#include <vector>
#include <limits.h>

double loss_prev = INT_MAX;

std::vector<double> simpleIterationMethod(const std::vector<std::vector<double>> &A, const std::vector<double> &b,
                                          double eps, int nm, int n, double b_znam)
{
    std::vector<double> x(n, 0.0);
    double tet = 0.0001;
    double err = 1;
    while (err > eps)
    {
        std::vector<double> x_new(n, 0.0);
        std::vector<double> err_chisl(n, 0.0);
        double chisl = 0;
#pragma omp parallel num_threads(nm)
        {
            int nthreads = omp_get_num_threads();
            int threadid = omp_get_thread_num();

            int items_per_thread = n / nthreads;
            int lb = threadid * items_per_thread;
            int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);
            for (int i = lb; i <= ub; i++)
            {
                x_new[i] = 0;
                for (int j = 0; j < n; j++)
                {
                    x_new[i] += A[i][j] * x[j];
                }
                x_new[i] -= b[i];
                err_chisl[i] = pow(x_new[i], 2);
                x_new[i] = x[i] - tet * x_new[i];
#pragma omp atomic
                chisl += err_chisl[i];
            }
        }
        err = sqrt(chisl) / sqrt(b_znam);
        if ((loss_prev - err) < 0.0001)
        {
            tet = tet * -1;
        }
        loss_prev = err;
        x = x_new;
    }
    return x;
}

int main()
{
    int n = 13600;
    int nm = 40;
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 1.0));

#pragma omp parallel for num_threads(nm)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
            {
                A[i][j] = 2.0;
            }
        }
    }

    std::vector<double> b(n, 1 + n);
    double b_znam = pow(n + 1, 2) * n;
    double tolerance = 0.00001;
    double t1 = omp_get_wtime();
    std::vector<double> solution = simpleIterationMethod(A, b, tolerance, nm, n, b_znam);
    t1 = omp_get_wtime() - t1;
    std::cout << "Решение системы:" << std::endl;
    for (int i = 0; i < solution.size(); ++i)
    {
        std::cout << "x[" << i << "] = " << solution[i] << std::endl;
    }
    std::cout << t1;
    return 0;
}
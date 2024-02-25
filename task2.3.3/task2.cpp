#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <malloc.h>
#include <vector>

std::vector<double> simpleIterationMethod(const std::vector<std::vector<double>> &A, const std::vector<double> &b,
                                          double eps, int nm, double b_znam)
{
    std::vector<double> x(b.size(), 0.0);
    int N = b.size();
    double tet = 0.0001;
    double err = 1;
    while (err > eps)
    {
        std::vector<double> x_new(b.size(), 0.0);

#pragma omp parallel for schedule(dynamic, 17000/20) num_threads(nm)
        for (int i = 0; i < b.size(); i++)
        {
            double tmp = 0;
            double loc_tmp = 0.0;
            for (int j = 0; j < b.size(); j++)
            {
                loc_tmp += A[i][j] * x[j];
            }
#pragma omp atomic
            tmp += loc_tmp;
            x_new[i] = x[i] - tet * (tmp - b[i]);
        }
        double x_tmp = 0.0;
        
#pragma omp parallel for schedule(dynamic, 17000/20) num_threads(nm)
        for (int i = 0; i < b.size(); i++)
        {
            double tmp = 0.0;
            double loc_tmp = 0.0;
            for (int j = 0; j < b.size(); j++)
            {
                loc_tmp += A[i][j] * x_new[j]; 
            }
#pragma omp atomic
            tmp += loc_tmp;
            x_tmp += pow((tmp - b[i]), 2);            
        }
        err = sqrt(x_tmp) / sqrt(b_znam);
        x = x_new;
    }
    // }
    return x;
}

int main()
{
    int n = 17000;
    int nm = 20;
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 1.0));
    double b_znam = pow(n+1, 2)*n;
    std::vector<double> b(n, 1 + n);
    
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
    
    double tolerance = 0.0000001; 
    double t1 = omp_get_wtime();
    std::vector<double> solution = simpleIterationMethod(A, b, tolerance, nm, b_znam);
    t1 = omp_get_wtime() - t1;
    std::cout << "Решение системы:" << std::endl;
    for (int i = 0; i < solution.size(); ++i)
    {
        std::cout << "x[" << i << "] = " << solution[i] << std::endl;
    }
    std::cout << t1;
    return 0;
}
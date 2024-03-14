#include <iostream>
#include <vector>
#include <thread>
#include <omp.h>

void initialize(std::vector<long long>& vector, int startIndex, int endIndex, int n, std::vector<long long>& matrix) {
    for (int i = startIndex; i < endIndex; ++i) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = i + j;
        }
    }
    for (int j = 0; j < n; j++) { 
        vector[j] = j;
    }
}

void multiplication(std::vector<long long>& vector, std::vector<long long>& matrix, std::vector<long long>& result, int startIndex, int endIndex, int n) {
    for (int i = startIndex; i < endIndex; i++) {
        result[i] = 0;
        for (int j = 0; j < n; j++) {
            result[i] += matrix[i * n + j] * vector[j];
        }
    }
}

int main() {
    int n = 40000;
    std::vector<long long> vector(n);
    std::vector<long long> matrix(n * n);
    std::vector<long long> result(n, 0);

    int numThreads = 40;

    int chunkSizeVec = n / numThreads;

    int startIndex = 0;
    std::vector<std::jthread> threads;

    

    for (int i = 0; i < numThreads; ++i) {
        int endIndex = (i == numThreads - 1) ? n : startIndex + chunkSizeVec;
        threads.emplace_back(initialize, std::ref(vector), startIndex, endIndex, n, std::ref(matrix));
        startIndex = endIndex;
    }

    for (auto& thread : threads) {
        thread.join();
    }


    threads.clear();
    startIndex = 0;

    double t;
    t = omp_get_wtime ();
    
    for (int i = 0; i < numThreads; ++i) {
        int endIndex = (i == numThreads - 1) ? n : startIndex + chunkSizeVec;
        threads.emplace_back(multiplication, std::ref(vector), std::ref(matrix), std::ref(result), startIndex, endIndex, n);
        startIndex = endIndex;
    }

    for (auto& thread : threads) { 
        thread.join();
    }
    
    t = omp_get_wtime () - t;

    std::cout << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << result[i] << " " << '\n';
    }
    std::cout << t << '\n';
    std::cout << 13.0131/t << '\n';
    return 0;
}

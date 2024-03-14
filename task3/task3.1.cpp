#include <iostream>
#include <vector>
#include <thread>
#include <time.h>

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
    int n = 20000;
    std::vector<long long> vector(n);
    std::vector<long long> matrix(n * n);
    std::vector<long long> result(n, 0);

    int numThreads = 3;

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
    
    for (int i = 0; i < numThreads; ++i) {
        int endIndex = (i == numThreads - 1) ? n : startIndex + chunkSizeVec;
        threads.emplace_back(multiplication, std::ref(vector), std::ref(matrix), std::ref(result), startIndex, endIndex, n);
        startIndex = endIndex;
    }

    for (auto& thread : threads) { 
        thread.join();
    }

    
    std::cout << std::endl;
    for (int i = 0; i < 20; ++i) {
        std::cout << result[i] << " ";
    }
    

    return 0;
}

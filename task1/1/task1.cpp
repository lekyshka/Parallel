#include <iostream>
#include <cmath>

#ifdef USE_DOUBLE
#define TYPE double
#define FORMAT "double: %.23lf\n"
#else
#define TYPE float
#define FORMAT "float: %.23f\n"
#endif

int size = 10000000;
TYPE arr[10000000];
int main() {
    TYPE sum = 0;
    for (int i = 0; i < size; i++) {
        arr[i] = sin(i * 2 * M_PI / size);
        sum += arr[i];
    }

    // std::cout << sum;
    printf(FORMAT, sum);

    return 0;
}
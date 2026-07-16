#include <stdio.h>
#include <math.h>

int is_prime(long long num) {
    if (num < 2) return 0;
    for (long long i = 2; i <= (long long)sqrt(num); i++) {
        if (num % i == 0) return 0;
    }
    return 1;
}

int main() {
    long long a = 1000000000, b = 10000000000LL;
    for (long long num = a; num <= b; num++) {
        if (is_prime(num)) {
            printf("%lld, ", num);
        }
    }
    return 0;
}





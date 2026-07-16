import numpy as np

def findprimesfrom(a, b):
    totnumbers = list(range(a, b+1))
    primes = []
    for num in totnumbers:
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                break
        else:
            primes.append(num)
    return primes


def sieve(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False

    
    return [i for i, prime in enumerate(is_prime) if prime]


print(sieve(10000000))  

#print(findprimesfrom(1000, 10000))






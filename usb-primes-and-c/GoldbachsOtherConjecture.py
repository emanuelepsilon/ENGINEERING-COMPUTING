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

def primecheck(num):
    for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                 return False
    return True

def goldbach(n):
    for i in range(2, n):
        if primecheck(i) == True:
            diff = n - i
            if diff % 2 == 0:
                 b_sqrd = diff / 2
                 b = int(b_sqrd**0.5)
                 if b_sqrd == b**2 and b > 0:
                      return True
    return False


          
     
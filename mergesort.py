#!/bin/python3

def merge(a,b):
    """ Function to merge two arrays """
    c = []
    n = 0
    while len(a) != 0 and len(b) != 0:
        if a[0] < b[0]:
            c.append(a[0])
            a.remove(a[0])
        else:
            c.append(b[0])
            b.remove(b[0])
            n = n + 1
    if len(a) == 0:
        c += b
    else:
        c += a
    return c,n

# Code for merge sort

def mergesort(x):
    """ Function to sort an array using merge sort algorithm """
    n1=n2=n3=0
    if len(x) == 0 or len(x) == 1:
        return x,0
    else:
        middle = len(x)/2
        a,n1 = mergesort(x[:middle])
        b,n2 = mergesort(x[middle:])
        c,n3 = merge(a,b)
        n = n1+n2+n3
        return c, n

if __name__ == "__main__":
    print("Insert the array for Mergesort:")
    n = int(input())
    arr = []
    arr_i = 0
    for arr_i in range(n):
        arr_t = int(input())
        arr.append(arr_t)
    result,nn = mergesort(arr)
    print (" ".join(map(str, result)))
    print("Inversions: " + str(nn))
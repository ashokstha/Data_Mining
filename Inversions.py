#!/bin/python3


def merge(arr_b, arr_c):
    arr_a = []
    n = 0
    i = j = 0
    while i < len(arr_b) and j < len(arr_c):
        if arr_b[i]<arr_c[j]:
            arr_a.append(arr_b[i])
            i = i+1
        else:
            arr_a.append(arr_c[j])
            j = j + 1
            n = n + len(arr_b) - i

    while i<len(arr_b):
        arr_a.append(arr_b[i])
        i = i + 1

    while j<len(arr_c):
        arr_a.append(arr_c[j])
        j = j + 1

    return arr_a, n


def inversions(arr):
    n = len(arr)
    if n == 1:
        return arr,0
    else:
        arr_b, num1 = inversions(arr[:n//2])
        arr_c, num2 = inversions(arr[n//2:])
        arr_a, num3 = merge(arr_b, arr_c)
        num_r = num1 + num2 + num3
        return arr_a, num_r


if __name__ == "__main__":
    print("Insert the array:")
    n = int(input())
    arr = []
    arr_i = 0
    for arr_i in range(n):
        arr_t = int(input())
        arr.append(arr_t)
    result, num = inversions(arr)
    print (" ".join(map(str, result)))
    print("Inversions: " + str(num))

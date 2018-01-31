#!/bin/python3


def insertion_sort(arr, i, n):
    if i > n-1:
        return arr
    else:
        key = arr[i]
        j = i-1
        while j >= 0 and arr[j] > key:
            arr[j+1] = arr[j]
            j = j-1

        arr[j+1] = key
        return insertion_sort(arr, i+1, n)


if __name__ == "__main__":
    print("Insert the array:")
    n = int(input())
    arr = []
    arr_i = 0
    for arr_i in range(n):
       arr_t = int(input())
       arr.append(arr_t)
    result = insertion_sort(arr, 1, len(arr))
    print (" ".join(map(str, result)))